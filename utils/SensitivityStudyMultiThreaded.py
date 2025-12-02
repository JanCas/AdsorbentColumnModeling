"""
Sensitivity Analysis Utility Class
Uses SALib for Sobol and Morris methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from SALib.sample import sobol as sobol_sample, morris as morris_sample
from SALib.analyze import sobol, morris


@dataclass
class OATResult:
    """Container for OAT sensitivity analysis results."""
    parameter: str
    base_value: float
    sensitivity_index: float
    values_tested: np.ndarray
    outputs: np.ndarray


class SensitivityAnalyzer:
    """
    Utility class for performing sensitivity analysis on model functions.
    Uses SALib for Sobol and Morris methods.
    
    Usage with function:
        analyzer = SensitivityAnalyzer(
            model_func=my_model,
            parameters={'k': 1.0, 'n': 2.0},
            bounds={'k': (0.1, 10), 'n': (1, 5)}
        )
        
    Usage with class instance:
        analyzer = SensitivityAnalyzer(
            model=my_model_instance,
            method='run',
            parameters={'k': 1.0, 'n': 2.0},
            bounds={'k': (0.1, 10), 'n': (1, 5)}
        )
    """
    
    def __init__(
        self,
        parameters: Dict[str, float],
        bounds: Dict[str, tuple],
        model_func: Optional[Callable] = None,
        model: Optional[object] = None,
        method: str = 'run',
        show_progress: bool = True,
        n_jobs: int = 1,
    ):
        """
        Args:
            parameters: Dict of parameter names to their base/nominal values.
            bounds: Dict of parameter names to (lower, upper) bounds.
            model_func: Function that takes **kwargs and returns a scalar or array.
            model: Class instance containing the model (alternative to model_func).
            method: Method name to call on the model instance (default: 'run').
            show_progress: Show tqdm progress bars.
            n_jobs: Number of parallel workers. 1 for sequential, >1 for parallel (uses threads).
        """
        if model_func is None and model is None:
            raise ValueError("Must provide either model_func or model")
        
        if model is not None:
            if not hasattr(model, method):
                raise ValueError(f"Model has no method '{method}'")
            self.model = model
            self.model_func = getattr(model, method)
        else:
            self.model = None
            self.model_func = model_func
        
        self.parameters = parameters.copy()
        self.bounds = bounds.copy()
        self._param_names = list(parameters.keys())
        self.show_progress = show_progress
        self.n_jobs = n_jobs
        self._validate()
        
        # SALib problem definition
        self.problem = {
            'num_vars': len(self._param_names),
            'names': self._param_names,
            'bounds': [list(bounds[name]) for name in self._param_names],
        }
    
    def _validate(self):
        missing = set(self.parameters.keys()) - set(self.bounds.keys())
        if missing:
            raise ValueError(f"Missing bounds for parameters: {missing}")
    
    def _evaluate(self, params: Dict[str, float]) -> float:
        result = self.model_func(**params)
        if isinstance(result, np.ndarray):
            return np.mean(result)
        return float(result)
    
    def _evaluate_samples(self, X: np.ndarray, desc: str = "Evaluating") -> np.ndarray:
        """Evaluate model for all rows in sample matrix X."""
        n_samples = X.shape[0]
        
        if self.n_jobs == 1:
            # Sequential execution with progress bar
            Y = np.zeros(n_samples)
            iterator = tqdm(enumerate(X), total=n_samples, desc=desc, disable=not self.show_progress)
            for i, row in iterator:
                params = dict(zip(self._param_names, row))
                Y[i] = self._evaluate(params)
        else:
            # Parallel execution - always use threads since processes require picklable functions
            # (nested closures referencing self.model_func can't be pickled)
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                def eval_row(row):
                    params = dict(zip(self._param_names, row))
                    return self._evaluate(params)
                
                Y = list(tqdm(
                    executor.map(eval_row, X),
                    total=n_samples,
                    desc=desc,
                    disable=not self.show_progress
                ))
            Y = np.array(Y)
        
        return Y
    
    # -------------------------------------------------------------------------
    # Analysis Methods
    # -------------------------------------------------------------------------
    
    def one_at_a_time(
        self,
        n_points: int = 20,
        parameters: Optional[List[str]] = None,
    ) -> Dict[str, OATResult]:
        """
        One-at-a-time (OAT) sensitivity analysis.
        Varies each parameter while holding others at base values.
        """
        params_to_analyze = parameters or self._param_names
        base_output = self._evaluate(self.parameters)
        results = {}
        
        def evaluate_param(param):
            lower, upper = self.bounds[param]
            values = np.linspace(lower, upper, n_points)
            
            if self.n_jobs == 1:
                outputs = np.zeros(n_points)
                for i, val in enumerate(values):
                    test_params = self.parameters.copy()
                    test_params[param] = val
                    outputs[i] = self._evaluate(test_params)
            else:
                def eval_val(val):
                    test_params = self.parameters.copy()
                    test_params[param] = val
                    return self._evaluate(test_params)
                
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    outputs = np.array(list(executor.map(eval_val, values)))
            
            sensitivity = (outputs.max() - outputs.min()) / abs(base_output) if base_output != 0 else 0
            return param, OATResult(
                parameter=param,
                base_value=self.parameters[param],
                sensitivity_index=sensitivity,
                values_tested=values,
                outputs=outputs,
            )
        
        param_iter = tqdm(params_to_analyze, desc="OAT Parameters", disable=not self.show_progress)
        for param in param_iter:
            param_iter.set_postfix(param=param)
            _, result = evaluate_param(param)
            results[param] = result
        
        return results
    
    def local_sensitivity(self, delta: float = 0.01) -> Dict[str, float]:
        """
        Local sensitivity using finite differences.
        Returns normalized partial derivatives at base point.
        """
        base_output = self._evaluate(self.parameters)
        sensitivities = {}
        
        for param, base_val in self.parameters.items():
            perturbed = self.parameters.copy()
            h = base_val * delta if base_val != 0 else delta
            perturbed[param] = base_val + h
            perturbed_output = self._evaluate(perturbed)
            
            if base_output != 0 and base_val != 0:
                sensitivities[param] = ((perturbed_output - base_output) / base_output) / (h / base_val)
            else:
                sensitivities[param] = (perturbed_output - base_output) / h
        
        return sensitivities
    
    def sobol(
        self,
        n_samples: int = 1024,
        calc_second_order: bool = False,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Sobol variance-based global sensitivity analysis using SALib.
        
        Args:
            n_samples: Base sample size.
            calc_second_order: Calculate second-order indices.
            seed: Random seed (sets numpy random state).
            
        Returns:
            SALib results dict with S1, ST, S1_conf, ST_conf, etc.
        """
        if seed is not None:
            np.random.seed(seed)
        X = sobol_sample.sample(self.problem, n_samples, calc_second_order=calc_second_order)
        Y = self._evaluate_samples(X, desc="Sobol")
        return sobol.analyze(self.problem, Y, calc_second_order=calc_second_order)
    
    def morris(
        self,
        n_trajectories: int = 10,
        n_levels: int = 4,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Morris method (Elementary Effects) using SALib.
        
        Args:
            n_trajectories: Number of trajectories.
            n_levels: Number of grid levels.
            seed: Random seed.
            
        Returns:
            SALib results dict with mu, mu_star, sigma, mu_star_conf.
        """
        X = morris_sample.sample(self.problem, n_trajectories, num_levels=n_levels, seed=seed)
        Y = self._evaluate_samples(X, desc="Morris")
        return morris.analyze(self.problem, X, Y)
    
    # -------------------------------------------------------------------------
    # Plotting Methods
    # -------------------------------------------------------------------------
    
    @staticmethod
    def plot_tornado(
        results: Dict[str, OATResult],
        figsize: Tuple[float, float] = (8, 5),
        title: str = 'Parameter Sensitivity',
        color: str = '#2563eb',
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Tornado/bar chart of OAT sensitivity indices."""
        sorted_results = sorted(results.items(), key=lambda x: x[1].sensitivity_index)
        names = [r[0] for r in sorted_results]
        values = [r[1].sensitivity_index for r in sorted_results]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        ax.barh(names, values, color=color)
        ax.set_xlabel('Sensitivity Index')
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_oat_curves(
        results: Dict[str, OATResult],
        figsize: Tuple[float, float] = None,
        ncols: int = 3,
        title: str = 'One-at-a-Time Response Curves',
    ) -> plt.Figure:
        """Grid of line plots showing output vs parameter value."""
        n_params = len(results)
        nrows = (n_params + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_2d(axes).flatten()
        
        for idx, (name, res) in enumerate(results.items()):
            ax = axes[idx]
            ax.plot(res.values_tested, res.outputs, '-o', markersize=3, linewidth=1.5)
            ax.axvline(res.base_value, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel(name)
            ax.set_ylabel('Output')
            ax.set_title(f'{name} (S={res.sensitivity_index:.2f})')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_local_sensitivity(
        sensitivities: Dict[str, float],
        figsize: Tuple[float, float] = (8, 5),
        title: str = 'Local Sensitivity (Elasticity)',
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Bar chart of local sensitivities."""
        sorted_items = sorted(sensitivities.items(), key=lambda x: abs(x[1]))
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        colors = ['#dc2626' if v < 0 else '#2563eb' for v in values]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        ax.barh(names, values, color=colors)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Normalized Sensitivity')
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig
    
    def plot_sobol(
        self,
        results: Dict,
        figsize: Tuple[float, float] = (10, 5),
        title: str = 'Sobol Sensitivity Indices',
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Bar chart comparing S1 and ST indices from SALib results."""
        names = self._param_names
        n = len(names)
        x = np.arange(n)
        width = 0.35
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        ax.bar(x - width/2, results['S1'], width, yerr=results['S1_conf'], 
               label='S1 (First-order)', color='#2563eb', capsize=3)
        ax.bar(x + width/2, results['ST'], width, yerr=results['ST_conf'], 
               label='ST (Total-order)', color='#dc2626', capsize=3)
        
        ax.set_ylabel('Sensitivity Index')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title)
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig
    
    def plot_morris(
        self,
        results: Dict,
        figsize: Tuple[float, float] = (8, 6),
        title: str = 'Morris Screening (μ* vs σ)',
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Scatter plot of μ* vs σ for Morris method."""
        names = self._param_names
        mu_star = results['mu_star']
        sigma = results['sigma']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        ax.scatter(mu_star, sigma, s=100, c='#2563eb', edgecolors='black', linewidth=1)
        
        for i, name in enumerate(names):
            ax.annotate(name, (mu_star[i], sigma[i]), 
                       textcoords='offset points', xytext=(5, 5), fontsize=10)
        
        max_val = max(max(mu_star), max(sigma)) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='σ = μ*')
        
        ax.set_xlabel('μ* (Mean of |Elementary Effects|)')
        ax.set_ylabel('σ (Std Dev of Elementary Effects)')
        ax.set_title(title)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        plt.tight_layout()
        return fig
    
    def plot_morris_bars(
        self,
        results: Dict,
        figsize: Tuple[float, float] = (8, 5),
        title: str = 'Morris Elementary Effects',
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Bar chart of μ* values with σ error bars."""
        names = self._param_names
        mu_star = results['mu_star']
        sigma = results['sigma']
        
        sorted_idx = np.argsort(mu_star)
        names_sorted = [names[i] for i in sorted_idx]
        mu_star_sorted = mu_star[sorted_idx]
        sigma_sorted = sigma[sorted_idx]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        ax.barh(names_sorted, mu_star_sorted, xerr=sigma_sorted, color='#2563eb', capsize=3)
        ax.set_xlabel('μ* (± σ)')
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    @staticmethod
    def to_dataframe(results: Dict[str, OATResult]):
        """Convert OAT results to pandas DataFrame."""
        import pandas as pd
        data = [{
            'parameter': name,
            'base_value': res.base_value,
            'sensitivity_index': res.sensitivity_index,
            'output_range': (res.outputs.min(), res.outputs.max()),
        } for name, res in results.items()]
        return pd.DataFrame(data).sort_values('sensitivity_index', ascending=False)
    
    def sobol_to_dataframe(self, results: Dict):
        """Convert Sobol results to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame({
            'parameter': self._param_names,
            'S1': results['S1'],
            'S1_conf': results['S1_conf'],
            'ST': results['ST'],
            'ST_conf': results['ST_conf'],
        }).sort_values('ST', ascending=False)
    
    def morris_to_dataframe(self, results: Dict):
        """Convert Morris results to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame({
            'parameter': self._param_names,
            'mu': results['mu'],
            'mu_star': results['mu_star'],
            'sigma': results['sigma'],
            'mu_star_conf': results['mu_star_conf'],
        }).sort_values('mu_star', ascending=False)


# -----------------------------------------------------------------------------
# Test / Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    
    class AdsorptionModel:
        def __init__(self, column_length: float):
            self.column_length = column_length
        
        def run(self, k, n, D, q_max):
            x = np.linspace(0, self.column_length, 100)
            return np.mean(q_max * k * x**n * np.exp(-D * x))
    
    model = AdsorptionModel(column_length=10.0)
    
    # Sequential
    print("=== Sequential ===")
    analyzer = SensitivityAnalyzer(
        model=model,
        method='run',
        parameters={'k': 1.0, 'n': 0.5, 'D': 0.1, 'q_max': 5.0},
        bounds={'k': (0.1, 5), 'n': (0.1, 2), 'D': (0.01, 1), 'q_max': (1, 10)},
        n_jobs=1,
    )
    
    start = time.time()
    sobol_res = analyzer.sobol(n_samples=256, seed=42)
    print(f"Sobol (sequential): {time.time() - start:.2f}s")
    
    # Parallel (threads)
    print(f"\n=== Parallel (n_jobs=4) ===")
    analyzer_parallel = SensitivityAnalyzer(
        model=model,
        method='run',
        parameters={'k': 1.0, 'n': 0.5, 'D': 0.1, 'q_max': 5.0},
        bounds={'k': (0.1, 5), 'n': (0.1, 2), 'D': (0.01, 1), 'q_max': (1, 10)},
        n_jobs=4,
    )
    
    start = time.time()
    sobol_res = analyzer_parallel.sobol(n_samples=256, seed=42)
    print(f"Sobol (parallel): {time.time() - start:.2f}s")
    print(analyzer_parallel.sobol_to_dataframe(sobol_res).to_string(index=False))
    
    # Morris
    print("\n=== Morris ===")
    start = time.time()
    morris_res = analyzer_parallel.morris(n_trajectories=20, seed=42)
    print(f"Morris: {time.time() - start:.2f}s")
    print(analyzer_parallel.morris_to_dataframe(morris_res).to_string(index=False))
    
    # Save plots
    analyzer_parallel.plot_sobol(sobol_res).savefig('sobol.png', dpi=150)
    analyzer_parallel.plot_morris(morris_res).savefig('morris_scatter.png', dpi=150)
    
    print("\nPlots saved.")