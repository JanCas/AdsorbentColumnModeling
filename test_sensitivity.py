#!/usr/bin/env python
"""
Test script for SensitivityAnalyzer with tqdm progress bars and fixed parameters
"""

import numpy as np
from utils.SensitivityStudy import SensitivityAnalyzer

# Simple test function that requires fixed params
def test_model(a, b, c=1.0, scale=1.0):
    """Test model: output = a * b * scale + c"""
    return a * b * scale + c

print("=" * 60)
print("Testing SensitivityAnalyzer with fixed parameters and tqdm")
print("=" * 60)

# Test with fixed parameters
print("\nCreating analyzer with:")
print("  Varying parameters: a, b")
print("  Fixed parameters: c=5.0, scale=2.0")
analyzer = SensitivityAnalyzer(
    model_func=test_model,
    parameters={'a': 1.0, 'b': 2.0},  # Vary these
    bounds={'a': (0.5, 1.5), 'b': (1.0, 3.0)},
    fixed_params={'c': 5.0, 'scale': 2.0}  # Keep these fixed
)

# Test OAT analysis with progress bars
print("\n--- Running OAT Analysis (should show progress bars) ---")
oat_results = analyzer.one_at_a_time(n_points=10)
print("\nOAT Results:")
for name, res in oat_results.items():
    print(f"  {name}: sensitivity = {res.sensitivity_index:.3f}")
    print(f"        base_value = {res.base_value:.3f}")
    print(f"        output range = [{res.outputs.min():.3f}, {res.outputs.max():.3f}]")

# Test local sensitivity with progress bars
print("\n--- Running Local Sensitivity (should show progress bar) ---")
local_results = analyzer.local_sensitivity()
print("\nLocal Sensitivity Results:")
for name, val in local_results.items():
    print(f"  {name}: {val:.3f}")

# Test with a more complex model
print("\n" + "=" * 60)
print("Testing with a more complex model")
print("=" * 60)

def complex_model(k, n, D=0.1, q_max=5.0, temperature=298.15):
    """More complex model for testing"""
    x = np.linspace(0, 10, 50)
    result = np.mean(q_max * k * x**n * np.exp(-D * x))
    result *= (298.15 / temperature)  # temperature correction
    return result

# Create analyzer with both varying and fixed parameters
analyzer2 = SensitivityAnalyzer(
    model_func=complex_model,
    parameters={'k': 1.0, 'n': 0.5},  # Only vary these
    bounds={'k': (0.1, 5), 'n': (0.1, 2)},
    fixed_params={'D': 0.1, 'q_max': 5.0, 'temperature': 300.0}  # Fixed
)

print("\nRunning Morris analysis (should show progress)...")
morris_results = analyzer2.morris(n_trajectories=5, seed=42)
print("\nMorris Results:")
print(f"  Parameters: {analyzer2._param_names}")
print(f"  μ*: {morris_results['mu_star']}")
print(f"  σ: {morris_results['sigma']}")

print("\nRunning Sobol analysis (should show progress)...")
sobol_results = analyzer2.sobol(n_samples=64, seed=42)  # Small n for quick test
print("\nSobol Results:")
print(f"  Parameters: {analyzer2._param_names}")
print(f"  S1: {sobol_results['S1']}")
print(f"  ST: {sobol_results['ST']}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)