import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

class DLEColumnProblem(Problem):
    def __init__(self, Q_min=0.005):  # 5 L/s = 18 m³/h
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=2,
            xl=np.array([0.5, 0.01]),     # L: 0.5m, u: 10 mm/s
            xu=np.array([5.0, 0.10]),     # L: 5m,   u: 100 mm/s
        )
        # Kinetic parameters
        self.k = 0.05          # rate constant [1/s] (~3 min⁻¹)
        
        # Sorbent bed properties
        self.d_p = 0.001       # particle diameter [m] (1 mm LMO granules)
        self.eps = 0.35        # void fraction
        
        # Brine properties (geothermal, ~80°C)
        self.mu = 0.35e-3      # viscosity [Pa·s] at 80°C
        self.rho = 1050        # density [kg/m³] (saline brine)
        
        # Column geometry
        self.D = 0.5           # diameter [m]
        self.Q_min = Q_min     # minimum throughput [m³/s]
        
        # Residence time limits
        self.tau_min = 600     # 10 min minimum
        self.tau_max = 1800    # 30 min maximum
    
    def _evaluate(self, X, out, *args, **kwargs):
        L, u = X[:, 0], X[:, 1]
        
        # Damköhler (dimensionless rate × residence time)
        Da = self.k * L / u
        
        # Ergun pressure drop
        term1 = 150 * self.mu * u * (1 - self.eps)**2 / (self.d_p**2 * self.eps**3)
        term2 = 1.75 * self.rho * u**2 * (1 - self.eps) / (self.d_p * self.eps**3)
        dP = L * (term1 + term2)
        
        out["F"] = np.column_stack([-Da, dP])  # maximize Da, minimize dP
        
        # Constraints
        A = np.pi * self.D**2 / 4
        Q = A * u
        tau = L / u  # residence time [s]
        
        # g <= 0 is feasible
        g1 = self.Q_min - Q           # Q >= Q_min
        g2 = tau - self.tau_max       # tau <= 30 min (don't want too slow)
        
        out["G"] = np.column_stack([g1, g2])


# Solve
problem = DLEColumnProblem(Q_min=0.005)  # 5 L/s ≈ 18 m³/h

algorithm = NSGA2(pop_size=100)
result = minimize(problem, algorithm, ('n_gen', 150), seed=42, verbose=False)

F = result.F.copy()
F[:, 0] *= -1  # back to Da
X = result.X

# Sort
idx = np.argsort(F[:, 0])
F, X = F[idx], X[idx]

# Derived quantities
A = np.pi * problem.D**2 / 4
Q = A * X[:, 1]
tau = X[:, 0] / X[:, 1]
EBCT = tau / 60  # min

print(f"Column D = {problem.D*100:.0f} cm, Q_min = {problem.Q_min*1000:.1f} L/s")
print(f"\n{'Da':>6} {'ΔP [kPa]':>10} | {'L [m]':>6} {'u [mm/s]':>8} {'τ [min]':>8} {'Q [L/s]':>8}")
print("-" * 62)
for i in range(0, len(F), max(1, len(F)//10)):
    print(f"{F[i,0]:6.1f} {F[i,1]/1000:10.2f} | {X[i,0]:6.2f} {X[i,1]*1000:8.1f} {tau[i]/60:8.1f} {Q[i]*1000:8.2f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sc = axes[0].scatter(F[:, 0], F[:, 1]/1000, c=EBCT, cmap='viridis')
axes[0].set_xlabel('Da [-]')
axes[0].set_ylabel('ΔP [kPa]')
axes[0].set_title('Pareto Front')
plt.colorbar(sc, ax=axes[0], label='EBCT [min]')

axes[1].scatter(X[:, 0], X[:, 1]*1000, c=F[:, 0], cmap='plasma')
axes[1].set_xlabel('L [m]')
axes[1].set_ylabel('u [mm/s]')
axes[1].set_title('Design Space')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='Da [-]')

axes[2].scatter(F[:, 0], EBCT, c=F[:, 1]/1000, cmap='coolwarm')
axes[2].axhline(10, color='g', linestyle='--', alpha=0.5, label='τ_min = 10 min')
axes[2].axhline(30, color='r', linestyle='--', alpha=0.5, label='τ_max = 30 min')
axes[2].set_xlabel('Da [-]')
axes[2].set_ylabel('EBCT [min]')
axes[2].set_title('Residence Time')
axes[2].legend()
plt.colorbar(axes[2].collections[0], ax=axes[2], label='ΔP [kPa]')

plt.tight_layout()
plt.show()