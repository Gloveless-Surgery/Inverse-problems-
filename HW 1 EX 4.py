import numpy as np  
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-(x**2)/(2)) / np.sqrt(2*np.pi)

def cond2(k):  # 2-norm condition number
    s = np.linalg.svd(k, compute_uv=False)
    return np.inf if s[-1] == 0 else s[0] / s[-1]

def condition_number(N):
    grid = np.linspace(0.0, 1.0, N)
    midpoints = (grid[:-1] + grid[1:]) / 2
    x_i = midpoints[:, None]    # shape (N,1)
    x_j = midpoints[None, :]    # shape (1,N)
    k = f(x_i - x_j)  # shape (N,N)
    return cond2(k)

N_values = np.arange(10, 201, 10)  # from 10 to 200 in steps of 10
cond_values = [condition_number(N) for N in N_values]

# --- plot ---
plt.figure(figsize=(7,5))
plt.plot(N_values, cond_values, marker='o')
plt.xlabel("Number of subintervals $N$")
plt.ylabel("Condition number $\\kappa_2(A)$")
plt.title("Condition number vs. mesh size")
plt.yscale("log")   # condition numbers grow quickly, log scale helps
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()