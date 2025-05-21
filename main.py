import STOM_higgs_tools
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import Max_likelihood

def get_B_expectation(xs, A, lamb):
    return A * np.exp(-xs / lamb)

# Generate data
vals = np.array(STOM_higgs_tools.generate_data(n_signals=400))

# Plot styling
plt.rcParams.update({'font.size': 12, 'font.family': 'serif', 'figure.dpi': 400})
fig, ax = plt.subplots(figsize=(6, 4))

# Histogram
bin_height, bin_edges = np.histogram(vals, range=[104.0, 155.0], bins=30)
mean = (bin_edges[:-1] + bin_edges[1:]) / 2
ystd = np.sqrt(bin_height)
xstd = (bin_edges[1:] - bin_edges[:-1]) / 2

ax.errorbar(mean, bin_height, yerr=ystd, xerr=xstd, fmt='o', markersize=3, color='black')

# =================================================================
# Fix 1: Correct background bin selection using dynamic masking
# =================================================================
mask = (mean < 121.0) | (mean > 129.0)
mean_background = mean[mask]
bin_height_background = bin_height[mask]

# =================================================================
# Fix 2: Proper MLE fit with bounds and status check
# =================================================================
initial_guess = [1800, 30]
result = minimize(
    lambda p: Max_likelihood.negative_log_likelihood(p, mean_background, bin_height_background),
    x0=initial_guess,
    bounds=[(0, None), (0, None)]  # Prevent negative parameters
)

if not result.success:
    raise ValueError(f"MLE failed: {result.message}")

A_mle, lamb_mle = result.x
print(f"MLE Results: A = {A_mle:.1f}, λ = {lamb_mle:.1f}")

# =================================================================
# Fix 3: Adjusted χ² grid search range near MLE results
# =================================================================
A_values = np.linspace(0.5*A_mle, 1.5*A_mle, 50)  # Center around MLE estimate
lamb_values = np.linspace(0.5*lamb_mle, 1.5*lamb_mle, 50)
chi2_grid = np.zeros((len(A_values), len(lamb_values)))

# =================================================================
# Fix 4: Proper χ² calculation using binned background data
# =================================================================
for i, A_trial in enumerate(A_values):
    for j, lamb_trial in enumerate(lamb_values):
        # Calculate expected background
        B_expected = get_B_expectation(mean_background, A_trial, lamb_trial)
        # Poisson χ²: Σ [(observed - expected)² / expected]
        chi2 = np.sum((bin_height_background - B_expected)**2 / B_expected)
        chi2_grid[i, j] = chi2

min_index = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
A_chi2, lamb_chi2 = A_values[min_index[0]], lamb_values[min_index[1]]
print(f"χ² Results: A = {A_chi2:.1f}, λ = {lamb_chi2:.1f}")

# =================================================================
# Plot both fits
# =================================================================
x = np.linspace(104, 155, 500)
ax.plot(x, get_B_expectation(x, A_mle, lamb_mle), label="MLE Fit", color='red')
ax.plot(x, get_B_expectation(x, A_chi2, lamb_chi2), '--', label="χ² Fit", color='blue')

ax.set_xlim(104, 155)
ax.set_xlabel("Rest mass (GeV)")
ax.set_ylabel("Number of entries")
ax.legend()
plt.savefig("histogram.png", bbox_inches='tight')
plt.show()

