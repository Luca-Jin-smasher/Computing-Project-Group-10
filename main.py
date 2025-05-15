import STOM_higgs_tools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as pyplot
from scipy.optimize import curve_fit

def get_B_expectation(xs, A, lamb):
    return A * np.exp(-xs / lamb)

vals = STOM_higgs_tools.generate_data(n_signals=400)


plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 400
})


fig, ax = plt.subplots(figsize=(6, 4))


bin_height, bin_edges = np.histogram(vals, range=[104.0, 155.0], bins=30)


mean = (bin_edges[:-1] + bin_edges[1:]) / 2
ystd = np.sqrt(bin_height)
xstd = (-bin_edges[:-1] + bin_edges[1:])/2


ax.errorbar(
    mean, bin_height, yerr=ystd, xerr=xstd,
    fmt='o', markersize=3, color='black',
    ecolor='black', elinewidth=1, capsize=2
)

mean_background = np.concatenate((mean[:11], mean[16:]), axis=0)
bin_height_background = np.concatenate((bin_height[:11], bin_height[16:]), axis=0)

par, cov = curve_fit(get_B_expectation, mean_background, bin_height_background, p0 = [1800, 30])

x = np.linspace(bin_edges[0], bin_edges[-1])

plt.plot(x, get_B_expectation(x, par[0], par[1]))

ax.set_xlim(104, 155)
ax.set_xlabel("Rest mass/GeV")
ax.set_ylabel("Number of entries")

plt.tight_layout()

plt.savefig("histogram.png", bbox_inches='tight')
plt.show()

print(par)
