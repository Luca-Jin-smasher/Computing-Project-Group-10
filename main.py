import STOM_higgs_tools
import matplotlib.pyplot as plt
import numpy as np


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


ax.set_xlim(104, 155)
ax.set_xlabel("Rest mass/GeV")
ax.set_ylabel("Number of entries")

plt.tight_layout()

plt.savefig("histogram.png", bbox_inches='tight')
plt.show()

