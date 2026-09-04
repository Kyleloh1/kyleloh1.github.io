"""
Single-panel schematic for "Decentralized Conformal Novelty Detection via
Quantized Model Exchange" (Loh & Xiang, arXiv:2605.08263).

Four agents, each holding a different local null distribution. Definitional
illustration of the setup; no simulated results.

Geometry deliberately matches encoder-decoder.png (1100x564, unframed, white,
large serif labels) so both publication figures read at the same scale.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, ORANGE, GREEN, PURPLE = "#036f78", "#c8650a", "#1a8b6f", "#7c2f90"
AGENT = [TEAL, ORANGE, GREEN, PURPLE]

rng = np.random.default_rng(11)

# heterogeneous nulls: different means, spreads and orientations
NULLS = [
    (np.array([-3.0,  0.04]), 0.58 * np.array([[0.30,  0.19], [0.19, 0.24]])),
    (np.array([-1.0, -0.08]), 0.58 * np.array([[0.17, -0.10], [-0.10, 0.38]])),
    (np.array([ 1.0,  0.08]), 0.58 * np.array([[0.36, -0.02], [-0.02, 0.14]])),
    (np.array([ 3.0, -0.04]), 0.58 * np.array([[0.21,  0.16], [0.16, 0.31]])),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Georgia", "DejaVu Serif"],
    "text.color": "#191817",
})

# 1100 x 564 at dpi 200, to match encoder-decoder.png
fig, ax = plt.subplots(figsize=(5.5, 2.82), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for k, (mu, cov) in enumerate(NULLS):
    X = rng.multivariate_normal(mu, cov, 130)
    ax.scatter(*X.T, s=10.5, c=AGENT[k], alpha=.72, linewidths=0, rasterized=True)
    ax.text(mu[0], 1.52, f"agent {k+1}", ha="center", fontsize=13.5, color=AGENT[k])

# unframed, like the reference figure
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-4.25, 4.25)
ax.set_ylim(-1.60, 2.20)

fig.tight_layout(pad=.15)
out = "decentralized-heterogeneous-nulls.png"
fig.savefig(out, facecolor="white")
print("  wrote", out)

from PIL import Image
print("  size:", Image.open(out).size, " reference:", Image.open(
    "/Users/kyleloh/build/build/Kyle-Loh-website/src/assets/encoder-decoder.png").size)
