"""
Split-conformal novelty detection with BH, honestly computed.

Pipeline (Bates, Candes, Lei & Ramdas, "Testing for outliers with conformal p-values"):
  1. Fit a nonconformity score on a TRAIN split of inliers only.
  2. Score a disjoint CALIBRATION split of inliers.
  3. For each test point j:  p_j = (1 + #{i in cal : s_i >= s_j}) / (n_cal + 1)
  4. Apply Benjamini-Hochberg at level alpha to {p_j}.
Under exchangeability of calibration and test inliers these p-values are
marginally super-uniform and PRDS, so BH controls FDR at alpha.
"""
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, ORANGE, GREEN, PURPLE = "#036f78", "#c8650a", "#1a8b6f", "#7c2f90"
GREY, RULE = "#6d675f", "#d9d5cf"

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- distributions
MU = np.array([0.0, 0.0])
COV = np.array([[1.0, 0.72], [0.72, 1.0]])          # correlated Gaussian inliers

def sample_inliers(k):
    return rng.multivariate_normal(MU, COV, size=k)

def sample_novelties(k):
    """Novelties: shifted, rotated, tighter Gaussian -- off the inlier ridge."""
    mu = np.array([2.05, -1.55])
    cov = np.array([[0.30, -0.13], [-0.13, 0.22]])
    return rng.multivariate_normal(mu, cov, size=k)

# ---------------------------------------------------------------- conformal core
def fit_score(train):
    """Nonconformity score: negative log-density under a KDE fit on inliers only."""
    kde = gaussian_kde(train.T)
    return lambda X: -np.log(kde(X.T) + 1e-300)

def conformal_pvalues(score_cal, score_test):
    """p_j = (1 + #{i : s_i >= s_j}) / (n + 1)  -- the standard marginal form."""
    n = len(score_cal)
    cal_sorted = np.sort(score_cal)
    # count of calibration scores >= each test score
    ge = n - np.searchsorted(cal_sorted, score_test, side="left")
    return (1.0 + ge) / (n + 1.0)

def bh(p, alpha):
    """Benjamini-Hochberg step-up. Returns boolean rejection mask and the cutoff."""
    m = len(p)
    order = np.argsort(p)
    ps = p[order]
    thresh = alpha * np.arange(1, m + 1) / m
    below = np.nonzero(ps <= thresh)[0]
    rej = np.zeros(m, dtype=bool)
    if len(below) == 0:
        return rej, 0.0, ps, thresh
    kmax = below[-1]
    rej[order[: kmax + 1]] = True
    return rej, ps[kmax], ps, thresh

def run_trial(alpha, n_train=800, n_cal=500, n_test=400, prop_novel=0.15, seed=None):
    global rng
    if seed is not None:
        rng = np.random.default_rng(seed)
    train = sample_inliers(n_train)
    cal = sample_inliers(n_cal)
    n_nov = int(round(prop_novel * n_test))
    test = np.vstack([sample_inliers(n_test - n_nov), sample_novelties(n_nov)])
    is_novel = np.zeros(n_test, dtype=bool)
    is_novel[n_test - n_nov:] = True
    s = fit_score(train)
    p = conformal_pvalues(s(cal), s(test))
    rej, cutoff, ps, thresh = bh(p, alpha)
    return dict(test=test, is_novel=is_novel, p=p, rej=rej,
                cutoff=cutoff, ps=ps, thresh=thresh, cal=cal)

ALPHA = 0.10
r = run_trial(ALPHA, seed=88)   # median-FDP draw over 200 trials (see note)

fdp = (r["rej"] & ~r["is_novel"]).sum() / max(1, r["rej"].sum())
power = (r["rej"] & r["is_novel"]).sum() / max(1, r["is_novel"].sum())
print(f"  single trial: {r['rej'].sum()} discoveries, FDP={fdp:.3f}, power={power:.3f}, cutoff={r['cutoff']:.4f}")

# ---------------------------------------------------------------- FDR sweep
alphas = np.linspace(0.02, 0.30, 12)
N_TRIALS = 120
mean_fdp, se_fdp = [], []
for a in alphas:
    fdps = []
    for t in range(N_TRIALS):
        rt = run_trial(a, n_train=500, n_cal=400, n_test=300, seed=1000 + t)
        fdps.append((rt["rej"] & ~rt["is_novel"]).sum() / max(1, rt["rej"].sum()))
    fdps = np.array(fdps)
    mean_fdp.append(fdps.mean())
    se_fdp.append(fdps.std(ddof=1) / np.sqrt(N_TRIALS))
mean_fdp, se_fdp = np.array(mean_fdp), np.array(se_fdp)
print("  FDR sweep: max(mean FDP - alpha) =", f"{(mean_fdp - alphas).max():+.4f}")

# ---------------------------------------------------------------- figure
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.2, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "text.color": GREY, "axes.labelcolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY,
})

fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.12), dpi=200)
fig.patch.set_facecolor("white")
for ax in axes.ravel():
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color(RULE)
    ax.tick_params(labelsize=7.7, pad=1.6)

def title(ax, t):
    ax.set_title(t, fontsize=9.0, color="#191817", pad=3.4, loc="left")

# --- (a) the data
ax = axes[0, 0]
tst, nov = r["test"], r["is_novel"]
ax.scatter(*tst[~nov].T, s=3.0, c=TEAL, alpha=.55, linewidths=0, rasterized=True)
ax.scatter(*tst[nov].T, s=4.6, c=ORANGE, alpha=.9, linewidths=0, rasterized=True)
title(ax, "a  inliers and novelties")
ax.set_xticks([]); ax.set_yticks([])
ax.text(.035, .93, "inlier", transform=ax.transAxes, va="top", fontsize=7.7, color=TEAL)
ax.text(.035, .82, "novelty", transform=ax.transAxes, va="top", fontsize=7.7, color=ORANGE)

# --- (b) conformal p-values
ax = axes[0, 1]
bins = np.linspace(0, 1, 26)
ax.hist(r["p"][~nov], bins=bins, color=TEAL, alpha=.75, linewidth=0)
ax.hist(r["p"][nov], bins=bins, color=ORANGE, alpha=.9, linewidth=0)
ax.axhline((~nov).sum() / (len(bins) - 1), color=GREY, lw=.7, ls=(0, (3, 2)))
title(ax, "b  conformal $p$-values")
ax.set_xlabel("$p$", fontsize=8.2, labelpad=1.5)
ax.set_yticks([])
ax.text(.55, .82, "uniform under\nthe null", transform=ax.transAxes, fontsize=7.4, color=GREY)

# --- (c) BH step-up
ax = axes[1, 0]
m = len(r["ps"])
k = np.arange(1, m + 1)
ax.plot(k, r["ps"], color=GREEN, lw=1.0)
ax.plot(k, r["thresh"], color=GREY, lw=.7, ls=(0, (3, 2)))
n_rej = r["rej"].sum()
ax.axvline(n_rej, color=PURPLE, lw=.8)
ax.set_xlim(0, 140); ax.set_ylim(0, .16)
title(ax, "c  Benjamini–Hochberg step-up")
ax.set_xlabel("rank $k$", fontsize=8.2, labelpad=1.5)
ax.set_ylabel("$p_{(k)}$", fontsize=8.2, labelpad=1.5)
ax.text(n_rej + 4, .137, f"reject {n_rej}", fontsize=7.7, color=PURPLE)
ax.text(n_rej + 4, .118, f"FDP = {fdp:.2f}", fontsize=7.4, color=GREY)
ax.text(96, .052, r"$k\alpha/m$", fontsize=7.9, color=GREY)

# --- (d) FDR control
ax = axes[1, 1]
ax.plot([0, .32], [0, .32], color=GREY, lw=.7, ls=(0, (3, 2)))
ax.fill_between(alphas, mean_fdp - 1.96 * se_fdp, mean_fdp + 1.96 * se_fdp,
                color=PURPLE, alpha=.16, linewidth=0)
ax.plot(alphas, mean_fdp, color=PURPLE, lw=1.1, marker="o", ms=2.2, mew=0)
ax.set_xlim(0, .32); ax.set_ylim(0, .32)
ax.set_xticks([0, .1, .2, .3]); ax.set_yticks([0, .1, .2, .3])
title(ax, "d  realised FDR vs nominal level")
ax.set_xlabel(r"nominal $\alpha$", fontsize=8.2, labelpad=1.5)
ax.text(.055, .255, r"FDR $\leq \alpha$", fontsize=8.2, color=PURPLE)
ax.text(.055, .205, r"$\approx \pi_0\alpha$", fontsize=7.4, color=GREY)

fig.tight_layout(pad=.5, w_pad=1.15, h_pad=1.05)
out = "conformal-novelty-fdr-hero.png"
fig.savefig(out, facecolor="white")
print("  wrote", out)
