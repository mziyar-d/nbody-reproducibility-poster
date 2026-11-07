"""
Poster-ready figures + a better statistical comparison.
Usage (Colab / terminal, inside the folder that holds the run folder):
    RUNS=final python3 analysis_v2.py
Optional env vars: LATE_START (default 70), BG (figure background colour), FIG (output folder)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.environ.get("RUNS", "runs"); FIG = os.environ.get("FIG", "figures_v2")
LATE_START = float(os.environ.get("LATE_START", 70)); BG = os.environ.get("BG", "#fdebd3")
os.makedirs(FIG, exist_ok=True)
TEAL, ORANGE, DARK, GREY = "#0f6b73", "#e8742a", "#14232b", "#8a9ba3"
plt.rcParams.update({"font.size": 20, "axes.labelsize": 22, "axes.titlesize": 24, "legend.fontsize": 17,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK,
                     "axes.edgecolor": DARK, "axes.linewidth": 1.6})

def load(name):
    d = np.load(os.path.join(RUNS, f"{name}.npz")); return {k: d[k] for k in d.files}

def radii(run):
    x = run["x"]; r = np.linalg.norm(x - x.mean(axis=1, keepdims=True), axis=2); r.sort(axis=1)
    n = r.shape[1]; return np.stack([r[:, int(f * n) - 1] for f in (0.1, 0.5, 0.9)], axis=1)  # (snapshots, 3)

def divergence(a, b):
    d = np.linalg.norm(a["x"] - b["x"], axis=2)
    return np.median(d, axis=1) / radii(a)[0, 1]

def smooth(y, k):
    if k <= 1: return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"); return np.convolve(pad, np.ones(k) / k, mode="valid")

ref = load("ref"); t = ref["t"]; dt_snap = t[1] - t[0]
variants = {"p12": "shift 1e-12", "p10": "shift 1e-10", "p08": "shift 1e-8",
            "permute": "star order permuted", "matmul": "matrix-form forces"}
cols = {"p12": ORANGE, "p10": "#f08c3e", "p08": "#f6b27a", "permute": TEAL, "matmul": "#3b9aa3"}
ls = {"p12": "-", "p10": "-", "p08": "-", "permute": "-", "matmul": (0, (5, 2))}
out = {}

# ---------- Figure 1: divergence ----------
fig, ax = plt.subplots(figsize=(9, 6.5))
D = {}
for k, lab in variants.items():
    D[k] = divergence(ref, load(k))
    m = t > 0
    ax.semilogy(t[m], np.maximum(D[k][m], 1e-17), color=cols[k], ls=ls[k], lw=3.2, label=lab)
    idx = np.where(D[k] > 0.1)[0]; out[f"t_div_{k}"] = float(t[idx[0]]) if len(idx) else None
ax.axhline(0.1, color=GREY, ls="--", lw=1.8); ax.text(t[-1], 0.13, "10% of half-mass radius", ha="right", va="bottom", fontsize=16, color=DARK)
ax.axhline(2.2e-16, color=DARK, ls=":", lw=1.8); ax.text(t[-1], 3.2e-16, "machine precision", ha="right", va="bottom", fontsize=16, color=DARK)
ax.set_ylim(1e-17, 5); ax.set_xlim(0, t[-1])
ax.set_xlabel("time [N-body units]"); ax.set_ylabel("median star-position difference / r_h(0)")
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.45)); fig.tight_layout()
fig.savefig(f"{FIG}/fig1_divergence_v2.png", dpi=300); plt.close(fig)

# divergence speed: how much time does a 100x smaller perturbation buy?
tp = [out["t_div_p08"], out["t_div_p10"], out["t_div_p12"]]
if None not in tp:
    out["time_gain_per_100x_smaller_shift"] = float((tp[2] - tp[0]) / 2.0)
    out["efold_time_from_shifts"] = float(((tp[2] - tp[0]) / 4.0) / np.log(10))   # 4 decades between 1e-8 and 1e-12
out["crossing_time_Nbody"] = 2 * np.sqrt(2)

# ---------- Figure 2: statistics, band centred on the reference ----------
seeds = [ref] + [load(f"seed{s}") for s in range(2, 11)]
R = np.array([radii(r) for r in seeds])            # (10, snapshots, 3)
sig_t = R.std(axis=0, ddof=1)                      # seed-to-seed spread at each time
Rref = radii(ref); Rvar = {k: radii(load(k)) for k in variants}
k_smooth = max(1, int(round(2.0 / dt_snap)))       # 2 time-unit running mean for readability
names = ["10% mass radius", "50% mass radius", "90% mass radius"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
for j, ax in enumerate(axes):
    c = smooth(Rref[:, j], k_smooth); s = smooth(sig_t[:, j], k_smooth)
    ax.fill_between(t, c - s, c + s, color=GREY, alpha=0.45, label="± seed-to-seed spread (10 clusters)")
    ax.plot(t, c, color=DARK, lw=3, label="reference")
    for k in ["p12", "permute", "matmul"]:
        ax.plot(t, smooth(Rvar[k][:, j], k_smooth), color=cols[k], lw=2.2, ls=ls[k], alpha=0.95, label=variants[k])
    ax.set_title(names[j]); ax.set_xlabel("time [N-body units]"); ax.set_xlim(0, t[-1])
axes[0].set_ylabel("radius [N-body units]"); axes[0].legend(frameon=False, loc="upper right", fontsize=14)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_statistics_v2.png", dpi=300); plt.close(fig)

# ---------- Late-time comparison (after every variant has fully diverged) ----------
w = t >= LATE_START
late_ens = np.array([radii(r)[w].mean(axis=0) for r in seeds])      # (10, 3) late-time mean radii per cluster
sig = late_ens.std(axis=0, ddof=1); late_ref = Rref[w].mean(axis=0)
out["late_window_start"] = LATE_START; out["late_seed_sigma_r10_r50_r90"] = sig.tolist()
rows = {}
for k in variants:
    d = np.abs(Rvar[k][w].mean(axis=0) - late_ref) / sig
    rows[k] = d.tolist(); out[f"late_dev_over_sigma_{k}"] = d.tolist()
out["late_dev_over_sigma_max"] = float(np.max(list(rows.values())))
E = np.array([np.abs((r['energy'].sum(1) - r['energy'].sum(1)[0]) / r['energy'].sum(1)[0]).max() for r in [ref] + [load(k) for k in variants] + seeds[1:]])
out["max_energy_error_all_runs"] = float(E.max())
json.dump(out, open(os.path.join(FIG, "summary_v2.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
print("\nlate-window |variant - reference| / sigma_seed  (columns: r10, r50, r90)")
for k, v in rows.items(): print(f"{k:8s}", np.round(v, 2))
