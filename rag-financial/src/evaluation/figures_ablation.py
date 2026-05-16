import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ablation results
pool_sizes = [5, 10, 15, 20, 30, 50]
precision  = [0.678, 0.651, 0.758, 0.728, 0.733, 0.719]
latency    = [2.663, 2.802, 3.560, 4.315, 6.213, 8.247]

# original three pipeline points for context
pipeline_latency   = [1.517, 1.463, 4.315]
pipeline_precision = [0.405, 0.433, 0.728]
pipeline_labels    = ["Fixed baseline", "Semantic baseline", "Alt. (pool=20)"]
pipeline_colors    = ["#888780", "#7F77DD", "#7F77DD"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# LEFT PLOT — precision vs latency curve (ablation)
ax1.plot(latency, precision, color="#1D9E75", linewidth=2,
         marker="o", markersize=7, zorder=5, label="Ablation curve")

# mark optimal point
opt_idx = precision.index(max(precision))
ax1.scatter(latency[opt_idx], precision[opt_idx],
            color="#1D9E75", s=200, zorder=6,
            edgecolors="white", linewidth=2)
ax1.annotate(f"★ Optimal\npool={pool_sizes[opt_idx]}\n({latency[opt_idx]}s, {precision[opt_idx]})",
             (latency[opt_idx], precision[opt_idx]),
             textcoords="offset points", xytext=(15, -30),
             fontsize=9, color="#1D9E75",
             arrowprops=dict(arrowstyle="->", color="#1D9E75"))

# label each point
for x, y, s in zip(latency, precision, pool_sizes):
    ax1.annotate(f"k={s}", (x, y),
                 textcoords="offset points", xytext=(6, 6),
                 fontsize=8, color="#444441")

# add baseline pipelines for context
for lx, ly, label, color in zip(pipeline_latency, pipeline_precision,
                                  pipeline_labels, pipeline_colors):
    ax1.scatter(lx, ly, color=color, s=120, zorder=4,
                marker="D", alpha=0.6)
    ax1.annotate(label, (lx, ly),
                 textcoords="offset points", xytext=(-10, 10),
                 fontsize=8, color=color)

ax1.set_xlabel("Average Latency per Query (seconds)")
ax1.set_ylabel("Context Precision Score (0-1)")
ax1.set_title("Fig. 7a — Precision vs. Latency\n(ablation across pool sizes)",
              fontsize=11, pad=10)
ax1.set_xlim(0.5, 10)
ax1.set_ylim(0.35, 0.85)

# RIGHT PLOT — precision and recall vs pool size
ax2.plot(pool_sizes, precision, color="#1D9E75", linewidth=2,
         marker="o", markersize=6, label="Context Precision")
recall = [0.638, 0.663, 0.713, 0.725, 0.663, 0.663]
ax2.plot(pool_sizes, recall, color="#7F77DD", linewidth=2,
         marker="s", markersize=6, label="Context Recall")

# mark optimal
ax2.axvline(x=15, color="#1D9E75", linestyle="--", alpha=0.5, linewidth=1.5)
ax2.text(15.5, 0.76, "★ pool=15\n(optimal)", fontsize=9, color="#1D9E75")

ax2.set_xlabel("FAISS Retrieval Pool Size (k)")
ax2.set_ylabel("Score (0-1)")
ax2.set_title("Fig. 7b — Precision & Recall vs. Pool Size",
              fontsize=11, pad=10)
ax2.set_xticks(pool_sizes)
ax2.legend(loc="lower right")
ax2.set_ylim(0.5, 0.85)

plt.tight_layout()
plt.savefig("/app/figures/fig4_ablation_curve.png", bbox_inches="tight")
plt.close()
print("saved → /app/figures/fig4_ablation_curve.png")