import numpy as np


def metric(result, name):
    if name == "validation":
        return result["validation"]["selection"]
    if name == "selection":
        return result["test"]["selection"]
    test = result["test"]
    groups = (
        list(test["environment_accuracy"].values())
        if "environment_accuracy" in test
        else [v["accuracy"] for v in test["groups"].values()]
    )
    return float(100 * (np.mean(groups) if name == "average" else min(groups)))


def draw(selected, config, labels, destination):
    METHODS, DATASETS = config["methods"], config["datasets"]
    LABELS = labels["labels"]
    ARMS = ("A_raw", "B_rank", "C_rank_corrupted")

    def arms(method):
        return ARMS[1:] if method in ("laat", "hard_removal") else ARMS

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = ("#c3c8cc", "#2875aa", "#d48236")
    labels = ("Original slot order", "+ Slot ranking", "+ Slot corruption")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(
        figsize=(20, 5.6 + 0.52 * (len(METHODS) - 9)), layout="constrained"
    )
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 0.08))
    left = fig.add_subplot(grid[0, 0])
    axes = (left, fig.add_subplot(grid[0, 1], sharey=left))
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.set_axis_off()
    for ax, metric_name, title in zip(
        axes,
        ("average", "worst"),
        ("Average-group accuracy (%)", "Worst-group accuracy (%)"),
    ):
        for i, m in enumerate(METHODS):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#f4f5f6", zorder=0)
            if not all(
                selected[d][m][a] is not None for d in DATASETS for a in arms(m)
            ):
                continue
            means, stds = {}, {}
            for arm in arms(m):
                values = np.array(
                    [
                        [metric(r, metric_name) for r in selected[d][m][arm]["entries"]]
                        for d in DATASETS
                    ]
                )
                seed_macro = np.vstack([values[:3], values[3:].mean(axis=0)]).mean(
                    axis=0
                )
                means[arm], stds[arm] = (
                    float(seed_macro.mean()),
                    float(seed_macro.std(ddof=1)),
                )
            # Each segment adds to the previous setup; the endpoint includes corruption.
            previous = 0.0
            for index, arm in enumerate(ARMS):
                if arm not in means:
                    continue
                delta = means[arm] - previous
                ax.barh(
                    i,
                    delta,
                    left=previous,
                    height=0.58,
                    color=colors[index],
                    edgecolor="white",
                    linewidth=0.6,
                    hatch="///" if delta < 0 else None,
                    zorder=2,
                )
                previous = means[arm]
            first = next(a for a in ARMS if a in means)
            ax.text(
                means[first] - 1,
                i,
                f"{means[first]:.2f}",
                ha="right",
                va="center",
                color="white" if first == ARMS[1] else "#303438",
                fontsize=13,
            )
            last = ARMS[-1]
            ax.errorbar(
                means[last],
                i,
                xerr=stds[last],
                fmt="none",
                ecolor="#363a3e",
                capsize=3,
                linewidth=1,
                zorder=3,
            )
            ax.annotate(
                f"{means[last]:.2f}±{stds[last]:.2f}",
                (means[last] + stds[last], i),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=13,
            )
        ax.set_yticks(
            range(len(METHODS)), [LABELS[m].removesuffix(" | Qwen") for m in METHODS]
        )
        ax.set_ylim(len(METHODS) - 0.5, -0.5)
        ax.set_xlim(20, 100)
        ax.set_xticks(np.arange(20, 101, 10))
        ax.grid(axis="x", color="#dfe2e5", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelleft=ax is axes[0])
        ax.set_title(title, fontsize=16, loc="left", pad=10)
    legend_ax.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, labels)],
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=20,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0,
        borderpad=0,
    )
    for text in fig.findobj(match=plt.Text):
        text.set_fontsize(20)
    for ax in axes:
        ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=22, pad=10)
    path = destination / "experiment2.pdf"
    from matplotlib.transforms import Bbox

    fig.set_dpi(160)
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())
    rows, cols = np.nonzero(np.any(pixels[:, :, :3] < 250, axis=2))
    height = pixels.shape[0]
    bbox = Bbox.from_extents(
        cols.min() / 160,
        (height - rows.max() - 1) / 160,
        (cols.max() + 1) / 160,
        (height - rows.min()) / 160,
    )
    fig.set_layout_engine("none")
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        temp = path.with_suffix(".tmp.pdf")
        fig.savefig(temp, bbox_inches=bbox)
        temp.replace(path)
    plt.close(fig)
