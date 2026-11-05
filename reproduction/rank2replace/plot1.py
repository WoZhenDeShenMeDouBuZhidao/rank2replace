import numpy as np


def metric(r, name):
    if name == "validation":
        return r["validation"]["selection"]
    if name == "selection":
        return r["test"]["selection"]
    test = r["test"]
    groups = (
        list(test["environment_accuracy"].values())
        if "environment_accuracy" in test
        else [v["accuracy"] for v in test["groups"].values()]
    )
    return float(100 * (np.mean(groups) if name == "average" else min(groups)))


def draw(selected, config, labels, destination):
    METHODS, DATASETS = config["methods"], config["datasets"]
    LABELS, NAMES = labels["labels"], labels["names"]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
    from itertools import product

    font_size = 17
    plot_labels = {
        **{m: label.removesuffix(" | Qwen") for m, label in LABELS.items()},
        "rank2replace": "Rank2Replace",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(20, 8.3 * len(METHODS) / 10), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.015, w_pad=0.025, hspace=0, wspace=0.025)
    grid = fig.add_gridspec(3, 3, height_ratios=(1.6, 1, 1), hspace=0, wspace=0.025)
    macro_ax = fig.add_subplot(grid[0, :])
    axes = np.array(
        [[fig.add_subplot(grid[row, col]) for col in range(3)] for row in (1, 2)]
    )
    styles = (
        ("average", "Average-group accuracy (%)", "o", "#2875aa"),
        ("worst", "Worst-group accuracy (%)", "^", "#c26927"),
    )
    macro_values = {}
    label_rows = []
    dataset_labels = {}
    for m in METHODS:
        complete = all(selected[d][m] is not None for d in DATASETS)
        for name, *_ in styles:
            if not complete:
                continue
            values = np.array(
                [[metric(r, name) for r in selected[d][m]["entries"]] for d in DATASETS]
            )
            seed_macro = np.vstack([values[:3], values[3:].mean(axis=0)]).mean(axis=0)
            macro_values[m, name] = seed_macro
    for ax, d in [(macro_ax, "macro"), *zip(axes.flat, DATASETS)]:
        complete = (
            len(macro_values) == 2 * len(METHODS)
            if d == "macro"
            else all(selected[d][m] is not None for m in METHODS)
        )
        for i, m in enumerate(METHODS):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#f4f5f6", zorder=0)
            if not complete:
                continue
            row_labels = []
            for name, label, marker, color in styles:
                values = (
                    macro_values[m, name]
                    if d == "macro"
                    else [metric(r, name) for r in selected[d][m]["entries"]]
                )
                mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
                highlighted = m == "rank2replace"
                ax.errorbar(
                    mean,
                    i,
                    xerr=std,
                    fmt=marker,
                    color=color,
                    markersize=(9 if name == "average" else 7.5)
                    if highlighted
                    else (7 if name == "average" else 5.5),
                    markerfacecolor="white" if name == "average" else color,
                    markeredgewidth=2.5 if highlighted else 1.5,
                    capsize=4 if highlighted else 3,
                    capthick=3.2 if highlighted else 1.8,
                    linewidth=3.2 if highlighted else 1.8,
                    zorder=4 if highlighted else 3,
                )
                if d == "macro":
                    is_worst = name == "worst"
                    annotation = ax.annotate(
                        f"{mean:.2f}±{std:.2f}",
                        (mean - std if is_worst else mean + std, i),
                        xytext=(-4 if is_worst else 4, 0),
                        textcoords="offset points",
                        ha="right" if is_worst else "left",
                        va="center",
                        fontsize=font_size,
                        fontweight="bold" if highlighted else "normal",
                    )
                    annotation.set_in_layout(False)
                    row_labels.append((annotation, mean, std, is_worst))
            if row_labels:
                label_rows.append((ax, i, row_labels))
        ax.set_yticks(
            range(len(METHODS)), [plot_labels[m] for m in METHODS], fontsize=font_size
        )
        ax.set_ylim(len(METHODS) - 0.5, -0.5)
        ax.set_xlim(20, 90)
        ax.set_xticks(np.arange(20, 91, 10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.grid(axis="x", which="major", color="#dfe2e5", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelleft=False)
        if d == "macro":
            for i, m in enumerate(METHODS):
                ax.text(
                    21,
                    i,
                    plot_labels[m],
                    fontsize=font_size,
                    ha="left",
                    va="center",
                    fontweight="bold" if m == "rank2replace" else "normal",
                )
        ax.tick_params(
            axis="x",
            which="both",
            labelsize=font_size,
            pad=2,
            labelbottom=d == "macro",
            bottom=d == "macro",
        )
        title = "Macro" if d == "macro" else NAMES[d]
        multiline = title.startswith("NICO++ ")
        if multiline:
            title = title.replace("NICO++ ", "NICO++\n", 1)
        dataset_labels[ax] = ax.text(
            31 if d == "macro" else 21,
            0.035,
            title,
            fontsize=20,
            weight="bold",
            linespacing=1,
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0),
        )
    macro_ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=c,
                marker=k,
                linestyle="none",
                label=l,
                markersize=7 if n == "average" else 5.5,
                markeredgewidth=1.5,
                markerfacecolor="white" if n == "average" else c,
            )
            for n, l, k, c in styles
        ],
        loc="upper right",
        bbox_to_anchor=(1, 1),
        borderaxespad=0.05,
        handlelength=1,
        handletextpad=0.4,
        fontsize=font_size,
        frameon=True,
    )
    # Measure labels after layout, then choose sides jointly for each method row.
    fig.set_dpi(160)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    for ax, i, row_labels in label_rows:
        bounds = ax.get_window_extent(renderer)
        obstacles = []
        for _, mean, std, _ in row_labels:
            left = ax.transData.transform((mean - std, i))[0]
            right = ax.transData.transform((mean + std, i))[0]
            obstacles.append((left - 3, right + 3))
        text_obstacles = [
            t.get_window_extent(renderer) for t in ax.texts if t.get_in_layout()
        ]
        if ax.get_legend() is not None:
            text_obstacles.append(ax.get_legend().get_window_extent(renderer))
        candidates = []
        for sides in product((-1, 1), repeat=len(row_labels)):
            boxes = []
            for (annotation, mean, std, _), side in zip(row_labels, sides):
                annotation.xy = (mean + side * std, i)
                annotation.set_position((side * 4, 0))
                annotation.set_ha("right" if side < 0 else "left")
                boxes.append(annotation.get_window_extent(renderer))
            overflow = sum(
                max(0, bounds.x0 + 3 - b.x0) + max(0, b.x1 - bounds.x1 + 3)
                for b in boxes
            )
            overlap = sum(
                max(0, min(b.x1, right) - max(b.x0, left))
                for b in boxes
                for left, right in obstacles
            )
            overlap += max(
                0, min(boxes[0].x1, boxes[1].x1) - max(boxes[0].x0, boxes[1].x0)
            )
            overlap += sum(
                max(0, min(b.x1, t.x1) - max(b.x0, t.x0))
                for b in boxes
                for t in text_obstacles
                if b.overlaps(t)
            )
            flips = sum(
                side != (-1 if is_worst else 1)
                for side, (_, _, _, is_worst) in zip(sides, row_labels)
            )
            candidates.append(((overflow, overlap, flips), sides))
        _, sides = min(candidates)
        for (annotation, mean, std, _), side in zip(row_labels, sides):
            annotation.xy = (mean + side * std, i)
            annotation.set_position((side * 4, 0))
            annotation.set_ha("right" if side < 0 else "left")
    target = destination / "experiment1.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib.transforms import Bbox

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
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        temporary = target.with_suffix(".tmp.pdf")
        fig.savefig(temporary, bbox_inches=bbox)
        temporary.replace(target)
    plt.close(fig)
