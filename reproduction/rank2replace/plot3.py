import numpy as np

VLM_NAMES = {
    "qwen": "Qwen3-VL",
    "ministral": "Ministral 3",
    "gemma": "Gemma 4",
    "internvl": "InternVL3.5",
    "minicpm": "MiniCPM-V4.5",
}
METRICS = ("average", "worst")


def save_figure(fig, name, destination):
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    fig.set_dpi(180)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    for text in fig.findobj(match=lambda artist: isinstance(artist, plt.Text)):
        if not text.get_visible() or not text.get_text():
            continue
        bbox = text.get_window_extent(renderer)
        assert (
            bbox.x0 >= -1
            and bbox.x1 <= width + 1
            and bbox.y0 >= -1
            and bbox.y1 <= height + 1
        ), text.get_text()
    pixels = np.asarray(fig.canvas.buffer_rgba())
    rows, cols = np.nonzero(np.any(pixels[:, :, :3] < 250, axis=2))
    bbox = Bbox.from_extents(
        cols.min() / 180,
        (pixels.shape[0] - rows.max() - 1) / 180,
        (cols.max() + 1) / 180,
        (pixels.shape[0] - rows.min()) / 180,
    )
    fig.set_layout_engine("none")
    path = destination / f"{name}.pdf"
    temporary = path.with_suffix(".tmp.pdf")
    fig.savefig(temporary, dpi=180, bbox_inches=bbox)
    temporary.replace(path)
    plt.close(fig)


def draw_downstream(values, methods, qualities, destination):
    METHODS, QUALITIES = methods, qualities
    VLMS = ("ministral", "gemma", "internvl", "minicpm")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(
        figsize=(11, 3.4 + 0.4 * (len(METHODS) - 3)), layout="constrained"
    )
    from .plot_style import METHOD_COLORS

    colors = METHOD_COLORS
    labels = {
        "irm": "IRM",
        "group_dro": "Group DRO",
        "coral": "CORAL",
        "scer": "SCER",
        "rank2replace": "Rank2Replace",
    }
    markers = {"average": "o", "worst": "^"}
    x = np.r_[np.arange(6) * 0.9, np.arange(5) * 1.15 + 5.65]
    ticklabels = ["Oracle", "10%", "20%", "30%", "40%", "50%"] + [
        VLM_NAMES[model] for model in ("qwen", *VLMS)
    ]
    for name in METRICS:
        ref = next(r for r in values["reference"] if r["metric"] == name)
        ax.axhspan(
            ref["mean"] - ref["std"],
            ref["mean"] + ref["std"],
            color=colors["rank2replace"],
            alpha=0.10,
            zorder=0,
        )
        ax.axhline(ref["mean"], color=colors["rank2replace"], ls="--", lw=1.5)
        ax.plot(
            x,
            np.full(len(x), ref["mean"]),
            color=colors["rank2replace"],
            marker=markers[name],
            markersize=4.5,
            ls="none",
        )
        for i, method in enumerate(METHODS):
            records = [
                next(
                    r
                    for r in values["downstream"]
                    if (r["method"], r["quality"], r["metric"]) == (method, q, name)
                )
                for q in QUALITIES
            ]
            means = np.array([r["mean"] for r in records])
            stds = np.array([r["std"] for r in records])
            position = x + (i - (len(METHODS) - 1) / 2) * 0.14
            ax.errorbar(
                position[:6],
                means[:6],
                yerr=stds[:6],
                color=colors[method],
                marker=markers[name],
                markersize=4.5,
                capsize=2.5,
                lw=1.4,
            )
            ax.errorbar(
                position[6:],
                means[6:],
                yerr=stds[6:],
                color=colors[method],
                marker=markers[name],
                markersize=4.5,
                capsize=2.5,
                ls="none",
                lw=1.2,
            )
    ax.axvline((x[5] + x[6]) / 2, color="#b8bdc2", lw=0.8, ls=":")
    ax.set_xlim(-0.3, x[-1] + 0.45)
    ax.set_ylim(35, 82)
    ax.set_xticks(x, ticklabels, fontsize=11)
    ax.set_yticks(np.arange(35, 81, 5))
    ax.set_ylabel("Macro-average accuracy (%)", fontsize=12, labelpad=6)
    ax.grid(axis="y", color="#e1e4e7", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=4)
    methods = ax.legend(
        [
            Line2D(
                [], [], color=colors[m], lw=2, ls="--" if m == "rank2replace" else "-"
            )
            for m in (*METHODS, "rank2replace")
        ],
        [labels[m] for m in (*METHODS, "rank2replace")],
        loc="lower right",
        bbox_to_anchor=(1, 0.075),
        ncol=len(METHODS) + 1,
        frameon=False,
        fontsize=11,
        columnspacing=0.8,
        handlelength=1.4,
        borderaxespad=0.3,
    )
    ax.add_artist(methods)
    ax.legend(
        [
            Line2D(
                [], [], color="#444444", marker=markers[name], ls="none", markersize=5
            )
            for name in METRICS
        ],
        ["Average-group", "Worst-group"],
        loc="lower right",
        ncol=2,
        frameon=False,
        fontsize=11,
        columnspacing=1.1,
        handlelength=1,
        borderaxespad=0.3,
    )
    save_figure(fig, "experiment3", destination)
