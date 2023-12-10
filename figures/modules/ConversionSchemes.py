from typing import Dict, Union, Optional, List, Tuple, Callable
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

Number = Union[float, int]


def _preprocess_scores_linearly(
    scores: Dict[int, Number], do_negation: bool = False
) -> Dict[int, Number]:
    sign = -1 if do_negation else 1
    negated = {k: sign * v for k, v in scores.items()}
    total = sum(negated.values())
    normalized = {k: v / total for k, v in negated.items()}
    return normalized


def _preprocess_scores_softmax(
    scores: Dict[int, Number],
    do_negation: bool = False,
    divide: bool = True,
    divide_factor: Optional[float] = None,
) -> Dict[int, Number]:
    sign = -1 if do_negation else 1
    negated = {k: sign * v for k, v in scores.items()}
    max_value = max(negated.values())
    if divide:
        assert (
            divide_factor is not None
        ), "You have to specify a value p in (0, 1). Softmax is computed as e^[x/(p*max)]"
        exponentiate = {
            k: np.exp(v / (divide_factor * max_value)) for k, v in negated.items()
        }
    else:
        exponentiate = {k: np.exp(v - max_value) for k, v in negated.items()}
    total = sum(exponentiate.values())
    softmax = {k: v / total for k, v in exponentiate.items()}
    return softmax


def plot_probability_distribution(
    cluster_scores: np.ndarray,
    args_list: List[Tuple[str, Optional[bool], Optional[float]]],
    bin_size=0.001,
):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["<b>Linear</b>", "<b>Softsub</b>", "<b>Softdiv</b>"],
        horizontal_spacing=0.1,
    )
    for col in range(1, 4):
        fig.add_trace(
            go.Histogram(
                x=[1 / 100 for _ in range(100)],
                name="uniform",
                opacity=0.75,
                histnorm="probability",
                xbins=dict(start=-0.1, end=1, size=bin_size),
                marker_color="black",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )
    keyToFunc: Dict[str, Callable] = {
        "linear": _preprocess_scores_linearly,
        "softmax": _preprocess_scores_softmax,
    }
    col = 1
    colors = ["#023e8a", "#40916c", "#ff8800", "#7209b7", "#f72585"]
    for i, (func, param_switch, param_value) in enumerate(args_list):
        suffix, extras = "", {}
        if func == "softmax":
            suffix = f" divf {param_value}" if param_switch else ""
            extras = dict(divide=param_switch, divide_factor=param_value)
        print(f"Tring to add to {col=}")
        fig.add_trace(
            go.Histogram(
                x=list(keyToFunc[func](cluster_scores, **extras).values()),
                name=f"{func}{suffix}",
                opacity=0.75,
                histnorm="probability",
                xbins=dict(start=-0.1, end=1, size=0.01 if i == 1 else bin_size),
                marker_color=colors[i % len(colors)],
            ),
            row=1,
            col=col,
        )
        col += 1
        col = min(3, col)
    fig.update_layout(
        barmode="overlay",
    )
    for col, text in [(1, "a"), (2, "b"), (3, "c")]:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=-0.02,
            y=1.25,
            xanchor="right",
            yanchor="top",
            text=f"<b>[{text.upper()}]</b>",
            font=dict(size=24, family="Helvetica"),
            showarrow=False,
            row=1,
            col=col,
        )
    fig.update_xaxes(dtick=0.005, range=[0,0.016], row=1, col=1)
    fig.update_yaxes(range=[0, 0.25], row=1, col=1)
    fig.update_yaxes(range=[0, 0.05], row=1, col=2)
    fig.update_xaxes(dtick=0.1, row=1, col=2)
    fig.update_yaxes(range=[0, 0.35], row=1, col=3)
    fig.update_xaxes(dtick=0.01, row=1, col=3)
    return fig
