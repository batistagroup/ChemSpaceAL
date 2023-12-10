import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml  # type:ignore
import os
from typing import List, Callable, Tuple
import pprint
import numpy as np

pp = pprint.PrettyPrinter(indent=4, width=100, compact=False)
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.loaders

prepare_generated_fnames = tools.loaders.setup_generations_fname_generator(
    presuffix="temp1.0_metrics", filters=True
)

metrics_to_plot = {
    "validity": "Validity",
    "% unique (rel. to generated)": "Uniqueness",
    "% novelty (rel. to train+AL sets)": "Novelty",
}


def read_metrics(file_path):
    metrics = {}
    with open(file_path, "r") as f:
        for line in f:
            if "*" in line:
                key, value = line.strip().split("= ")
                metrics[key.split(": ")[0]] = float(value)
            elif ":" in line:
                key, value = line.strip().split(": ")
                metrics[key] = float(value)
    return metrics


def plot_model_quality_scatters(
    traces: List,
    shapes: List,
    n_rows: int,
    n_cols: int,
    subplot_titles: List[str],
):
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
    )
    for trace, row, col in traces:
        fig.add_trace(trace, row=row, col=col)
    for shape, row, col in shapes:
        fig.add_shape(shape, row=row, col=col)
    fig.update_xaxes(title_text="Model Names", row=1, col=1)
    fig.update_yaxes(title_text="File Counts", row=1, col=1)
    return fig


def prepare_unique_valid_novel_scatters(
    generation_path: str,
    metrics_list: List[str],
):
    xVals = [f"Iteration {i}" for i in range(6)]
    random_names = ["model7_baseline"] + [f"model7_random_al{i}" for i in range(1, 6)]
    channel_names = lambda channel: ["model7_baseline"] + [
        f"model7_mix100_{channel}_al{i}" for i in range(1, 6)
    ]
    channels = ["random", "diffusion", "linear", "softdiv", "softsub"]
    name_sets = [random_names, *(channel_names(channel) for channel in channels[1:])]

    col_counter = 1
    traces, shapes = [], []
    colors = ["#252323", "#43AA8B", "#4895ef", "#560bad", "#f72585"]
    for metric in metrics_list:
        for i, name_set in enumerate(name_sets):
            yVals = []
            for name in name_set:
                file_path = generation_path + name + "_temp1.0_metrics.txt"
                metrics = read_metrics(file_path)
                yVals.append(metrics.get(metric, 0))
            traces.append(
                (
                    go.Scatter(
                        x=xVals,
                        y=yVals,
                        mode="markers+lines",
                        marker=dict(color=colors[i]),
                        name=channels[i],
                        showlegend=col_counter == 1,
                    ),
                    1,
                    col_counter,
                )
            )
            line_y = 100
            shapes.append(
                (
                    dict(
                        type="line",
                        x0=xVals[0],
                        y0=line_y,
                        x1=xVals[-1],
                        y1=line_y,
                        line=dict(color="black", width=3, dash="dash"),
                    ),
                    1,
                    col_counter,
                )
            )
        col_counter += 1
    return traces, shapes


def create_unique_valid_novel_figure(generation_path: str):
    traces, shapes = prepare_unique_valid_novel_scatters(
        generation_path,
        metrics_list=[
            "validity",
            "% unique (rel. to generated)",
            "% novelty (rel. to train+AL sets)",
        ],
    )
    return plot_model_quality_scatters(
        traces,
        shapes,
        n_rows=1,
        n_cols=3,
        subplot_titles=["Validity", "Uniqueness", "Novelty"],
    )


def create_table_trace(values, extra_params=None, showtext=None):
    if extra_params is None:
        extra_params = dict()
    values = values[::-1, :]
    if showtext is not None:
        extra_params.update(dict(text=values.astype(str), texttemplate=showtext))
    return go.Heatmap(
        x=[f"Iter. {i}" for i in range(1, 6)],
        y=[f"Iter. {i}" for i in range(5)][::-1],
        z=values,
        **extra_params,
    )


def prepare_nofilters_fnames(channel: str) -> List[str]:
    prefix = "" if channel == "random" else "mix100_"
    return [f"model7_{prefix}{channel}_al{i}_temp1.0_metrics" for i in range(1, 6)]


def prepare_num_generations(
    path: str,
    fnames: List[str],
)->go.Scatter:
    num_generated:List[int] = []
    for name in fnames:
        file_path = path + name + ".txt"
        metrics = read_metrics(file_path)
        num_generated.append(int(metrics["generated"]))
    return num_generated


def prepare_quality_table(
    generations_path: str,
    fnames: List[str],
    metric_functions: List[Callable],
    n_rows: int,
    n_cols: int,
):
    xVals = [f"Iteration {i}" for i in range(6)]

    traces = []
    white = dict(colorscale="gray_r", zmin=0, zmax=100, showscale=False)
    regular = lambda i: dict(coloraxis=f"coloraxis{i}")
    for outer_i, metric_function in enumerate(metric_functions):
        qualities = np.empty([5, 5], dtype=object)
        for col, name in enumerate(fnames):
            # col +=1
            print(f"load {name=}, {col=}")
            file_path = generations_path + name + ".txt"
            metrics = read_metrics(file_path)
            print(metrics)
            for row in range(col):
                print(metric_function(row))  # metrics[metric_function(col)])
                qualities[row][col - 1] = metrics[metric_function(row)]
        row = outer_i // n_cols + 1
        col = outer_i % n_cols + 1
        traces.append((create_table_trace(np.zeros([5, 5]), white), row, col))
        text = "%{text:.2f}" if row == 1 else "%{text:.1f}"
        traces.append((create_table_trace(qualities, regular(row), text), row, col))
    return traces


def create_memorization_heatmap_figure(
    traces: List,
    n_rows: int,
    n_cols: int,
    subplot_titles: List[str],
):
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
    )
    for trace, row, col in traces:
        fig.add_trace(trace, row=row, col=col)

    for row, col, text in [(1, 1, "a"), (1, 2, "b"), (2, 1, "c"), (2, 2, "d")]:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=-0.06,
            y=1.35,
            xanchor="right",
            yanchor="top",
            text=f"<b>[{text.upper()}]</b>",
            font=dict(size=20, family="Helvetica"),
            showarrow=False,
            row=row,
            col=col,
        )
    return fig
