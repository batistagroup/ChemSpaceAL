import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.loaders
import tools.projectors

prepare_loader = tools.loaders.prepare_loader


DATASET_INFO = {
    "moses": ("moses_train", "moses_test", "model2"),
    "guacamol": ("guacamol_v1_train", "guacamol_v1_valid", "model3"),
    "combined": (
        "combined_processed_freq1000_block133_train",
        "combined_processed_freq1000_block133_val",
        "model7",
    ),
}


def load_training_smiles(
    datasets_path: str, dataset: str, sample: Optional[int] = None
):
    train_df = pd.read_csv(datasets_path + DATASET_INFO[dataset][0] + ".csv.gz")
    valid_df = pd.read_csv(datasets_path + DATASET_INFO[dataset][1] + ".csv.gz")
    df = pd.concat([train_df, valid_df]).drop_duplicates(subset=["smiles"])
    if sample is not None:
        df = df.sample(sample, replace=False, random_state=42)
    return df["smiles"].to_list()


def load_generated_smiles(
    generations_path: str, dataset: str, sample: Optional[int] = None
):
    df = pd.read_csv(
        generations_path + DATASET_INFO[dataset][2] + "_baseline_temp1.0_processed.csv"
    )
    if sample is not None:
        df = df.sample(sample, replace=False, random_state=42)
    return df["smiles"].to_list()


def reduce_training_and_generations(
    training_projection, generated_projection, pca_path: str, pca_fname: str
):
    reduce = lambda array: tools.projectors.reduce_dataframe(
        array,
        reduction="PCA",
        pca_path=pca_path,
        pca_fname=pca_fname,
        reduction_parameters=dict(reduce_to_2d=True),
    )
    training_reduced = [reduce(df) for df in training_projection]
    generated_reduced = [reduce(df) for df in generated_projection]
    return training_reduced, generated_reduced


def prepare_scatter_traces(
    training_reduced,
    generated_reduced,
    labels,
    colorscale: List[str] = ["#03045e", "#0096c7"],
    trace_opacity: float = 0.8,
    marker_size: int = 5,
    marker_width: int = 0,
):
    traces = []
    rows, cols = 1, 3
    for i in range(len(training_reduced)):
        create_trace = lambda array, idx: go.Scatter(
            x=array[:, 0],
            y=array[:, 1],
            mode="markers",
            name=labels[idx],
            visible=True,
            showlegend=i == 0,
            marker=dict(
                line=dict(width=marker_width, color="DarkSlateGrey"),
                size=marker_size,
                color=colorscale[idx],
                showscale=False,
                opacity=trace_opacity,
            ),
        )
        traces.append((create_trace(generated_reduced[i], 0), i // cols + 1, i % cols + 1))
        traces.append((create_trace(training_reduced[i], 1), i // cols + 1, i % cols + 1))
    return traces


def plot_scatter2d_wsubplots(
    traces,
    subplot_titles,
    n_rows: int,
    n_cols: int,
):
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.03,
    )
    for trace, row, col in traces:
        fig.add_trace(trace, row=row, col=col)

    for col, text in [(1, "a"), (2, "b"), (3, "c")]:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=-0.02,
            y=1.15,
            xanchor="right",
            yanchor="top",
            text=f"<b>[{text.upper()}]</b>",
            font=dict(size=24, family="Helvetica"),
            showarrow=False,
            row=1,
            col=col,
        )
    for col in (1, 3):
        fig.update_xaxes(title_text="", row=1, col=col)
    return fig
