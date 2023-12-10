import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Union, Dict, Tuple
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.loaders
import tools.projectors

prepare_loader = tools.loaders.prepare_loader

import re

REGEX_PATTERN = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|@@|\?|>|!|\*|\$|\%[0-9]{2}|[0-9])"
DATASET_INFO = {
    "moses": ("moses_train", "moses_test", "model2"),
    "guacamol": ("guacamol_v1_train", "guacamol_v1_valid", "model3"),
    "combined": (
        "combined_processed_freq1000_block133_train",
        "combined_processed_freq1000_block133_val",
        "model7",
    ),
}
Number = Union[int, float]


def load_raw_dataset(datasets_path: str, fname: str):
    return pd.read_csv(datasets_path + fname + ".csv.gz")["smiles"].to_list()


def analyze_vocabulary_and_length(
    smiles: List[str],
) -> Tuple[Dict[str, Number], List[int]]:
    regex = re.compile(REGEX_PATTERN)
    token_to_freq: Dict[str, Number] = {}
    block_sizes = []

    for smiel in smiles:
        tokens = regex.findall(smiel.strip())
        block_sizes.append(len(tokens))
        for token in tokens:
            token_to_freq[token] = token_to_freq.get(token, 0) + 1

    return token_to_freq, block_sizes


def plot_block_sizes_and_vocabulary_frequency(
    token_to_freq: Dict[str, int], block_sizes: List[int]
) -> go.Figure:
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("<b>Block Sizes</b>", "<b>Token Frequencies</b>"),
    )

    # Subplot 1: Histogram of Block Sizes
    fig.add_trace(
        go.Histogram(
            x=block_sizes,
            name="Block Sizes",
            showlegend=False,
            marker_color="#023e8a",
        ),
        row=1,
        col=1,
    )
    # Add a vertical line at block size 133
    fig.add_vline(
        x=133, line_width=2, line_dash="dash", line_color="gray", row=1, col=1
    )

    # Overlay percentile values
    percentiles = sorted([95, 99, 99.9, 99.99], reverse=True)
    for i, p in enumerate(percentiles):
        value = np.percentile(block_sizes, p)
        fig.add_annotation(
            x=0.5,
            y=0.0 + i * 0.05,
            text=f"{p}th percentile: {value:.2f}",
            showarrow=False,
            xref="x domain",
            xanchor="left",
            yref="y domain",
            row=1,
            col=1,
        )

    # Subplot 2: Bar Chart of Token Frequencies
    sorted_tokens = sorted(token_to_freq.items(), key=lambda x: x[1], reverse=True)
    tokens, freqs = zip(*sorted_tokens)
    fig.add_trace(
        go.Bar(
            x=tokens,
            y=freqs,
            name="Token Frequencies",
            showlegend=False,
            marker_color="#023e8a",
        ),
        row=1,
        col=2,
    )

    # Find the last token with frequency > 1000
    last_index = next(i for i, f in enumerate(freqs) if f <= 1000)
    fig.add_vline(
        x=last_index, line_width=2, line_dash="dash", line_color="gray", row=1, col=2
    )

    # Add annotations for token frequency thresholds
    thresholds = [1000, 500, 100]
    for i, t in enumerate(thresholds):
        count = sum(1 for f in freqs if f < t)
        fig.add_annotation(
            x=0.5,
            y=0.0 + i * 0.05,
            text=f"Tokens < {t}: {count}",
            showarrow=False,
            xref="x2 domain",
            xanchor="left",
            yref="y2 domain",
            row=1,
            col=2,
        )
    return fig


def plot_molecular_weight_distribution(
    col1_mw: np.ndarray, col2_mw: np.ndarray, col3_mw: np.ndarray
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("<b>MOSES</b>", "<b>GuacaMol</b>", "<b>Combined Dataset</b>"),
    )

    # Add a histogram trace for molecular weight distribution
    fig.add_trace(
        go.Histogram(
            x=col1_mw,
            xbins=dict(start=0, end=1000, size=2),
            marker_color="#023e8a",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=col2_mw,
            xbins=dict(start=0, end=1500, size=2),
            marker_color="#023e8a",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=col3_mw,
            xbins=dict(start=0, end=2000, size=2),
            marker_color="#023e8a",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    return fig


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
        traces.append(
            (create_trace(generated_reduced[i], 0), i // cols + 1, i % cols + 1)
        )
        traces.append(
            (create_trace(training_reduced[i], 1), i // cols + 1, i % cols + 1)
        )
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
