import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.loaders
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List
from modules.Graph import Graph

prepare_kmeans_fnames = lambda filters: tools.loaders.setup_generations_fname_generator(
    presuffix="mix_k100means", filters=filters
)


def plot_cluster_size_evolution(
    path: str,
    fnames: List[str],
    n_rows: int,
    n_cols: int,
) -> go.Figure:
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"<b>Iteration {i}</b>" for i in range(7)],
    )
    for i, fname in enumerate(fnames):
        kmeans_tuple = pickle.load(open(path + fname + "_extended.pkl", "rb"))

        def _count(labels):
            unique, counts = np.unique(labels, return_counts=True)
            number_counts = dict(zip(unique, counts))
            return list(number_counts.values())

        row = i // n_cols + 1
        col = i % n_cols + 1
        colors = ["#080708", "#2DD881", "#0077b6", "#FF8552"]
        names = ["Lowest Loss", "High Loss", "Our Approach", "High Var"]
        for j, kmeans in enumerate(kmeans_tuple):
            if j in {1, 3}:
                continue
            fig.add_trace(
                go.Histogram(
                    x=_count(kmeans_tuple[j].labels_),
                    xbins=dict(start=0, end=2200, size=300),
                    marker_color=colors[j],  # "#03045e",
                    name=names[j],
                    showlegend=i == 0,
                    opacity=0.75,
                ),
                row=row,
                col=col,
            )
    # update bar mode to overlay
    fig.update_layout(barmode="overlay")
    gr = Graph()
    gr.update_parameters(
        dict(
            width=900,
            height=400,
            annotation_size=20,
        )
    )
    gr.style_figure(fig, force_annotations=False)
    xaxis_range = [0, 2200]  # Define the range you want for x-axis
    yaxis_range = [0, 45]
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_xaxes(range=xaxis_range, row=r, col=c)
            fig.update_yaxes(range=yaxis_range, row=r, col=c)

    # Set 'Cluster Size' as x-axis title for subplots in the 2nd row
    for c in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Cluster Size", row=n_rows, col=c)

    # Set 'Frequency' as y-axis title for subplots in the 1st column
    for r in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Frequency", row=r, col=1)
    # update legend position
    fig.update_layout(legend=dict(x=0.7, y=1.20, orientation="h"))  #
    return fig
