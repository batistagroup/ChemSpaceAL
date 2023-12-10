from modules.Graph import Graph
import modules.secret
import os
import plotly.graph_objects as go
import pickle
import numpy as np

EXPORT_PATH = os.path.join(
    os.getcwd(), "figures", "exports", "distribution_analysis", ""
)
PCA_PATH = modules.secret.PRODUCTION_RUNS_PATH + "3. Sampling/pca_weights/"
PCA_FNAME = "scaler_pca_combined_processed_freq1000_block133_120"

_, pca = pickle.load(open(PCA_PATH + PCA_FNAME + ".pkl", "rb"))


fig = go.Figure()
variance = pca.explained_variance_ratio_.cumsum() * 100
components = list(range(1, 121))
fig.add_trace(
    go.Scatter(
        x=components,
        y=variance,
        mode="markers+lines",
        marker=dict(color="#0077b6"),
        showlegend=False,
    ),
)
fig.add_shape(
    type="line",
    x0=components[0],
    y0=100,
    x1=components[-1],
    y1=100,
    line=dict(color="black", width=3, dash="dash"),
)
for i, percentile in enumerate([25, 50, 75, 90, 95, 99]):
    n_comps = np.argmax(variance > percentile)
    fig.add_annotation(
        text=f"{percentile}% of variance is explained by {n_comps} components",
        x=20,
        y=50 - 6 * i,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=12),
    )
graph = Graph()
graph.update_parameters(
    dict(
        width=500,
        height=400,
        t_margin=10,
        xaxis_title="Number of Components",
        yaxis_title="Percentage of Variance Explained",
        xdticks=10,
        ydticks=10,
    )
)
graph.style_figure(fig)
graph.save_figure(
    fig,
    path=EXPORT_PATH,
    fname="variance_explained",
    svg=True,
)
