import modules.ConversionSchemes as cs
from modules.ScoreDistribution import compute_cluster_scores
import modules.secret
import numpy as np
import os
from modules.Graph import Graph

SCORING_PATH = modules.secret.PRODUCTION_RUNS_PATH + "5. Scoring/scored_dataframes/"
EXPORT_PATH = os.path.join(
    os.getcwd(), "figures", "exports", "conversion_schemes", ""
)

cluster_scores = compute_cluster_scores(
    scoring_path=SCORING_PATH,
    fname="model7_baseline_1IEP_mix_k100_ADMET+FGs",
    aggregation_mode="mean",
    keep_cluster_id=True
)
fig = cs.plot_probability_distribution(
    cluster_scores,
    [
        ("linear", None, None),
        ("softmax", False, np.nan),
        ("softmax", True, 1),
        ("softmax", True, 0.5),
        ("softmax", True, 0.25),
    ],
    bin_size=0.001
)

graph = Graph()
graph.update_parameters(dict(
        xaxis_title="Sampling Fraction",
        yaxis_title="Relative Frequency",
        width=900,
        height=300,
        axis_title_size=18,
        annotation_size=24,
    ))
graph.style_figure(fig)
fig.update_layout(legend=dict(
        x=0.9, 
        y=0.575,
        xanchor="left",
        yanchor="middle",
        font=dict(size=14),
    ))
graph.save_figure(
    fig,
    path=EXPORT_PATH,
    fname="probability_distributions",
    svg=True,
)