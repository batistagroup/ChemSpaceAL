import modules.AnalyzeDistribution as ad
from modules.ChemicalSimilarity import ABL_BINDERS
import modules.secret
import pprint
import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
from typing import Optional, List, cast
from modules.Graph import Graph

pp = pprint.PrettyPrinter(indent=2, width=100)
PCA_PATH = modules.secret.PRODUCTION_RUNS_PATH + "3. Sampling/pca_weights/"
GENERATIONS_PATH = modules.secret.PRODUCTION_RUNS_PATH + "2. Generation/smiles/"
SCORING_PATH = modules.secret.PRODUCTION_RUNS_PATH + "5. Scoring/scored_dataframes/"
ALTRAINS_PATH = modules.secret.PRODUCTION_RUNS_PATH + "6. ActiveLearning/training_sets/"
PCA_FNAME = "scaler_pca_combined_processed_freq1000_block133_120"
EXPORT_PATH = os.path.join(
    os.getcwd(), "figures", "exports", "distribution_analysis", ""
)
PICKLES_PATH = os.path.join(EXPORT_PATH, "pickles", "")

load_generations = ad.prepare_loader(GENERATIONS_PATH)
load_scored = ad.prepare_loader(SCORING_PATH, return_full=True)
load_altrainsets = ad.prepare_loader(ALTRAINS_PATH)
np.random.seed(42)


# (
#     scores_dfs,
#     generated_smiles,
#     altrains_smiles,
#     diffusion_smiles,
# ) = ad.load_smiles_for_config(
#     configs=[
#         ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "softsub"),
#     ],
#     gen_loader=load_generations,
#     scored_loader=load_scored,
#     altrain_loader=load_altrainsets,
#     save_path=EXPORT_PATH,
#     save_fname="softsub_1iep_run_smiles",
# )
# ad.calculate_descriptors(
#     containers_list=[
#         [df["smiles"].to_numpy() for df in scores_dfs],
#         generated_smiles,
#         altrains_smiles,
#         diffusion_smiles,
#     ],
#     save_path=PICKLES_PATH,
#     save_fname="softsub_1iep_run_(scores,generated,altrains,diffusion)",
# )

# ad._calculate_descriptors_for_an_array(smiles=list(ABL_BINDERS.values()), save_path=PICKLES_PATH, save_name="ABL_BINDERS", desc_mode="mix")

scores_dfs, generated_smiles, altrains_smiles, diffusion_smiles = pickle.load(
    open(PICKLES_PATH + "softsub_1iep_run_smiles.pkl", "rb")
)
descriptors = ad.load_descriptors(
    PICKLES_PATH, "softsub_1iep_run_(scores,generated,altrains,diffusion)"
)
abl_descriptors = ad.load_descriptors(PICKLES_PATH, "ABL_BINDERS")

# reduction_config = dict(
#     reduction="PCA",
#     reduction_parameters=dict(
#         reduce_to_2d=True,
#     ),
# )
# reduction_config = dict(
#     reduction="UMAP",
#     reduction_parameters=dict(
#         n_neighbors=15,
#         min_dist=0.1,
#         verbose=True,
#     ),
# )
# reduction_config = dict(
#     reduction="t-SNE",
#     reduction_parameters=dict(
#         perplexity=40, early_exaggeration=60, verbose=True, n_jobs=8
#     ),
# )
# match reduction_config["reduction"]:
#     case "t-SNE":
#         params = cast(dict, reduction_config["reduction_parameters"])
#         fsuffix = f"_ee{params['early_exaggeration']}_per{params['perplexity']}"
#     case _:
#         fsuffix = ""

# ad.fit_reduction_on_smiles(
#     containers_list=[
#         [df["smiles"].to_numpy() for df in scores_dfs],
#         generated_smiles,
#         altrains_smiles,
#         diffusion_smiles,
#         [list(ABL_BINDERS.values())],
#     ],
#     descriptors=pd.concat([abl_descriptors, descriptors]),
#     reduction_config=reduction_config,
#     save_path=os.path.join(EXPORT_PATH, "pickles", ""),
#     save_fname=f"softsub_1iep_run_{reduction_config['reduction']}" + fsuffix,
#     pca_path=PCA_PATH,
#     pca_fname=PCA_FNAME,
# )
map_loader = lambda fname: pickle.load(
    open(
        os.path.join(EXPORT_PATH, "pickles", fname + ".pkl"),
        "rb",
    )
)
pca_mapping = map_loader("softsub_1iep_run_PCA")
tsne_mapping = map_loader("softsub_1iep_run_t-SNE_ee60_per40")

# ---------------------------------------------------------------
# Correlation between position and score (PCA and t-SNE)
# traces = ad.prepare_scored_traces(
#     scored_dfs=scores_dfs,
#     tsne_mapping=tsne_mapping,
#     pca_mapping=pca_mapping,
#     bin_size_factor=1,
#     force_zmax=80,
#     force_zmin=30,
#     colorscale="Electric",
# )
# fig = ad.plot_heatmap_wsubplots(
#     traces=traces,
#     subplot_titles=[
#         "<b>PCA</b>",
#         "<b>t-SNE</b>",
#     ],
#     rows=1,
#     cols=2,
#     shared_xaxes=False,
#     shared_yaxes=False,
# )
# ABL_TO_STYLE = {
#     "dasatinib": "#90e0ef",  # 37 - this is the interaction score
#     "bosutinib": "#00b4d8",  # 42
#     "nilotinib": "#c77dff",  # 55
#     "ponatinib": "#c77dff",  # 58.5
#     "imatinib": "#ff4d6d",  # 64.5
#     "bafetinib": "#ff4d6d",  # 64.5
# }
# for abl_name, abl_smile in ABL_BINDERS.items():
#     for mapping, col in [(pca_mapping, 1), (tsne_mapping, 2)]:
#         fig.add_trace(
#             go.Scatter(
#                 x=[mapping[abl_smile][0]],
#                 y=[mapping[abl_smile][1]],
#                 mode="markers",
#                 marker=dict(
#                     color=ABL_TO_STYLE[abl_name],
#                     size=5,
#                 ),
#                 showlegend=False,
#             ),
#             row=1,
#             col=col,
#         )

# graph = Graph()
# graph.update_parameters(
#     dict(
#         annotation_size=20,
#         width=800,
#         height=400,
#         tick_position="",
#         b_margin=80,
#     )
# )

# direct_xaxes = [
#     dict(
#         title_text="Principal Component 1 (18.6% variance explained)",
#         row=1,
#         col=1,
#         range=[-20, 20],
#     ),
#     dict(title_text="t-SNE Component 1", row=1, col=2),
# ]
# direct_yaxes = [
#     dict(
#         title_text="Principal Component 2<br>(5.7% variance explained)",
#         row=1,
#         col=1,
#         range=[-20, 20],
#     ),
#     dict(title_text="t-SNE Component 2", row=1, col=2),
# ]
# label_annotations = []
# for col, text in [(1, "a"), (2, "b")]:
#     label_annotations.append(
#         dict(
#             xref="x domain",
#             yref="y domain",
#             x=-0.03 + col * 0.02,
#             y=1.15,
#             xanchor="right",
#             yanchor="top",
#             text=f"<b>[{text.upper()}]</b>",
#             font=dict(size=24, family="Helvetica"),
#             showarrow=False,
#             row=1,
#             col=col,
#         )
#     )

# graph.style_figure(fig, force_annotations=False)
# for config in direct_xaxes:
#     fig.update_xaxes(**config)
# for config in direct_yaxes:
#     fig.update_yaxes(**config)
# for annotation in label_annotations:
#     fig.add_annotation(**annotation)
# graph.save_figure(
#     figure=fig,
#     path=EXPORT_PATH,
#     fname="scored_molecules_mix100_softsub_1iep_run",
# )

# ---------------------------------------------------------------


# ---------------------------------------------------------------
# Evolution of the distribution
# bin_size = 0.5
# forced_zmin = 0
# forced_zmax = 50
# reduction = "PCA"
# mapping = pca_mapping
# -------------------
bin_size = 1.5
forced_zmin = 0
forced_zmax = 15
reduction = "t-SNE"
mapping = tsne_mapping
# -------------------

n_iters = 5
n_rows, n_cols = 4, 6
colorscale = "Thermal"


generated_reduced = [
    [mapping[smile] for smile in smile_list] for smile_list in generated_smiles
]
altrains_reduced = [
    [mapping[smile] for smile in smile_list] for smile_list in altrains_smiles
]
diffusion_reduced = [
    [mapping[smile] for smile in smile_list] for smile_list in diffusion_smiles
]
all_reduced = [
    np.vstack(generated_reduced),
    np.vstack(altrains_reduced),
    np.vstack(diffusion_reduced),
]

common_boundaries = ad.get_data_boundaries(data_list=all_reduced, round_to=1)

absolute_trace_creator = lambda smiles_containers: [
    ad.create_heatmap_trace(
        data=np.array([mapping[smile] for smile in smile_list]),
        boundaries=common_boundaries,
        bin_size=bin_size,
        force_zmax=forced_zmax,
        force_zmin=forced_zmin,
        showscale=i == 0,
        showlegend=False,
        coloraxis="coloraxis1",
    )
    for i, smile_list in enumerate(smiles_containers)
]
generations = absolute_trace_creator(generated_smiles)
altrains = absolute_trace_creator(altrains_smiles)

diff_trace_creator = lambda minuend, subtrahend: ad.create_traces_from_difference(
    minuend=minuend,
    subtrahend=subtrahend,
    mapping=mapping,
    boundaries=common_boundaries,
    forced_subtrahend_index=0,
    bin_size=bin_size,
    force_zmax=forced_zmax,
    force_zmin=forced_zmin,
)
generations_dif = diff_trace_creator(generated_smiles, generated_smiles)
sampled_generated = np.random.choice(generated_smiles[0], size=5_000, replace=False)
altrains_dif = diff_trace_creator(altrains_smiles, [sampled_generated])


annotations = []
for i, (subtitle, x, y) in enumerate(
    [
        ("<b>Distribution of generated molecules</b>", 0, 1.05),
        ("<b>Distribution of AL training sets</b>", 0, 0.785),
        (
            "<b>Changes in generated molecules relative to generations from iteration 0</b>",
            0,
            0.50,
        ),
        (
            "<b>Changes in AL training sets relative to generations from iteration 0</b>",
            # "<b>Changes in AL training sets relative to training sets from preceding iteration</b>",
            0,
            0.215,
        ),
    ]
):
    annotations.append(
        dict(
            text=subtitle,
            xref="paper",
            yref="paper",
            x=x,  # Adjust as needed to place the label on the left 0, 0)
            y=y,
            showarrow=False,
            font=dict(size=16, color="#333", family="Helvetica"),  # Font size))
        )
    )

# label_annotations = []
for row, text in [(1, "a"), (2, "b"), (3, "c"), (4, "d")]:
    annotations.append(
        dict(
            xref="x domain",
            yref="y domain",
            x=-0.08,  # -1.15,
            y=1.3,  # 2.56 - 1.24 * row,
            xanchor="right",
            yanchor="top",
            text=f"<b>[{text.upper()}]</b>",
            font=dict(size=24, family="Helvetica"),
            showarrow=False,
            row=row,
            col=1,
        )
    )
# title = f"Distribution of {description} (discretized into pixels with size {bin_size})<br>in 196 descriptors-based space (visualized in 2D with {reduction_config['reduction']})"
fig = ad.plot_heatmap_wsubplots(
    [(trace, i // n_cols + 1, i % n_cols + 1) for i, trace in enumerate(generations)]
    + [(trace, i // n_cols + 2, i % n_cols + 1) for i, trace in enumerate(altrains)]
    + [
        (trace, i // n_cols + 3, i % n_cols + 1)
        for i, trace in enumerate(generations_dif)
    ]
    + [
        (trace, i // n_cols + 4, i % n_cols + 1) for i, trace in enumerate(altrains_dif)
    ],
    subplot_titles=[],  # 1 * [f"Iteration {i}" for i in range(n_iters + 1)],
    horizontal_spacing=0.01,
    vertical_spacing=0.05,
    rows=n_rows,
    cols=n_cols,
)
direct_xaxes = [
    dict(tickvals=[], row=r, col=c) for c in range(1, 7) for r in range(1, 4)
] + [dict(title_text=f"Iteration {i}", row=4, col=i + 1) for i in range(n_iters + 1)]
direct_yaxes = [
    dict(tickvals=[], row=r, col=c) for c in range(2, 7) for r in range(1, 5)
]
direct_layout = dict(
    coloraxis1=dict(
        colorscale=colorscale,
        cmax=forced_zmax,  # 150 for PCA
        cmin=0,
        colorbar=dict(x=1.01, y=1.02, len=0.5, yanchor="top", title="Count"),
    ),
    coloraxis2=dict(
        colorscale="RdBu",
        cmin=-forced_zmax,  # 150 for PCA
        cmax=forced_zmax,
        colorbar=dict(
            x=1.01,
            y=0.49,
            len=0.5,
            yanchor="top",
            title="Difference<br>in count",
        ),
    ),
)
ABL_TO_STYLE = {
    "dasatinib": "#00f6ff",  # 37 - this is the interaction score
    "bosutinib": "#00f6ff",  # 42
    "nilotinib": "#70e000",  # 55
    "ponatinib": "#70e000",  # 58.5
    "imatinib": "#ff206e",  # 64.5
    "bafetinib": "#ff206e",  # 64.5
}
for abl_name, abl_smile in ABL_BINDERS.items():
    for row in range(1, 3):
        for col in range(1, 7):
            fig.add_trace(
                go.Scatter(
                    x=[mapping[abl_smile][0]],
                    y=[mapping[abl_smile][1]],
                    mode="markers",
                    marker=dict(
                        color=ABL_TO_STYLE[abl_name],
                        size=5,
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
graph = Graph()
graph.update_parameters(
    dict(
        axis_title_size=12,
        width=900,
        height=600,
        showlegend=False,
        tick_position="",
        ymirror=True,
        xmirror=True,
    )
)
graph.style_figure(fig, force_annotations=False)
for config in direct_xaxes:
    fig.update_xaxes(**config)
for config in direct_yaxes:
    fig.update_yaxes(**config)
for annotation in annotations:
    fig.add_annotation(**annotation)
fig.update_layout(**direct_layout)
graph.save_figure(
    figure=fig,
    path=EXPORT_PATH,
    fname=f"distrib_evolution_{reduction}",
)
# ---------------------------------------------------------------
