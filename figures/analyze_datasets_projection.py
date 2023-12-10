import modules.secret
from modules.Graph import Graph
import modules.DatasetAnalysis as dataset_analysis
import modules.AnalyzeDistribution as analyze_dist
import os
from typing import cast, Iterable

PRETRAINING_PATH = modules.secret.PRODUCTION_RUNS_PATH + "1. Pretraining/datasets/"
GENERATIONS_PATH = modules.secret.PRODUCTION_RUNS_PATH + "2. Generation/smiles/"
PCA_PATH = modules.secret.PRODUCTION_RUNS_PATH + "3. Sampling/pca_weights/"
PCA_FNAME = "scaler_pca_combined_processed_freq1000_block133_120"
EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "datasets_analysis", "")


datasets = ["moses", "guacamol", "combined"]
reduction = "PCA"
desc_type = "mix"
train_sample = 10_000
generation_sample = 10_000

training_smiles = [
    dataset_analysis.load_training_smiles(
        PRETRAINING_PATH, dataset, sample=train_sample
    )
    for dataset in datasets
]
generated_smiles = [
    dataset_analysis.load_generated_smiles(
        GENERATIONS_PATH, dataset, sample=generation_sample
    )
    for dataset in datasets
]

smile_set = set()
for smile_container in [training_smiles, generated_smiles]:
    for smiles in smile_container:
        smile_set |= set(smiles)
all_smiles = [training_smiles[i] + generated_smiles[i] for i in range(len(datasets))]
# analyze_dist._calculate_descriptors_for_an_array(smiles=list(smile_set), save_path=EXPORT_PATH, save_name="moses_guac_combined", desc_mode=desc_type)
descriptors = analyze_dist.load_descriptors(
    load_path=EXPORT_PATH, load_fname="moses_guac_combined"
)
training_projection = [
    analyze_dist.project_smiles(smiles, descriptors) for smiles in training_smiles
]
generated_projection = [
    analyze_dist.project_smiles(smiles, descriptors) for smiles in generated_smiles
]
training_reduced, generated_reduced = dataset_analysis.reduce_training_and_generations(
    training_projection, generated_projection, PCA_PATH, PCA_FNAME
)

# def invert_even_elts(array):
#     return [array[1], array[0], array[3], array[2], array[5], array[4]]


traces = dataset_analysis.prepare_scatter_traces(
    training_reduced, generated_reduced,
    labels=["Generations", "Training Set"],
    colorscale=("#240046", "#80ed99"),
    trace_opacity=0.5,
    marker_size=2,
    marker_width=0.1,
)

fig = dataset_analysis.plot_scatter2d_wsubplots(
    traces=traces,
    subplot_titles=[
        "<b>MOSES</b>",
        "<b>GuacaMol</b>",
        "<b>Combined Dataset</b>",
    ],
    n_rows=1,
    n_cols=3,
)

graph = Graph()
graph.update_parameters(dict(
        width=1100,
        height=400,
        xrange=[-18, 50],
        yrange=[-18, 23],
        xtick_len=4,
        ytick_len=4,
        xtick_width=1,
        ytick_width=1,
        axis_title_size=18,
        xaxis_title="Principal Component 1 (18.6% variance explained)",
        yaxis_title="Principal Component 2<br>(5.7% variance explained)",
        show_xzero=True,
        show_yzero=True,
        annotation_size=20,
    ))
graph.style_figure(fig, force_annotations=False)
fig.update_layout(legend=dict(
        x=0.01,
        y=1.0,
        xanchor="left",
        yanchor="top",
        font=dict(size=16),
        orientation="h",
    ), yaxis2_title="", yaxis3_title="")
graph.save_figure(
    figure=fig,
    path=EXPORT_PATH,
    fname=f"{'+'.join(datasets)}_{desc_type}_{reduction}_trainsample{train_sample}_trainsample{generation_sample}",
    html=False,
    svg=True,
)
