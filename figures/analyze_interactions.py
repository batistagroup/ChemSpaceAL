import modules.secret
from modules.Graph import Graph
import modules.InteractionsCounts as ic
import os
import pickle

GENERATIONS_PATH = modules.secret.PRODUCTION_RUNS_PATH + "2. Generation/smiles/"
SCORING_PATH = modules.secret.PRODUCTION_RUNS_PATH + "5. Scoring/scoring_extended/"
PDBBIND_PATH = modules.secret.PRODUCTION_RUNS_PATH + "pdbbind/"
EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "interactions_counts", "")

affinities = pickle.load(open(PDBBIND_PATH + "affinities.pkl", "rb"))
scores = pickle.load(open(PDBBIND_PATH + "refined_scores_final_2.pkl", "rb"))

assert not ic.check_for_multiple_ligands(affinities, scores)

# ------------------------------
# PDBBind Analysis
# fig = ic.create_pdbbind_figure(
#     affinities,
#     scores,
#     subplot_titles=[
#         "<b>Interaction Score vs. Binding Affinity</b>",
#         "<b>Distribution of Interaction Scores</b>",
#     ],
# )
# graph = Graph()
# graph.update_parameters(
#     dict(
#         width=900,
#         height=400,
#         xmirror=True,
#         ymirror=True,
#         xtick_len=5,
#         ytick_len=5,
#         xtick_width=1,
#         ytick_width=1,
#         axis_title_size=20,
#         annotation_size=18,
#     )
# )
# graph.style_figure(fig, force_annotations=False)
# graph.save_figure(figure=fig, path=EXPORT_PATH, fname="correlation_test", svg=True)
# ------------------------------

# ------------------------------
mapping = {
    "Hydrophobic": "Hydrophobic",
    "HBDonor": "Hydrogen-bond",
    "HBAcceptor": "Hydrogen-bond",
    "Cationic": "Ionic",
    "Anionic": "Ionic",
    "ArRingCation": "Aromatic ring and cation",
    "VdWContact": "Van der Waals",
    "XbDonor": "Halogen-bond",
    "F2FPiStack": "Face-to-face pi-stacking",
    "E2FPiStack": "Edge-to-face pi-stacking",
    "MetallAcceptor": "Metallic",
}

_, _, unique_interactions = ic.count_interactions(
    ic.rename_interactions(scores, mapping), "once"
)

desc_type = "mix"
n_iters = 5
for calc_method in ["each", "once"]:
    for channel in ["softdiv", "linear", "random", "diffusion", "softsub"]:  #
        if channel == "random":
            identifier = "random"
            suffix = "random"
            title = "Random Sampling"
        else:
            identifier = f"{desc_type}100_{channel}"
            suffix = f"{desc_type}_k100"
            title = f"{channel.capitalize()}-based Sampling"
        fig = ic.create_interactions_heatmap_figure(
            scoring_path=SCORING_PATH,
            fnames=[
                f"model7_baseline_{desc_type}_k100",
                *(f"model7_{identifier}_al{i}_{suffix}" for i in range(1, n_iters + 1)),
            ],
            calc_method=calc_method,
            row_titles=[
                "PDB Bind",
                "Iteration 0",
                *(f"Iteration {i}" for i in range(1, n_iters + 1)),
            ],
            labels=unique_interactions,
            mapping=mapping,
            normalize_counts=False,
            control_dict=ic.rename_interactions(scores, mapping),
        )
        graph = Graph()
        graph.update_parameters(
            dict(
                width=950,
                height=300,
                show_xgrid=False,
                tick_position="",
                title=f"Interaction Counts per 1000 Molecules ({title})",
            )
        )
        graph.style_figure(fig)
        graph.save_figure(
            figure=fig,
            path=EXPORT_PATH,
            fname=f"interactions_table_{desc_type}k100_{channel}_{calc_method}",
        )
