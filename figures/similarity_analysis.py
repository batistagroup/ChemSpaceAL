from modules.Graph import Graph
import modules.secret
import modules.ChemicalSimilarity as chem_sim
import os
import pprint
import plotly.graph_objects as go
from typing import List
from tqdm import tqdm 

pp = pprint.PrettyPrinter(indent=2, width=100)
EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "chemical_similarity", "")
GENERATIONS_PATH = modules.secret.PRODUCTION_RUNS_PATH + "2. Generation/smiles/"
SCORING_PATH = modules.secret.PRODUCTION_RUNS_PATH + "5. Scoring/scored_dataframes/"


def analyze_distribution_molecules(
    smiles_lists: List[List[str]], fp_type: str, sim_type: str, fname: str
):
    fig = chem_sim.create_mean_max_similarity_figure(
        smile_lists,
        fp_type=fp_type,
        sim_type=sim_type,
        h_space=0.12,
        colorbars=[
            dict(
                x=0.44,  # Position of the colorbar (0 - left, 1 - right)
                len=1,  # Length of the colorbar
                title=f"{sim_type}<br>Similarity",  # Title of the colorbar
            ),
            dict(
                x=1,  # Position of the colorbar (0 - left, 1 - right)
                len=1,  # Length of the colorbar
                title=f"{sim_type}<br>Similarity",  # Title of the colorbar
            ),
        ],
    )
    gr = Graph()
    gr.update_parameters(dict(width=1000, height=350, annotation_size=24))
    gr.style_figure(fig)
    gr.save_figure(fig, path=EXPORT_PATH, fname=f"{fname}_{sim_type}_on_{fp_type}")


def analyze_abl_binders():
    for fp_type in chem_sim.ALL_FP_TYPES:
        fig = chem_sim.create_abl_inhibitors_heatmap_all_simtypes(
            fp_type=fp_type, sim_types=chem_sim.ALL_SIMILARITY_METRICS
        )
        gr = Graph()
        gr.update_parameters(
            dict(
                width=800,
                height=500,
                annotation_size=24,
                title=f"<b>Fingerprint Type: {fp_type}</b>",
                t_margin=80,
            )
        )
        gr.style_figure(fig)
        # fig.show()
        gr.save_figure(fig, path=EXPORT_PATH, fname=f"abl_inhibitors_{fp_type}")


if __name__ == "__main__":
    n_iters = 5
    configs = [
        ("model7_hnh_admet", "HNH", "ADMET", "softsub"),
        ("model7_hnh_admetfg", "HNH", "ADMET+FGs", "softsub"),
        ("model2_hnh", "HNH", "ADMET+FGs", "admetfg_softsub"),
        ("model7_1iep_admet", "1IEP", "ADMET", "softsub"),
        ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "softsub"),
        ("model2_1iep", "1IEP", "ADMET+FGs", "admetfg_softsub"),
    ]
    fp_types = ["RDKit FP", "MACCS", "ECFP"]
    sim_types = ["Tanimoto", "Dice"]
    pbar = tqdm(configs)
    for prefix, target, filters, channel in pbar:
        for fp_type in fp_types:
            for sim_type in sim_types:
                pbar.set_description(f"{prefix} {fp_type} {sim_type}")
                fnames = chem_sim.prepare_scored_fnames(prefix, n_iters, channel, filters, target)
                load_scored = chem_sim.prepare_loader(SCORING_PATH)
                smile_lists = [load_scored(fname)[:10] for fname in fnames]
                fname = f"{prefix}_{channel}_{target}_{filters}"
                analyze_distribution_molecules(smile_lists, fp_type, sim_type, fname)
