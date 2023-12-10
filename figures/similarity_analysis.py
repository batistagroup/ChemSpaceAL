from modules.Graph import Graph
import modules.secret
import modules.ChemicalSimilarity as chem_sim
import os
import pprint
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pickle

pp = pprint.PrettyPrinter(indent=2, width=100)
EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "chemical_similarity", "")
GENERATIONS_PATH = modules.secret.PRODUCTION_RUNS_PATH + "2. Generation/smiles/"
SCORING_PATH = modules.secret.PRODUCTION_RUNS_PATH + "5. Scoring/scored_dataframes/"


def analyze_distribution_molecules(
    smiles_lists: List[List[str]],
    fp_type: str,
    sim_type: str,
    fname: str,
    mean_zmin: Optional[float] = None,
    mean_zmax: Optional[float] = None,
    max_zmin: Optional[float] = None,
    max_zmax: Optional[float] = None,
):
    fig, mean_lists, max_lists = chem_sim.create_mean_max_similarity_figure(
        smiles_lists,
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
        mean_zmin=mean_zmin,
        mean_zmax=mean_zmax,
        max_zmin=max_zmin,
        max_zmax=max_zmax,
    )
    gr = Graph()
    gr.update_parameters(dict(width=1000, height=350, annotation_size=24))
    gr.style_figure(fig)
    gr.save_figure(fig, path=EXPORT_PATH, fname=f"{fp_type}_by_{sim_type}_{fname}")
    pickle.dump(
        (fig, mean_lists, max_lists),
        open(EXPORT_PATH + f"{fp_type}_by_{sim_type}_{fname}.pkl", "wb"),
    )


def analyze_abl_with_rdkit():
    fp_type = "RDKit FP"
    fp_type = "ECFP"
    sim_type = "Tanimoto"
    fig = go.Figure()
    fig.add_trace(
        chem_sim.create_abl_trace(
            fp_type,
            sim_type,
            3,
            colorbar=dict(
                    title=f"{sim_type}<br>Similarity",
                    tickvals=[0, 0.3, 0.6, 0.9]
                )
        )
    )
    gr = Graph()
    gr.update_parameters(
        dict(
            width=600,
            height=300,
            annotation_size=24,
            # title=f"<b>Fingerprint Type: {fp_type}</b>",
            t_margin=20,
            b_margin=0,
            l_margin=0,
            r_margin=0
        )
    )
    gr.style_figure(fig)
    # update the colorbar for the Heatmap trace to mention the similarity metric

    gr.save_figure(fig, path=EXPORT_PATH, fname=f"abl_inhibitors_{fp_type}_{sim_type}")


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
    # analyze_abl_binders()
    analyze_abl_with_rdkit()
    # configs = [
    #     ("model7_1iep_admet", "1IEP", "ADMET", "softsub"),
    #     ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "softsub"),
    #     ("model2_1iep", "1IEP", "ADMET+FGs", "admetfg_softsub"),
    # ]
    # fp_types = ["RDKit FP", "MACCS", "ECFP"]
    # sim_types = ["Tanimoto"]
    # pbar = tqdm(configs)
    # for prefix, target, filters, channel in pbar:
    #     for fp_type in fp_types:
    #         for sim_type in sim_types:
    #             pbar.set_description(f"{prefix} {fp_type} {sim_type}")
    #             fnames = chem_sim.prepare_scored_fnames(
    #                 prefix, n_iters, channel, filters, target
    #             )
    #             load_scored = chem_sim.prepare_loader(SCORING_PATH)
    #             smile_lists = [load_scored(fname) for fname in fnames]
    #             fname = f"{prefix}_{channel}_{target}_{filters}"
    #             analyze_distribution_molecules(smile_lists, fp_type, sim_type, fname)
    configs = [
        ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "softsub"),
        ("model2_1iep", "1IEP", "ADMET+FGs", "admetfg_softsub"),
        ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "random"),
        ("model7_1iep_admet", "1IEP", "ADMET", "softsub"),
        ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "randomwsampling"),
        ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "diffusion"),
    ]
    # For Tanimoto
    fpToHyperparams: Dict[str, Dict[str, List[float]]] = {
        "RDKit FP": {"mean": [0.27, 0.45], "max": [0.41, 0.80]},
        # "MACCS": {"mean": [0.41, 0.59], "max": [0.65, 0.95]},
        # "ECFP": {"mean": [0.08, 0.22], "max": [0.18, 0.66]},
    }
    sim_type = "Tanimoto"
    # For Asymmetric
    # fpToHyperparams: Dict[str, Dict[str, List[float]]] = {
    #     "RDKit FP": {"mean": [0.51, 0.67], "max": [0.69, 0.99]},
    #     "MACCS": {"mean": [0.66, 0.84], "max": [0.88, 1.0]},
    #     "ECFP": {"mean": [0.18, 0.38], "max": [0.40, 0.85]},
    # }
    # sim_type = "Asymmetric"
    # pbar = tqdm(configs)
    # for prefix, target, filters, channel in pbar:
    #     for fp_type, hyperparams in fpToHyperparams.items():
    #         pbar.set_description(f"{prefix} {fp_type} {sim_type}")
    #         fnames = chem_sim.prepare_scored_fnames(
    #             prefix, n_iters, channel, filters, target
    #         )
    #         fnames = chem_sim.prepare_generated_fnames(
    #             prefix, n_iters, channel, filters, target
    #         )
    #         # load_scored = chem_sim.prepare_loader(SCORING_PATH)
    #         # smile_lists = [load_scored(fname) for fname in fnames]
    #         load_generated = chem_sim.prepare_loader(GENERATIONS_PATH)
    #         smile_lists = [load_generated(fname) for fname in fnames]
    #         fname = f"{prefix}_{channel}_{target}_{filters}"
    #         # analyze_distribution_molecules(
    #         #     smile_lists,
    #         #     fp_type,
    #         #     sim_type,
    #         #     fname,
    #         #     mean_zmin=hyperparams["mean"][0],
    #         #     mean_zmax=hyperparams["mean"][1],
    #         #     max_zmin=hyperparams["max"][0],
    #         #     max_zmax=hyperparams["max"][1],
    #         # )
