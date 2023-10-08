import os
from tqdm import tqdm
import prolif
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd

from .Configuration import *

# Dictionary containing scores for different protein-ligand interactions
interaction_scores = {
    "Hydrophobic": 2.5,
    "HBDonor": 3.5,
    "HBAcceptor": 3.5,
    "Anionic": 7.5,
    "Cationic": 7.5,
    "CationPi": 2.5,
    "PiCation": 2.5,
    "VdWContact": 1.0,
    "XBAcceptor": 3.0,
    "XBDonor": 3.0,
    "FaceToFace": 3.0,
    "EdgeToFace": 1.0,
    "MetalDonor": 3.0,
    "MetalAcceptor": 3.0,
}


def get_contacts(protein: str, ligand: str) -> int:
    """
    Calculate the interaction score between a protein and ligand.

    Parameters:
    - protein (str): Path to the protein's PDB file.
    - ligand (str): Path to the ligand's SD file.

    Returns:
    - int: Cumulative interaction score.
    """
    # Convert PDB and SD files to prolif and rdkit Molecule objects respectively
    prot = prolif.Molecule(Chem.MolFromPDBFile(protein, removeHs=False))
    lig = Chem.SDMolSupplier(ligand, removeHs=False)
    lig = prolif.Molecule.from_rdkit(lig[0])

    # Compute the protein-ligand interaction fingerprint
    fp = prolif.Fingerprint(interactions=list(interaction_scores.keys()))
    fp.run_from_iterable([lig], prot, progress=False)

    try:
        # Convert fingerprint to DataFrame and compute cumulative score
        df = fp.to_dataframe()
        df_stacked = df.stack(level=[0, 1, 2])
        df_reset = df_stacked.to_frame().reset_index()
        df_reset.columns = ["Frame", "ligand", "protein", "interaction", "value"]
        df_reset["score"] = df_reset["interaction"].apply(
            lambda x: interaction_scores[x]
        )
        return df_reset["score"].sum()
    except:
        # Handle cases with no meaningful interactions
        return 0


def score_ligands(config: dict) -> dict:
    """
    Scores all ligands in the specified directory based on their interactions with a given protein.

    Parameters:
    - config (dict): Configuration dictionary containing paths and other settings.

    Returns:
    - dict: A dictionary containing ligand names as keys and their scores as values.
    """
    ligand_list = [
        config["diffdock_results_path"] + lig
        for lig in os.listdir(config["diffdock_results_path"])
        if lig.endswith(".sdf")
    ]
    ligand_scores = {}
    pbar = tqdm(ligand_list, total=len(ligand_list))
    for lig in pbar:
        if (name := lig.split("/")[-1].split(".")[0]) not in ligand_scores:
            score = get_contacts(config["protein_path"], lig)
            ligand_scores[name] = score
    return ligand_scores


def plot_ligand_scores(config: dict, ligand_scores: dict):
    """
    Plot a histogram and KDE for the ligand scores.

    Parameters:
    - config (dict): Configuration dictionary containing paths and other settings.
    - ligand_scores (dict): Dictionary of ligand scores.
    """
    data = list(ligand_scores.values())

    # Plot histogram
    plt.figure(figsize=(8, 6), dpi=80)
    plt.hist(data, bins=50, density=True, color="gray", alpha=0.7, edgecolor="black")

    # Plot KDE line
    smoothed_data = np.linspace(min(data), max(data), 1000)
    kde = gaussian_kde(data)
    smoothed_line = kde(smoothed_data)
    plt.plot(smoothed_data, smoothed_line, linewidth=2.5, color="black")

    # Configure plot details
    plt.xlabel("Values", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Density Plot of Scores", fontsize=20)
    plt.grid(linestyle="--", linewidth=0.5, alpha=0.7)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tick_params(axis="both", which="both", direction="in", length=4)
    plt.show()

    # Print summary statistics
    zero_count = data.count(0)
    print(
        f"There are {zero_count} ligands with 0 connections ({np.round(zero_count / len(data)*100, 1)}%)"
    )
    for i in range(int(np.max(data)) + 1):
        count = len([value for value in data if value > i])
        print(
            f"There are {count} ligands with scores greater than {i} ({np.round(count / len(data)*100, 1)}%)"
        )
    print(f"The mean score for all ligands is {np.round(np.mean(data), 1)}")
    print(f"The lowest score for all ligands is {np.min(data)}")


def parse_and_prepare_diffdock_data(
    ligand_scores: dict, config: dict, lower_percentile=50, threshold=11, scored_db=None
) -> pd.DataFrame:
    """
    Filter and prepare the diffdock data based on ligand scores.

    Parameters:
    - ligand_scores (dict): Dictionary of ligand scores.
    - config (dict): Configuration dictionary containing paths and other settings.
    - lower_percentile (int): Percentile below which scores are considered low. Default is 50.
    - threshold (int): Score threshold for filtering. Default is 11.
    - scored_db (dict): A dictionary containing already scored ligands, to avoid re-scoring.

    Returns:
    - pd.DataFrame: Dataframe containing parsed and prepared data.
    """
    if scored_db is None:
        scored_db = {}
    diffdock_samples = pd.read_csv(config["diffdock_samples_path"])
    if lower_percentile is not None:
        threshold = np.percentile(list(ligand_scores.values()), lower_percentile)
    all_ligands = {
        int(complex_name[7:]): score for complex_name, score in ligand_scores.items()
    }
    getter = lambda x: scored_db.get(x, all_ligands.get(x, 0))
    diffdock_samples["score"] = [
        getter(complex_number) for complex_number in diffdock_samples.index
    ]
    diffdock_samples.to_csv(config["path_to_scored"])
    good_ligands = diffdock_samples[diffdock_samples["score"] >= threshold]
    good_ligands.to_csv(config["path_to_good_mols"])
    return diffdock_samples
