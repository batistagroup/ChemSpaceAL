from ChemSpaceAL.Configuration import Config
from rdkit import Chem
import prolif
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional


def score_protein_ligand_pose(
    config: Config, protein_path: str, ligand_path: str
) -> float:
    """
    Calculate the interaction score between a protein and ligand.

    Parameters:
    - protein (str): Path to the protein's PDB file.
    - ligand (str): Path to the ligand's SD file.

    Returns:
    - int: Cumulative interaction score.
    """
    assert isinstance(
        config.prolif_weights, dict
    ), ".set_scoring_parameters() wasn't called"
    # Convert PDB and SD files to prolif and rdkit Molecule objects respectively
    protein = prolif.Molecule(Chem.MolFromPDBFile(protein_path, removeHs=False))
    ligand = Chem.SDMolSupplier(ligand_path, removeHs=False)
    ligand = prolif.Molecule.from_rdkit(ligand[0])

    # Compute the protein-ligand interaction fingerprint
    fp = prolif.Fingerprint(interactions=list(config.prolif_weights.keys()))
    fp.run_from_iterable([ligand], protein, progress=False)

    try:  # TO-DO get rid of the try loop
        # Convert fingerprint to DataFrame and compute cumulative score
        df = fp.to_dataframe()
        df_stacked = df.stack(level=[0, 1, 2])
        df_reset = df_stacked.to_frame().reset_index()
        df_reset.columns = ["Frame", "ligand", "protein", "interaction", "value"]
        df_reset["score"] = df_reset["interaction"].apply(
            lambda x: config.prolif_weights[x]
        )
        return df_reset["score"].sum()
    except:
        # Handle cases with no meaningful interactions
        return 0


def score_ligands(config: Config) -> Dict[str, float]:
    """
    Scores all ligands in the specified directory based on their interactions with a given protein.

    Parameters:
    - config (dict): Configuration dictionary containing paths and other settings.

    Returns:
    - dict: A dictionary containing ligand names as keys and their scores as values.
    """
    assert isinstance(
        (protein_path := config.cycle_temp_params["path_to_protein"]), str
    ), ".set_scoring_parameters() wasn't called"
    ligand_paths_list = [
        config.cycle_temp_params["path_to_poses"] + lig
        for lig in os.listdir(config.cycle_temp_params["path_to_poses"])
        if lig.endswith(".sdf")
    ]
    ligand_scores = {}
    pbar = tqdm(ligand_paths_list, total=len(ligand_paths_list))
    for ligand_path in pbar:
        if (name := ligand_path.split("/")[-1].split(".")[0]) not in ligand_scores:
            score = score_protein_ligand_pose(config, protein_path, ligand_path)
            ligand_scores[name] = score
    return ligand_scores


def parse_and_prepare_diffdock_data(
    ligand_scores: Dict[str, float],
    config: Config,
    scored_db: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Filter and prepare the diffdock data based on ligand scores.

    Parameters:
    - ligand_scores (dict): Dictionary of ligand scores.
    - config (dict): Configuration dictionary containing paths and other settings.
    - scored_db (dict): A dictionary containing already scored ligands, to avoid re-scoring.

    Returns:
    - pd.DataFrame: Dataframe containing parsed and prepared data.
    """
    if scored_db is None:
        scored_db = {}
    sampled_mols = pd.read_csv(config.cycle_temp_params["path_to_sampled"])
    # if lower_percentile is not None:
    # threshold = np.percentile(list(ligand_scores.values()), lower_percentile)
    all_ligands = {
        int(complex_name[7:]): score for complex_name, score in ligand_scores.items()
    }
    getter = lambda x: scored_db.get(x, all_ligands.get(x, 0))
    sampled_mols["score"] = [
        getter(complex_number) for complex_number in sampled_mols.index
    ]
    sampled_mols.to_csv(config.cycle_temp_params["path_to_scored"])
    # good_ligands = diffdock_samples[diffdock_samples["score"] >= threshold]
    # good_ligands.to_csv(config["path_to_good_mols"])
    return sampled_mols
