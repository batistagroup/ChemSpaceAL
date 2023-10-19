# from rdkit import Chem
# from rdkit.Chem import Fragments
# import numpy as np
# from openpyxl import load_workbook

# from .Dataset import SMILESDataset
# from .Model import GPT, GPTConfig
# from .Configuration import *
import torch
from torch.nn import functional as F
import re
from tqdm import tqdm
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Fragments

from ChemSpaceAL.Model import GPT
from ChemSpaceAL.Configuration import Config, AdmetDict
from ChemSpaceAL.Dataset import SMILESDataset

from typing import Set, List, Callable


@torch.no_grad()
def sample(
    model: GPT, x: torch.Tensor, steps: int, temperature: float = 1.0
) -> torch.Tensor:
    """Sample sequences from the model.

    Args:
        model (GPT): The GPT model.
        x (torch.Tensor): Input tensor.
        steps (int): Number of sampling steps.
        temperature (float): Sampling temperature.

    Returns:
        torch.Tensor: The generated sequences.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, 1)
        x = torch.cat((x, ix), dim=1)
    return x


def restricted_fg_checker(restricted_fgs: List[str]) -> Callable:
    def _check_mol(molecule: rdkit.Chem.Mol) -> bool:
        for restricted_key in restricted_fgs:
            method = getattr(Fragments, restricted_key)
            if method(molecule) != 0:
                return True
        return False

    return _check_mol


def admet_checker(admet_criteria: AdmetDict) -> Callable:
    def _check_mol(
        molecule: rdkit.Chem.Mol,
    ) -> bool:
        for key, _dict in admet_criteria.items():
            func = _dict["func"]
            assert callable(
                func
            ), f"Expected a callable for criteria {key}, got {type(func)}"
            val = func(molecule)
            if val < _dict["lower"] or val > _dict["upper"]:
                return False
        return True

    return _check_mol


def generate_smiles(config: Config):
    """Generate SMILES strings using the model.

    Args:
        config_dict (dict): Configuration dictionary.

    Returns:
        None: The function saves the generated molecules to disk.
    """
    regex = re.compile(config.regex_pattern)
    mconf = config.model_config
    if config.verbose:
        print(f"--- Starting generation")
        print(f"    Loading dataset descriptors ...")
    dataset = SMILESDataset()
    dataset.load_desc_attributes(mconf.generation_params["desc_path"])
    if config.verbose:
        print(f"    Creating a model and loading weights ...")
    mconf.set_dataset_attributes(
        vocab_size=dataset.vocab_size, block_size=dataset.block_size
    )
    model = GPT(mconf).to(mconf.device)
    model.load_state_dict(
        torch.load(
            mconf.generation_params["load_ckpt_path"],
            map_location=torch.device(mconf.device),
        )
    )
    model.to(mconf.device)
    torch.compile(model)

    block_size = model.get_block_size()
    assert (
        block_size == dataset.block_size
    ), "Warning: model block size and dataset block size are different"

    molecules_list: List[str] = []
    molecules_set: Set[str] = set()
    molecules_set_filtered: Set[str] = set()
    completions = []
    pbar = tqdm()
    if config.verbose:
        print(f"    Starting generation ...")
    while True:
        x = (
            torch.tensor(
                [
                    dataset.stoi[s]
                    for s in regex.findall(mconf.generation_params["context"])
                ],
                dtype=torch.long,
            )[None, ...]
            .repeat(mconf.generation_params["batch_size"], 1)
            .to(mconf.device)
        )
        y = sample(model, x, block_size, temperature=mconf.generation_params["temp"])

        target_criterium = mconf.generation_params["target_criterium"]
        force_filters = mconf.generation_params["force_filters"]
        if "ADMET" in force_filters:
            satisfies_admet = admet_checker(mconf.generation_params["admet_criteria"])
        if "FGs" in force_filters:
            contains_restricted_fg = restricted_fg_checker(
                mconf.generation_params["restricted_fgs"]
            )
        for gen_mol in y:
            if target_criterium == "force_number_completions":
                pbar.update()
                pbar.set_description(f"Generated {len(molecules_list)} completions")
            completion = "".join([dataset.itos[int(i)] for i in gen_mol])
            completions.append(completion)
            if completion[0] == "!" and completion[1] == "~":
                completion = "!" + completion[2:]
            if "~" not in completion:
                continue
            mol_string = completion[1 : completion.index("~")]
            mol = get_mol(mol_string)

            if mol is not None:
                if target_criterium == "force_number_unique":
                    pbar.update()
                    pbar.set_description(
                        f"Generated {len(molecules_set)} unique canonical smiles"
                    )
                canonic_smile = Chem.MolToSmiles(mol)
                molecules_list.append(canonic_smile)
                molecules_set.add(canonic_smile)
                if force_filters is not None:
                    mol_passes = True
                    if "ADMET" in force_filters:
                        if not satisfies_admet(mol):
                            mol_passes = False
                    if "FGs" in force_filters:
                        if contains_restricted_fg(mol):
                            mol_passes = False
                    if mol_passes:
                        pbar.update()
                        pbar.set_description(
                            f"Generated {len(molecules_set_filtered)} unique canonical smiles that pass filters"
                        )
                        molecules_set_filtered.add(canonic_smile)
        target_number = mconf.generation_params["target_number"]
        match target_criterium:
            case "force_number_completions":
                if len(molecules_list) >= target_number:
                    break
            case "force_number_unique":
                if len(molecules_set) >= target_number:
                    break
            case "force_number_filtered":
                if len(molecules_set_filtered) >= target_number:
                    break
    pbar.close()

    completions_df = pd.DataFrame({"smiles": completions})
    completions_df.to_csv(config.cycle_temp_params["completions_fname"])

    molecules_df = pd.DataFrame({"smiles": list(molecules_set)})
    molecules_df.to_csv(config.cycle_temp_params["unique_smiles_fname"])

    if force_filters is not None:
        molecules_filtered_df = pd.DataFrame({"smiles": list(molecules_set_filtered)})
        molecules_filtered_df.to_csv(config.cycle_temp_params["filtered_smiles_fname"])

    # characterize_generated_molecules(config_dict, molecules_list)


def get_mol(smile_string):
    """Get a molecule object from a SMILES string.

    Args:
        smile_string (str): The SMILES string.

    Returns:
        rdkit.Chem.Mol: The molecule object or None if invalid.
    """
    mol = Chem.MolFromSmiles(smile_string)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return mol


def check_novelty(
    generated,
    train_list,
    sig_digs=3,
    multiplier=100,
    denominator=None,
    subtracted=True,
    show_work=False,
):
    """Check novelty of generated molecules.

    Args:
        generated (set): Set of generated molecules.
        train_list (list): List of training sets.
        ... (other args): Various parameters for computation and formatting.

    Returns:
        str or float: Novelty value or a formula (depending on show_work).
    """
    total_train = set()
    for train in train_list:
        total_train = total_train | train
    repeated = generated & total_train

    if denominator is None:
        denominator = len(generated)

    # Formula to compute novelty
    value = (
        (1 - len(repeated) / denominator) if subtracted else len(repeated) / denominator
    )
    out = np.round(multiplier * value, sig_digs)

    if show_work:
        term = len(repeated)
        formula = (
            f"{multiplier} * (1-{term}/{denominator})"
            if subtracted
            else f"{multiplier} * {term}/{denominator}"
        )
        return f"{formula} = {out}"
    else:
        return out


def dump_dic_to_text(dic, path, header=None):
    """Dump a dictionary to a text file.

    Args:
        dic (dict): The dictionary.
        path (str): Path to save the file.
        header (str, optional): Header string.
    """
    with open(path, "w") as f:
        if header is not None:
            f.write(f"{header}\n")
        for key, value in dic.items():
            f.write(f"{key}: {value}\n")


def export_metrics_to_workbook(metrics, config, fname):
    """Export metrics to an Excel workbook.

    Args:
        metrics (dict): Metrics dictionary.
        config (dict): Configuration dictionary.
        fname (str): Filename.
    """
    metric_to_col = {
        "generated": "B",
        "valid": "C",
        "unique": "D",
        "validity": "E",
        "% unique (rel. to generated)": "F",
        "% unique (rel. to valid)": "G",
        "% novelty (rel. to train set)": "H",
        "% novelty (rel. to train+AL sets)": "I",
        "% repetitions (from AL0 training set)": "J",
        "% repetitions (from AL1 training set)": "K",
        "% repetitions (from AL2 training set)": "L",
        "% repetitions (from AL3 training set)": "M",
        "% repetitions (from AL4 training set)": "N",
        "% repetitions (from AL5 training set)": "O",
        "% repetitions (from AL6 training set)": "P",
        "% repetitions (from AL7 training set)": "Q",
        "% repetitions (from scored from round 0)": "R",
        "% repetitions (from scored from round 1)": "S",
        "% repetitions (from scored from round 2)": "T",
        "% repetitions (from scored from round 3)": "U",
        "% repetitions (from scored from round 4)": "V",
        "% repetitions (from scored from round 5)": "W",
        "% repetitions (from scored from round 6)": "X",
        "% repetitions (from scored from round 7)": "Y",
        "% fraction of AL0 training set in generated": "Z",
        "% fraction of AL1 training set in generated": "AA",
        "% fraction of AL2 training set in generated": "AB",
        "% fraction of AL3 training set in generated": "AC",
        "% fraction of AL4 training set in generated": "AD",
        "% fraction of AL5 training set in generated": "AE",
        "% fraction of AL6 training set in generated": "AF",
        "% fraction of AL7 training set in generated": "AG",
        "% fraction of scored from round 0 in generated": "AH",
        "% fraction of scored from round 1 in generated": "AI",
        "% fraction of scored from round 2 in generated": "AJ",
        "% fraction of scored from round 3 in generated": "AK",
        "% fraction of scored from round 4 in generated": "AL",
        "% fraction of scored from round 5 in generated": "AM",
        "% fraction of scored from round 6 in generated": "AN",
        "% fraction of scored from round 7 in generated": "AO",
    }


def characterize_generated_molecules(config_dict, molecules_list=None):
    """Characterize generated molecules based on certain metrics.

    Args:
        config_dict (dict): Configuration parameters.
        molecules_list (list, optional): List of molecules. Defaults to None.
    """
    completions = pd.read_csv(config_dict["path_to_completions"])["smiles"]
    molecules_set = set(
        pd.read_csv(config_dict["path_to_predicted_filtered"])["smiles"]
    )

    if molecules_list is None:
        molecules_list = []
        for completion in tqdm(completions, total=len(completions)):
            if completion[0] == "!" and completion[1] == "~":
                completion = "!" + completion[2:]
            if "~" not in completion:
                continue
            mol_string = completion[1 : completion.index("~")]
            mol = get_mol(mol_string)
            if mol is not None:
                molecules_list.append(Chem.MolToSmiles(mol))

    train_data = set(pd.read_csv(config_dict["train_path"])[config_dict["smiles_key"]])
    scored_sets = {
        i: set(pd.read_csv(path)["smiles"])
        for i, path in enumerate(config_dict["diffdock_scored_path_list"])
    }
    al_sets = {
        i: set(pd.read_csv(path)["smiles"])
        for i, path in enumerate(config_dict["al_trainsets_path_list"])
    }

    multiplier = 100
    metrics = {
        "generated": len(completions),
        "valid": len(molecules_list),
        "unique": len(molecules_set),
        "validity": np.round(multiplier * len(molecules_list) / len(completions), 3),
        "% unique (rel. to generated)": np.round(
            multiplier * len(molecules_set) / len(completions), 3
        ),
        "% unique (rel. to valid)": np.round(
            multiplier * len(molecules_set) / len(molecules_list), 3
        ),
        "% novelty (rel. to train set)": check_novelty(
            molecules_set, (train_data,), multiplier=multiplier
        ),
        "% novelty (rel. to train+AL sets)": check_novelty(
            molecules_set, (train_data, *list(al_sets.values())), multiplier=multiplier
        ),
    }

    for al_round, al_set in al_sets.items():
        metrics[f"% repetitions (from AL{al_round} training set)"] = check_novelty(
            molecules_set,
            (al_set,),
            subtracted=False,
            multiplier=multiplier,
            show_work=True,
        )

    for score_round, score_set in scored_sets.items():
        metrics[
            f"% repetitions (from scored from round {score_round})"
        ] = check_novelty(
            molecules_set,
            (score_set,),
            subtracted=False,
            multiplier=multiplier,
            show_work=True,
        )

    for al_round, al_set in al_sets.items():
        metrics[
            f"% fraction of AL{al_round} training set in generated"
        ] = check_novelty(
            molecules_set,
            (al_set,),
            subtracted=False,
            multiplier=multiplier,
            denominator=len(al_set),
            show_work=True,
        )

    for score_round, score_set in scored_sets.items():
        metrics[
            f"% fraction of scored from round {score_round} in generated"
        ] = check_novelty(
            molecules_set,
            (score_set,),
            subtracted=False,
            multiplier=multiplier,
            denominator=len(score_set),
            show_work=True,
        )

    dump_dic_to_text(metrics, config_dict["path_to_metrics"])
    export_metrics_to_workbook(
        metrics, config_dict, config_dict["generation_path"].split("/")[-1]
    )
