import torch
from torch.nn import functional as F
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Fragments

from ChemSpaceAL.Model import GPT
from ChemSpaceAL.Configuration import Config, AdmetDict
from ChemSpaceAL.Dataset import SMILESDataset

from typing import Set, List, Callable, Union, Optional, Any, Dict


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

        target_criterion = mconf.generation_params["target_criterion"]
        force_filters = mconf.generation_params["force_filters"]
        if "ADMET" in force_filters:
            satisfies_admet = admet_checker(mconf.generation_params["admet_criteria"])
        if "FGs" in force_filters:
            contains_restricted_fg = restricted_fg_checker(
                mconf.generation_params["restricted_fgs"]
            )
        for gen_mol in y:
            if target_criterion == "force_number_completions":
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
                if target_criterion == "force_number_unique":
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
        match target_criterion:
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


def get_mol(smile_string: str) -> Union[None, rdkit.Chem.Mol]:
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
    comparison_set: Set[str],
    reference_sets: List[Set[str]],
    sig_digs: int = 3,
    multiplier: int = 100,
    denominator: Optional[int] = None,
    subtracted: bool = True,
    show_work: bool = False,
) -> Union[str, float]:
    """Check novelty of generated molecules.

    Args:
        generated (set): Set of generated molecules.
        train_list (list): List of training sets.
        ... (other args): Various parameters for computation and formatting.

    Returns:
        str or float: Novelty value or a formula (depending on show_work).
    """
    combined_reference_set: Set[str] = set()
    for r_set in reference_sets:
        combined_reference_set = combined_reference_set | r_set
    repeated = comparison_set & combined_reference_set

    if denominator is None:
        denominator = len(comparison_set)

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


def dump_dic_to_text(dic: Dict[str, Any], path: str, header: Optional[str] = None):
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


def characterize_generated_molecules(config: Config):
    """Characterize generated molecules based on certain metrics.

    Args:
        config_dict (dict): Configuration parameters.
        molecules_list (list, optional): List of molecules. Defaults to None.
    """
    completions = pd.read_csv(config.cycle_temp_params["completions_fname"])["smiles"]
    if config.model_config.generation_params["force_filters"] is not None:
        key = "filtered_smiles_fname"
    else:
        key = "unique_smiles_fname"
    molecules_set = set(pd.read_csv(config.cycle_temp_params[key])["smiles"])

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

    train_data = set(
        pd.read_csv(config.pretrain_data_path + config.training_fname)[
            config.smiles_key
        ]
    )
    assert (
        config.previously_scored_mols is not None
    ), "Please call .set_previous_arrays() on config before trying to characterize the generated molecules"
    scored_sets = {
        i: set(pd.read_csv(path)["smiles"])
        for i, path in enumerate(config.previously_scored_mols)
    }
    assert (
        config.previous_al_train_sets is not None
    ), "Please call .set_previous_arrays() on config before trying to characterize the generated molecules"
    al_sets = {
        i: set(pd.read_csv(path)["smiles"])
        for i, path in enumerate(config.previous_al_train_sets)
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
            comparison_set=molecules_set,
            reference_sets=[train_data],
            multiplier=multiplier,
        ),
        "% novelty (rel. to train+AL sets)": check_novelty(
            comparison_set=molecules_set,
            reference_sets=[train_data, *list(al_sets.values())],
            multiplier=multiplier,
        ),
    }

    for al_round, al_set in al_sets.items():
        metrics[f"% repetitions (from AL{al_round} training set)"] = check_novelty(
            comparison_set=molecules_set,
            reference_sets=[al_set],
            subtracted=False,
            multiplier=multiplier,
            show_work=True,
        )

    for score_round, score_set in scored_sets.items():
        metrics[
            f"% repetitions (from scored from round {score_round})"
        ] = check_novelty(
            comparison_set=molecules_set,
            reference_sets=[score_set],
            subtracted=False,
            multiplier=multiplier,
            show_work=True,
        )

    for al_round, al_set in al_sets.items():
        metrics[
            f"% fraction of AL{al_round} training set in generated"
        ] = check_novelty(
            comparison_set=molecules_set,
            reference_sets=[al_set],
            subtracted=False,
            multiplier=multiplier,
            denominator=len(al_set),
            show_work=True,
        )

    for score_round, score_set in scored_sets.items():
        metrics[
            f"% fraction of scored from round {score_round} in generated"
        ] = check_novelty(
            comparison_set=molecules_set,
            reference_sets=[score_set],
            subtracted=False,
            multiplier=multiplier,
            denominator=len(score_set),
            show_work=True,
        )
    assert (
        config.cycle_temp_params["generation_metrics_fname"] is not None
    ), f"The name of the metrics file (generation_metrics_fname) was not initialized"
    dump_dic_to_text(metrics, config.cycle_temp_params["generation_metrics_fname"])
