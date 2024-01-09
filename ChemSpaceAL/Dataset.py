import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import yaml  # type:ignore
import re
from typing import Optional, List, Tuple

from . import Configuration


class SMILESDataset(Dataset):
    """
    A custom Dataset class for handling SMILES (Simplified Molecular Input Line Entry System) strings.
    """

    def __init__(self):
        self.desc_only: bool

    def _load_dataset(
        self,
        data: List[str],
        chars: List[str],
        block_size: int,
        len_data: int,
        regex_pattern: str,
    ):
        """
        Initializes the dataset.

        Parameters:
            data: List of SMILES strings.
            chars: Characters to build vocabulary.
            block_size: Size of the block for processing.
            len_data: Length of the data.
        """
        self.desc_only = False
        self.vocab = set(chars)
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

        self.data = data
        self.block_size = block_size
        self.len_data = len_data
        self.regex_pattern = regex_pattern

    def export_descriptors(self, export_path: str):
        """
        Exports the dataset descriptors to a file using the YAML format.

        Parameters:
            export_path: Path to save the descriptors.
        """
        attr_dict = {
            "desc_only": self.desc_only,
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "stoi": self.stoi,
            "itos": self.itos,
            "len_data": self.len_data,
        }
        with open(export_path, "w") as f:
            yaml.dump(attr_dict, f)

    def load_desc_attributes(self, load_path: str):
        """
        Loads dataset descriptors from a YAML file and updates the object's attributes.

        Parameters:
            load_path: Path to load the descriptors from.
        """
        self.desc_only = True
        with open(load_path, "r") as f:
            attr_dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.__dict__.update(attr_dict)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        assert (
            not self.desc_only
        ), "Dataset wasn't loaded, only descriptors are available"
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the processed SMILES string at a given index.

        Parameters:
            idx: Index to retrieve the data.

        Returns:
            x, y: Processed input and target tensors.
        """
        assert (
            not self.desc_only
        ), "Dataset wasn't loaded, only descriptors are available"
        smiles = self.data[idx].strip()
        regex = re.compile(self.regex_pattern)
        smiles_matches = regex.findall(smiles)

        # do we really need to check this though?
        # assert (
        #     len(set(smiles_matches)) <= self.block_size
        # ), f"A smiles {smiles=} {len(smiles)} tokens was loaded, but the block size is limited to {self.block_size}"

        # if len(smiles_matches) > self.block_size + 1:
        # smiles = smiles[: self.block_size + 1]

        embedded_smile = [self.stoi[s] for s in smiles_matches]
        x = torch.tensor(embedded_smile[:-1], dtype=torch.long)
        y = torch.tensor(embedded_smile[1:], dtype=torch.long)
        return x, y


def load_data(
    config: Configuration.Config,
    mode: str,
    forced_block_size: Optional[int] = None,
    forced_vocab: Optional[List[str]] = None,
):
    """
    Load data based on the provided configuration dictionary.

    Parameters:
    - config (dict): Configuration dictionary containing data parameters.
    - forced_block_size (int, optional): Forced block size, should be provided for Active Learning mode only.
    - forced_vocab (list, optional): Forced vocabulary, should be provided for Active Learning mode only.

    Returns:
    - dataset (SMILESDataset): SMILES dataset for the given mode.
    """
    compression = "gzip" if "gz" in config.training_fname else None

    if mode == "Pretraining":
        train_data = pd.read_csv(
            config.pretrain_data_path + config.training_fname, compression=compression
        )
        val_data = pd.read_csv(
            config.pretrain_data_path + config.validation_fname, compression=compression
        )

        if config.slice_data:
            train_data = train_data[: config.slice_data]
            val_data = val_data[: config.slice_data]

        smiles_iterators: List[np.ndarray] = [
            train_data[config.smiles_key].values,
            val_data[config.smiles_key].values,
        ]
        desc_path = (
            config.pretrain_desc_path + config.training_fname.split(".")[0] + ".yaml"
        )
    # Handle data loading for 'Active Learning' mode
    elif mode == "Active Learning":
        assert (
            al_path := config.cycle_temp_params["path_to_al_training_set"]
        ) is not None, (
            f"The name of the AL training set (al_train_fname) was not initialized"
        )
        cur_iter = f"al{config.al_iteration}"
        prev_iter = f"al{config.al_iteration - 1}"
        al_path = al_path.replace(cur_iter, prev_iter)
        print(f"Will load AL training set from", config.rel_path(al_path))
        al_data = pd.read_csv(al_path)
        smiles_iterators = [al_data[config.smiles_key].values]
        # desc_path = config.al_desc_path + al_fname.split(".")[0] + ".yaml"
        desc_path = (
            config.pretrain_desc_path + config.training_fname.split(".")[0] + ".yaml"
        )
    else:
        raise KeyError(
            f"Only 'pretraining' and 'active learning' modes are currently supported"
        )

    regex = re.compile(config.regex_pattern)
    char_set = {"!", "~", "<"}  # start, end, padding tokens respectively

    max_len = 0
    for smiles in smiles_iterators:
        for smile in smiles:
            chars = regex.findall(smile.strip())
            max_len = max(max_len, len(chars))
            char_set.update(chars)

    chars = sorted(list(char_set))
    max_len += 1

    if forced_block_size:
        assert mode == "Active Learning", "Cannot force a block size in pretraining"
        max_len = forced_block_size

    if forced_vocab:
        assert mode == "Active Learning", "Cannot force a vocabulary in pretraining"
        chars = sorted(list(forced_vocab))

    datasets = []
    for smiles in smiles_iterators:
        padded_data = [
            "!" + smile + "~" + "<" * (max_len - 1 - len(regex.findall(smile.strip())))
            for smile in smiles
        ]
        dataset = SMILESDataset()
        dataset._load_dataset(
            data=padded_data,
            chars=chars,
            block_size=max_len,
            len_data=len(smiles),
            regex_pattern=config.regex_pattern,
        )
        datasets.append(dataset)

    datasets[0].export_descriptors(desc_path)

    if mode == "Active Learning":
        return datasets[0]
    else:
        return datasets
