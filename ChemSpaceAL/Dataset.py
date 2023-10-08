from torch.utils.data import Dataset
import yaml
import torch
import pandas as pd
import re

class SMILESDataset(Dataset):
    '''
    A custom Dataset class for handling SMILES (Simplified Molecular Input Line Entry System) strings.
    '''

    def __init__(self, data=None, chars=None, block_size=None, len_data=None):
        '''
        Initializes the dataset.
        
        Parameters:
            data: List of SMILES strings.
            chars: Characters to build vocabulary.
            block_size: Size of the block for processing.
            len_data: Length of the data.
        '''
        if chars is None:
            self.desc_only = True
            return
        
        self.desc_only = False
        self.vocab = set(chars)
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

        self.block_size = block_size
        self.data = data
        self.len_data = len_data

    def export_descriptors(self, export_path):
        '''
        Exports the dataset descriptors to a file using the YAML format.

        Parameters:
            export_path: Path to save the descriptors.
        '''
        attr_dict = {
            'desc_only': self.desc_only,
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'stoi': self.stoi,
            'itos': self.itos,
            'len_data': self.len_data
        }
        with open(export_path, 'w') as f:
            yaml.dump(attr_dict, f)

    def load_desc_attributes(self, load_path):
        '''
        Loads dataset descriptors from a YAML file and updates the object's attributes.

        Parameters:
            load_path: Path to load the descriptors from.
        '''
        with open(load_path, 'r') as f:
            attr_dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.__dict__.update(attr_dict)

    def __len__(self):
        '''Returns the length of the dataset.'''
        assert not self.desc_only, 'Dataset is not initialized'
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Returns the processed SMILES string at a given index.

        Parameters:
            idx: Index to retrieve the data.

        Returns:
            x, y: Processed input and target tensors.
        '''
        assert not self.desc_only, 'Dataset is not initialized'
        smiles = self.data[idx].strip()
        REGEX_PATTERN = r'(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|@@|\?|>|!|~|\*|\$|%[0-9]{2}|[0-9])'
        regex = re.compile(REGEX_PATTERN)
        smiles_matches = regex.findall(smiles)

        if len(smiles_matches) > self.block_size + 1:
            smiles = smiles[:self.block_size + 1]

        embedded_smile = [self.stoi[s] for s in smiles_matches]
        x = torch.tensor(embedded_smile[:-1], dtype=torch.long)
        y = torch.tensor(embedded_smile[1:], dtype=torch.long)
        return x, y


def load_data(config_dict, forced_block_size=None, forced_vocab=None):
    """
    Load data based on the provided configuration dictionary.

    Parameters:
    - config_dict (dict): Configuration dictionary containing data parameters.
    - forced_block_size (int, optional): Forced block size, should be provided for Active Learning mode only.
    - forced_vocab (list, optional): Forced vocabulary, should be provided for Active Learning mode only.

    Returns:
    - dataset (SMILESDataset): SMILES dataset for the given mode.
    """
    mode = config_dict['mode']

    # Check if the input data is in gz format
    compression = 'gzip' if 'gz' in config_dict["train_path"] else None

    # Handle data loading for 'Pretraining' mode
    if mode == 'Pretraining':
        slice_data = config_dict["slice_data"]
        train_data = pd.read_csv(config_dict["train_path"], compression=compression)
        val_data = pd.read_csv(config_dict["val_path"], compression=compression)

        # Optional slicing of the data
        if slice_data:
            train_data = train_data[:slice_data]
            val_data = val_data[:slice_data]

        iterators = (train_data[config_dict['smiles_key']].values, val_data[config_dict['smiles_key']].values)
        assert len(train_data) == len(train_data[config_dict['smiles_key']].values), "SMILES values count mismatch."

    # Handle data loading for 'Active Learning' mode
    elif mode == 'Active Learning':
        al_data = pd.read_csv(config_dict["AL_training_path"])
        iterators = (al_data[config_dict['smiles_key']].values, )

    else:
        raise KeyError(f"Only 'pretraining' and 'active learning' modes are currently supported")

    # Regular expression pattern for parsing SMILES
    REGEX_PATTERN = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|@@|\?|>|!|~|\*|\$|%[0-9]{2}|[0-9])"
    regex = re.compile(REGEX_PATTERN)
    char_set = {'<', '!', '~'}

    max_len = 0
    for iterator in iterators:
        for i in iterator:
            chars = regex.findall(i.strip())
            max_len = max(max_len, len(chars))
            char_set.update(chars)

    chars = sorted(list(char_set))
    max_len += 1

    if forced_block_size:
        assert mode == 'Active Learning', "Cannot force a block size in pretraining"
        max_len = forced_block_size

    if forced_vocab:
        assert mode == 'Active Learning', "Cannot force a vocabulary in pretraining"
        chars = sorted(list(forced_vocab))

    datasets = []
    for iterator in iterators:
        padded_data = ['!' + i + '~' + '<' * (max_len - 1 - len(regex.findall(i.strip()))) for i in iterator]
        dataset = SMILESDataset(data=padded_data, chars=chars, block_size=max_len, len_data=len(iterator))
        datasets.append(dataset)

    datasets[0].export_descriptors(config_dict["descriptors_path"])

    if mode == 'Active Learning':
        return datasets[0]

    return datasets   
