import torch
import os
from typing import Union, Dict, Any, List, Tuple, Optional


class Config:
    """
    Configuration class for ChemSpace Active Learning.
    Holds various settings and parameters required for the ChemSpaceAL workflow.
    """

    def __init__(
        self,
        base_path: str,  # =os.getcwd() + "/",
        training_filename: str,  # ="training.csv.gz",
        validation_filename: str,  # ="validation.csv.gz",
        mode: str = "Pretraining",
        slice_data: Union[
            bool, int
        ] = False,  # if False, use all data, if an integer N, data will be sliced as data[:N]
        n_head: int = 8,
        n_embed: int = 256,
        att_bias: bool = False,
        att_drop_rate: float = 0.1,
        do_flash: bool = True,
        ff_mult: int = 4,  # ff_mult * n_embed = n_hidden units in the feedforward layer of the transformer
        doGELU: bool = True,  # otherwise use ReLU (potential todo: make this a string argument)
        gpt_drop_rate: float = 0.1,  # the p-parameter of Dropout after embedding
        n_layer: int = 8,
        gpt_bias: bool = True,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.965, 0.99),  # SophiaG hyperparameters
        rho: float = 0.04,  # SophiaG hyperparameter
        batch_size: int = 512,
        num_workers: int = 0,
        lr_decay: bool = True,
        pretraining_checkpoint_file: str = "pretraining_checkpoint.pt",
        al_checkpoint_file: str = "al_checkpoint.pt",
        project_name: str = "ChemSpaceAL",
        wandb_runname: str = "pretraining_run",
        context: str = "!",
        gen_batch_size: int = 64,
        temp: float = 1.0,
        gen_size: int = 10_000,
        completions_file: str = "completions.csv",
        predicted_file: str = "predicted.csv",
        predicted_filtered_file: str = "filtered_predicted.csv",
        smiles_key: str = "smiles",
        previously_scored_mols: Optional[List] = None,
        previous_al_train_sets: Optional[List] = None,
        metrics_file:str="metrics.txt",
        gen_mol_descriptors_file:str="generated_molecules_descriptors.csv",
        pca_file:str="pca.pkl",
        kmeans_save_file:str="kmeans.pkl",
        clusters_save_file:str="clusters.pkl",
        samples_save_file:str="samples.pkl",
        diffdock_save_file:str="diffdock.pkl",
        protein_file:str="protein.pdb",
        diffdock_samples_file:str="sampled.csv",
        scored_file:str="scored.csv",
        good_mols_file:str="good_mols.csv",
        AL_set_save_file:str="AL_training_set.csv",
        AL_training_file:str="AL_training_set.csv",
    ):
        # Setting model and training configurations
        self.mode = mode
        self.base_path = base_path
        self.training_filename = training_filename
        self.slice_data = slice_data
        self.validation_filename = validation_filename
        self.n_head = n_head
        self.n_embed = n_embed
        self.att_bias = att_bias
        self.att_drop_rate = att_drop_rate
        self.do_flash = do_flash
        self.ff_mult = ff_mult
        self.doGELU = doGELU
        self.gpt_drop_rate = gpt_drop_rate
        self.n_layer = n_layer
        self.gpt_bias = gpt_bias
        self.weight_decay = weight_decay
        self.betas = betas
        self.rho = rho
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Setting device based on availability
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Miscellaneous configurations
        self.lr_decay = lr_decay
        self.pretraining_checkpoint_file = pretraining_checkpoint_file
        self.al_checkpoint_file = al_checkpoint_file
        self.project_name = project_name
        self.wandb_runname = wandb_runname
        self.context = context
        self.gen_batch_size = gen_batch_size
        self.temp = temp
        self.gen_size = gen_size

        # File configurations
        self.completions_file = completions_file
        self.predicted_file = predicted_file
        self.predicted_filtered_file = predicted_filtered_file
        self.smiles_key = smiles_key
        self.previously_scored_mols = previously_scored_mols
        self.previous_al_train_sets = previous_al_train_sets
        self.metrics_file = metrics_file
        self.gen_mol_descriptors_file = gen_mol_descriptors_file
        self.pca_file = pca_file
        self.kmeans_save_file = kmeans_save_file
        self.clusters_save_file = clusters_save_file
        self.samples_save_file = samples_save_file
        self.diffdock_save_file = diffdock_save_file
        self.protein_file = protein_file
        self.diffdock_samples_file = diffdock_samples_file
        self.scored_file = scored_file
        self.good_mols_file = good_mols_file
        self.AL_set_save_file = AL_set_save_file
        self.AL_training_file = AL_training_file

        # Configuration dictionary for convenience
        self.config_dict: Dict[str, Any] = {}
        self.set_config_dict()
        self.update_config_dict()

    def set_config_dict(self):
        """Set the configuration dictionary with paths and parameters."""

        base_pretraining = self.base_path + "1. Pretraining/datasets/"
        base_gen_path = self.base_path + "2. Generation/"
        base_sampling_path = self.base_path + "3. Sampling/"
        base_diffdock_path = self.base_path + "4. Diffdock/"
        base_scoring = self.base_path + "5. Scoring/scored_molecules/"
        base_active_learning = self.base_path + "6. ActiveLearning/"
        base_model_parameters = self.base_path + "ModelParameters/"

        self.config_dict = {
            "mode": self.mode,
            "train_path": base_pretraining + self.training_filename,
            "slice_data": self.slice_data,
            "val_path": base_pretraining + self.validation_filename,
            "n_head": self.n_head,
            "n_embed": self.n_embed,
            "att_bias": self.att_bias,
            "att_drop_rate": self.att_drop_rate,
            "do_flash": self.do_flash,
            "ff_mult": self.ff_mult,
            "doGELU": self.doGELU,
            "gpt_drop_rate": self.gpt_drop_rate,
            "n_layer": self.n_layer,
            "gpt_bias": self.gpt_bias,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "rho": self.rho,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "device": self.device,
            "lr_decay": self.lr_decay,
            "pretraining_checkpoint_path": base_model_parameters
            + self.pretraining_checkpoint_file,
            "al_checkpoint_path": base_model_parameters + self.al_checkpoint_file,
            "wandb_project": self.project_name,
            "wandb_runname": self.wandb_runname,
            "generation_context": self.context,
            "gen_batch_size": self.gen_batch_size,
            "inference_temp": self.temp,
            "gen_size": self.gen_size,
            "path_to_completions": base_gen_path + self.completions_file,
            "path_to_predicted": base_gen_path + self.predicted_file,
            "path_to_predicted_filtered": base_gen_path + self.predicted_filtered_file,
            "base_path": self.base_path,
            "smiles_key": self.smiles_key,
            "diffdock_scored_path_list": [
                f"{self.base_path}{base_scoring}{i}"
                for i in self.previously_scored_mols
            ],
            "al_trainsets_path_list": [
                f"{self.base_path}{base_active_learning}/{i}"
                for i in self.previous_al_train_sets
            ],
            "path_to_metrics": base_gen_path + self.metrics_file,
            "generation_path": base_gen_path,
            "path_to_gen_mol_descriptors": base_gen_path
            + self.gen_mol_descriptors_file,
            "path_to_pca": base_sampling_path + self.pca_file,
            "kmeans_save_path": base_sampling_path + self.kmeans_save_file,
            "clusters_save_path": base_sampling_path + self.clusters_save_file,
            "samples_save_path": base_sampling_path + self.samples_save_file,
            "diffdock_save_path": base_sampling_path + self.diffdock_save_file,
            "diffdock_results_path": base_diffdock_path + "poses/",
            "protein_path": self.base_path + self.protein_file,
            "diffdock_samples_path": base_diffdock_path + self.diffdock_samples_file,
            "path_to_scored": base_diffdock_path + self.scored_file,
            "path_to_good_mols": base_diffdock_path + self.good_mols_file,
            "AL_set_save_path": base_active_learning + self.AL_set_save_file,
            "AL_training_path": base_active_learning + self.AL_training_file,
        }

    def update_config_dict(self):
        """Update the configuration dictionary based on the mode."""

        # Base paths for Pretraining and Active Learning
        base_pretraining_descriptors = (
            self.base_path + "1. Pretraining/dataset_descriptors/"
        )
        base_active_learning_descriptors = (
            self.base_path + "6. ActiveLearning/dataset_descriptors/"
        )

        if self.mode == "Pretraining":
            descriptors_file = self.training_filename.split(".")[0] + ".yaml"
            self.config_dict.update(
                {
                    "lr_warmup": True,
                    "epochs": 30,
                    "learning_rate": 3e-4,
                    "descriptors_path": base_pretraining_descriptors + descriptors_file,
                }
            )

        elif self.mode == "Active Learning":
            descriptors_file = self.AL_training_file.split(".")[0] + ".yaml"
            self.config_dict.update(
                {
                    "lr_warmup": False,
                    "epochs": 10,
                    "learning_rate": 3e-5,
                    "descriptors_path": base_active_learning_descriptors
                    + descriptors_file,
                }
            )

        else:
            raise KeyError(
                f"requested {self.mode} but only Pretraining and Active Learning are supported"
            )
