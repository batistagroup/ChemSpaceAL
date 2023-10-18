import torch
from torch.nn import Module
import os
from typing import Union, Dict, Any, List, Tuple, Optional, Any
from .InitializeWorkspace import FOLDER_STRUCTURE as fldr_struc
from pprint import pprint


class Config:
    """
    Configuration class for ChemSpace Active Learning.
    Holds various settings and parameters required for the ChemSpaceAL workflow.
    """

    def __init__(
        self,
        base_path: str,
        cycle_prefix: str,
        al_iteration: int,
        cycle_suffix: str,
        training_fname: str,
        validation_fname: str,
        smiles_key: str = "smiles",
        # mode: str = "Pretraining",
        slice_data: Union[bool, int] = False,
        # pretraining_checkpoint_file: str = "pretraining_checkpoint.pt",
        # al_checkpoint_file: str = "al_checkpoint.pt",
        # project_name: str = "ChemSpaceAL",
        # wandb_runname: str = "pretraining_run",
        context: str = "!",
        gen_size: int = 10_000,
        gen_batch_size: int = 64,
        temperature: float = 1.0,
        # completions_file: str = "completions.csv",
        # predicted_file: str = "predicted.csv",
        # predicted_filtered_file: str = "filtered_predicted.csv",
        previously_scored_mols: Optional[List] = None,
        previous_al_train_sets: Optional[List] = None,
        # metrics_file: str = "metrics.txt",
        # gen_mol_descriptors_file: str = "generated_molecules_descriptors.csv",
        # pca_file: str = "pca.pkl",
        # kmeans_save_file: str = "kmeans.pkl",
        # clusters_save_file: str = "clusters.pkl",
        # samples_save_file: str = "samples.pkl",
        # diffdock_save_file: str = "diffdock.pkl",
        # protein_file: str = "protein.pdb",
        # diffdock_samples_file: str = "sampled.csv",
        # scored_file: str = "scored.csv",
        # good_mols_file: str = "good_mols.csv",
        # AL_set_save_file: str = "AL_training_set.csv",
        # AL_training_file: str = "AL_training_set.csv",
        verbose: bool = True,
    ):
        # Setting model and training configurations

        self.base_path = base_path
        self.base_sep_count = base_path.count(os.sep)
        self.cycle_prefix = cycle_prefix
        self.al_iteration = al_iteration
        self.cycle_suffix = cycle_suffix

        # Pretraining configs
        self.training_fname = training_fname
        self.validation_fname = validation_fname
        # if slice_data is False, use all data, if an integer N, data will be sliced as data[:N]
        self.slice_data = slice_data
        self.smiles_key = smiles_key

        # Generation Parameters
        self.context = context
        self.gen_size = gen_size
        self.gen_batch_size = gen_batch_size
        self.temperature = temperature

        self.previously_scored_mols = previously_scored_mols
        self.previous_al_train_sets = previous_al_train_sets

        self.verbose = verbose
        self.regex_pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|@@|\?|>|!|~|\*|\$|\%[0-9]{2}|[0-9])"

        # Configuration dictionary for convenience
        # self.config_dict: Dict[str, Any] = {}
        self.cycle_temp_params: Dict[str, Optional[str]] = {
            "al_train_fname": None,
        }
        self.model_config = ModelConfig()

        self.set_config_paths()
        # self.update_config_dict()

    def set_config_paths(self):
        """Set the configuration dictionary with paths and parameters."""
        self.pretrain_data_path = os.path.join(
            self.base_path,
            "1_Pretraining",
            fldr_struc["1_Pretraining"]["dataset_folder"],
            "",
        )
        self.pretrain_weight_path = os.path.join(
            self.base_path,
            "1_Pretraining",
            fldr_struc["1_Pretraining"]["weight_folder"],
            "",
        )
        self.pretrain_desc_path = os.path.join(
            self.base_path,
            "1_Pretraining",
            fldr_struc["1_Pretraining"]["desc_folder"],
            "",
        )
        self.generations_path = os.path.join(
            self.base_path,
            "2_Generation",
            "",
        )
        self.sampling_desc_path = os.path.join(
            self.base_path,
            "3_Sampling",
            fldr_struc["3_Sampling"]["desc_folder"],
            "",
        )
        self.sampling_pca_path = os.path.join(
            self.base_path,
            "3_Sampling",
            fldr_struc["3_Sampling"]["pca_folder"],
            "",
        )
        self.sampling_kmeans_path = os.path.join(
            self.base_path,
            "3_Sampling",
            fldr_struc["3_Sampling"]["kmeans_folder"],
            "",
        )
        self.sampling_clusters_path = os.path.join(
            self.base_path,
            "3_Sampling",
            fldr_struc["3_Sampling"]["clustering_folder"],
            "",
        )
        self.scoring_target_path = os.path.join(
            self.base_path,
            "4_Scoring",
            fldr_struc["4_Scoring"]["target_folder"],
            "",
        )
        self.scoring_candidate_path = os.path.join(
            self.base_path,
            "4_Scoring",
            fldr_struc["4_Scoring"]["candidate_folder"],
            "",
        )
        self.scoring_pose_path = os.path.join(
            self.base_path,
            "4_Scoring",
            fldr_struc["4_Scoring"]["pose_folder"],
            "",
        )
        self.scoring_score_path = os.path.join(
            self.base_path,
            "4_Scoring",
            fldr_struc["4_Scoring"]["score_folder"],
            "",
        )
        self.al_train_path = os.path.join(
            self.base_path,
            "5_ActiveLearning",
            fldr_struc["5_ActiveLearning"]["train_folder"],
            "",
        )
        self.al_desc_path = os.path.join(
            self.base_path,
            "5_ActiveLearning",
            fldr_struc["5_ActiveLearning"]["desc_folder"],
            "",
        )
        self.al_weight_path = os.path.join(
            self.base_path,
            "5_ActiveLearning",
            fldr_struc["5_ActiveLearning"]["weight_folder"],
            "",
        )

    def set_training_parameters(
        self,
        mode: str,
        learning_rate: Optional[float] = None,
        lr_warmup: Optional[bool] = None,
        epochs: Optional[int] = None,
        # al_fname: Optional[str] = None,
        load_weight_path: Optional[str] = None,
        wandb_project_name: Optional[str] = None,
        wandb_runname: Optional[str] = None,
    ):
        if mode == "Pretraining":
            # desc_path = (
            #     self.pretrain_desc_path + self.training_fname.split(".")[0] + "yaml"
            # )
            save_model_weights = (
                self.pretrain_weight_path
                + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}.pt"
            )
            self.model_config.train_params = {
                "epochs": epochs if epochs is not None else 30,
                "learning_rate": learning_rate if learning_rate is not None else 3e-4,
                "lr_warmup": lr_warmup if lr_warmup is not None else True,
                "load_model_weight": self.pretrain_weight_path + load_weight_path
                if load_weight_path is not None
                else None,
                "save_model_weight": save_model_weights,
                # "descriptors_path": desc_path,
            }

        elif mode == "Active Learning":
            # if al_fname is None:
            #     if (al_fname := self.cycle_temp_params["al_fname"]) is None:
            #         raise ValueError(
            #             "The name of the Active Learning Set isn't stored in current session, please provide through al_fname argument"
            #         )
            # desc_path = self.al_desc_path + al_fname.split(".")[0] + "yaml"
            assert (
                self.al_iteration >= 1
            ), "al_iteration cannot be less than 1 in Active Learning mode"
            load_model_weights = (
                self.al_weight_path
                + f"{self.cycle_prefix}_al{self.al_iteration-1}_{self.cycle_suffix}.pt"
            )
            save_model_weights = (
                self.al_weight_path
                + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}.pt"
            )
            self.model_config.train_params = {
                "epochs": epochs if epochs is not None else 10,
                "learning_rate": learning_rate if learning_rate is not None else 3e-5,
                "lr_warmup": lr_warmup if lr_warmup is not None else False,
                "load_model_weight": load_model_weights,
                "save_model_weight": save_model_weights,
                # "descriptors_path": base_active_learning_descriptors + descriptors_file,
            }

        else:
            raise KeyError(
                f"requested {mode} but only Pretraining and Active Learning are supported"
            )
        if wandb_project_name is not None:
            self.model_config.train_params["wandb_project_name"] = wandb_project_name
            assert (
                wandb_runname is not None
            ), "You need to provide wandb_runname as well"
            self.model_config.train_params["wandb_runname"] = wandb_runname

        if self.verbose:
            message = f"""--- The following training parameters were set:
    number of epochs: {self.model_config.train_params['epochs']}
    learning rate: {self.model_config.train_params['learning_rate']}
    learning warmup enabled? {self.model_config.train_params['lr_warmup']}"""
            if (
                lpath := self.model_config.train_params["load_model_weight"]
            ) is not None:
                rel_path = os.sep.join(lpath.split(os.sep)[self.base_sep_count :])
                message += f"\n    model weights will be loaded from {rel_path}"
            if (
                spath := self.model_config.train_params["save_model_weight"]
            ) is not None:
                rel_path = os.sep.join(spath.split(os.sep)[self.base_sep_count :])
                message += f"\n    model weights will be saved to {rel_path}"
            if wandb_project_name is None:
                message += f"\n  . note: wandb_project_name and wandb_runname were not provided, you can ignore this message if you don't plan to log runs to wandb"
            else:
                message +=f"\n    runs will be stored in the {self.model_config.train_params['wandb_project_name']} wandb project"
                message += f"\n    under the name {self.model_config.train_params['wandb_runname']}"
            print(message)

        # print(self.pretrain_data_path)
        # print(self.pretrain_weight_path)
        # print(self.pretrain_desc_path)
        # print(self.generations_path)
        # print(self.sampling_desc_path)
        # print(self.sampling_pca_path)
        # print(self.sampling_kmeans_path)
        # print(self.sampling_clusters_path)
        # print(self.scoring_target_path)
        # print(self.scoring_candidate_path)
        # print(self.scoring_pose_path)
        # print(self.scoring_score_path)
        # print(self.al_train_path)
        # print(self.al_desc_path)
        # print(self.al_weight_path)

    #     self.config_dict = {
    #         "mode": self.mode,
    #         "train_path": base_pretraining + self.training_fname,
    #         "slice_data": self.slice_data,
    #         "val_path": base_pretraining + self.validation_fname,
    #         "n_head": self.n_head,
    #         "n_embed": self.n_embed,
    #         "att_bias": self.att_bias,
    #         "att_drop_rate": self.att_drop_rate,
    #         "do_flash": self.do_flash,
    #         "ff_mult": self.ff_mult,
    #         "doGELU": self.doGELU,
    #         "gpt_drop_rate": self.gpt_drop_rate,
    #         "n_layer": self.n_layer,
    #         "gpt_bias": self.gpt_bias,
    #         "weight_decay": self.weight_decay,
    #         "betas": self.betas,
    #         "rho": self.rho,
    #         "batch_size": self.batch_size,
    #         "num_workers": self.num_workers,
    #         "device": self.device,
    #         "lr_decay": self.lr_decay,
    #         "pretraining_checkpoint_path": base_model_parameters
    #         + self.pretraining_checkpoint_file,
    #         "al_checkpoint_path": base_model_parameters + self.al_checkpoint_file,
    #         "wandb_project": self.project_name,
    #         "wandb_runname": self.wandb_runname,
    #         "generation_context": self.context,
    #         "gen_batch_size": self.gen_batch_size,
    #         "inference_temp": self.temp,
    #         "gen_size": self.gen_size,
    #         "path_to_completions": base_gen_path + self.completions_file,
    #         "path_to_predicted": base_gen_path + self.predicted_file,
    #         "path_to_predicted_filtered": base_gen_path + self.predicted_filtered_file,
    #         "base_path": self.base_path,
    #         "smiles_key": self.smiles_key,
    #         "diffdock_scored_path_list": [
    #             f"{self.base_path}{base_scoring}{i}"
    #             for i in self.previously_scored_mols
    #         ],
    #         "al_trainsets_path_list": [
    #             f"{self.base_path}{base_active_learning}/{i}"
    #             for i in self.previous_al_train_sets
    #         ],
    #         "path_to_metrics": base_gen_path + self.metrics_file,
    #         "generation_path": base_gen_path,
    #         "path_to_gen_mol_descriptors": base_gen_path
    #         + self.gen_mol_descriptors_file,
    #         "path_to_pca": base_sampling_path + self.pca_file,
    #         "kmeans_save_path": base_sampling_path + self.kmeans_save_file,
    #         "clusters_save_path": base_sampling_path + self.clusters_save_file,
    #         "samples_save_path": base_sampling_path + self.samples_save_file,
    #         "diffdock_save_path": base_sampling_path + self.diffdock_save_file,
    #         "diffdock_results_path": base_diffdock_path + "poses/",
    #         "protein_path": self.base_path + self.protein_file,
    #         "diffdock_samples_path": base_diffdock_path + self.diffdock_samples_file,
    #         "path_to_scored": base_diffdock_path + self.scored_file,
    #         "path_to_good_mols": base_diffdock_path + self.good_mols_file,
    #         "AL_set_save_path": base_active_learning + self.AL_set_save_file,
    #         "AL_training_path": base_active_learning + self.AL_training_file,
    #     }

    # def update_config_dict(self):
    #     """Update the configuration dictionary based on the mode."""

    #     # Base paths for Pretraining and Active Learning
    #     base_pretraining_descriptors = (
    #         self.base_path + "1. Pretraining/dataset_descriptors/"
    #     )
    #     base_active_learning_descriptors = (
    #         self.base_path + "6. ActiveLearning/dataset_descriptors/"
    #     )


class ModelConfig:
    """GPT Configuration class."""

    def __init__(
        self,
        n_head: int = 8,
        n_embed: int = 256,
        att_drop_rate: float = 0.1,
        att_activation: str = "GELU",
        att_bias: bool = False,
        do_flash: bool = True,
        ff_mult: int = 4,
        n_layer: int = 8,
        gpt_drop_rate: float = 0.1,
        gpt_bias: bool = True,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        batch_size: int = 512,
        num_workers: int = 0,
        lr_decay: bool = True,
    ):

        # Model parameters
        # n_embed controls the dimensionality of the embedding vector for each character in the vocabulary
        self.n_embed = n_embed
        self.n_head = n_head
        # do_flash controls whether FlashAttention is used, otherwise regular einsum based multiplication
        self.do_flash = do_flash
        # att_drop_rate controls "p" of the Dropout layer in attention block
        self.att_drop_rate = att_drop_rate

        self.att_activation: Module
        if att_activation == "GELU":
            self.att_activation = torch.nn.GELU()
        elif att_activation == "ReLU":
            self.att_activation = torch.nn.ReLU()
        else:
            raise ValueError(
                f"only GeLU and ReLU are supported for the activation layer in attention block"
            )
        # att_bias controls whether bias is included in the MLP layer following attention block
        self.att_bias = att_bias
        # ff_mult*n_embed is the dimensionality of the hidden layer in the MLP following attention block
        self.ff_mult = ff_mult
        # gpt_drop_rate controls "p" of the Dropout layer
        self.gpt_drop_rate = gpt_drop_rate
        # n_layer is the number of Transformer Blocks
        self.n_layer = n_layer
        # gpt_bias controls whether bias is enabled in the final MLP before a token prediction is made
        self.gpt_bias = gpt_bias


        # SophiaG parameters
        self.weight_decay = weight_decay
        self.betas = betas
        self.rho = rho

        # Training Parameters
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

        self.train_params: Dict[str, Any] = {}

    def set_dataset_attributes(
        self,
        vocab_size: int,
        block_size: int,
        num_warmup_tokens: int,
        total_num_tokens: int,
        loss_ignore_index: int,):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_warmup_tokens = num_warmup_tokens
        self.total_num_tokens = total_num_tokens
        self.loss_ignore_index = loss_ignore_index


if __name__ == "__main__":
    base_path = os.getcwd() + "/PaperRuns/"
    config = Config(
        base_path=base_path,
        cycle_prefix="model0",
        al_iteration=0,
        cycle_suffix="ch0",
        training_fname="moses_test.csv.gz",
        validation_fname="moses_test.csv.gz",
    )
