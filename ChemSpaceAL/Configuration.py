import torch
from torch.nn import Module
import os
from typing import Union, Dict, Any, List, Tuple, Optional, Any, Set, Callable, cast
from ChemSpaceAL.InitializeWorkspace import FOLDER_STRUCTURE as fldr_struc
import textwrap
from rdkit.Chem import Descriptors
from rdkit import Chem

REGEX_PATTERN: str = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|@@|\?|>|!|~|\*|\$|\%[0-9]{2}|[0-9])"
RESTRICTED_FGS: List[str] = [
    "fr_azide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_thiocyan",
    "fr_prisulfonamd",
    "fr_C_S",
    "fr_azo",
    "fr_diazo",
    "fr_epoxide",
    "fr_ester",
    "fr_COO2",
    "fr_Imine",
    "fr_N_O",
    "fr_SH",
    "fr_aldehyde",
    "fr_dihydropyridine",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_ketone",
    "fr_thiophene",
    "fr_phenol",
]
AdmetDict = Dict[str, Dict[str, Union[Callable, int, float]]]
FUNC_ADMET: AdmetDict = {
    "MW": {"func": lambda mol: Descriptors.MolWt(mol), "lower": 100, "upper": 600},
    "nHA": {
        "func": lambda mol: Descriptors.NumHAcceptors(mol),
        "lower": 0,
        "upper": 12,
    },
    "nHD": {"func": lambda mol: Descriptors.NumHDonors(mol), "lower": 0, "upper": 7},
    "nRot": {
        "func": lambda mol: Descriptors.NumRotatableBonds(mol),
        "lower": 0,
        "upper": 11,
    },
    "nRing": {
        "func": lambda mol: Descriptors.RingCount(mol),
        "lower": 0,
        "upper": 6,
    },  # AdMET recommends [0,6], QED recommends >0
    "nHet": {
        "func": lambda mol: Descriptors.NumHeteroatoms(mol),
        "lower": 1,
        "upper": 15,
    },
    "fChar": {"func": lambda mol: Chem.GetFormalCharge(mol), "lower": -4, "upper": 4},
    "TPSA": {"func": lambda mol: Descriptors.TPSA(mol), "lower": 0, "upper": 140},
    "logP": {
        "func": lambda mol: Descriptors.MolLogP(mol),
        "lower": -0.4,
        "upper": 6.5, # TODO: specify which Abl binder has log.p 6.3 
    },  # AdMET Lab recommends [0,3], [-0.4, 5.6] from Ghose
}
# Dictionary containing scores for different protein-ligand interactions
INTERACTION_WEIGHTS: Dict[str, float] = {
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


class Config:
    """
    Configuration class for ChemSpace Active Learning.
    Holds various settings and parameters required for the ChemSpaceAL workflow.
    """

    target_to_description: Dict[str, str] = {
        "force_number_completions": "generations",
        "force_number_unique": "unique canonical smiles",
        "force_number_filtered": "unique canonical smiles that pass filters",
    }
    filter_options: Set[str] = {"ADMET", "ADMET+FGs", "FGs"}
    supported_descriptors: Set[str] = {"mix", "mqn", "mixmqn"}
    selection_modes: Set[str] = {"threshold", "percentile"}
    prolif_interactions: Set[str] = set(INTERACTION_WEIGHTS.keys())
    probability_modes: Set[str] = {"softmax", "softdiv", "linear", "uniform"}

    def __init__(
        self,
        base_path: str,
        cycle_prefix: str,
        al_iteration: int,
        cycle_suffix: str,
        training_fname: str,
        validation_fname: str,
        smiles_key: str = "smiles",
        slice_data: Union[bool, int] = False,
        previously_scored_mols: Optional[List] = None,
        previous_al_train_sets: Optional[List] = None,
        verbose: bool = True,
        regex_pattern: Optional[str] = None,
    ):
        self.base_path = base_path
        self.base_sep_count = base_path.count(os.sep)
        self.rel_path = lambda path: os.sep.join(
            path.split(os.sep)[self.base_sep_count :]
        )
        self.cycle_prefix = cycle_prefix
        self.al_iteration = al_iteration
        self.cycle_suffix = cycle_suffix

        # Pretraining configs
        self.training_fname = training_fname
        self.validation_fname = validation_fname
        # if slice_data is False, use all data, if an integer N, data will be sliced as data[:N]
        self.slice_data = slice_data
        self.smiles_key = smiles_key

        self.previously_scored_mols = previously_scored_mols
        self.previous_al_train_sets = previous_al_train_sets

        self.verbose = verbose
        self.regex_pattern = (
            regex_pattern if regex_pattern is not None else REGEX_PATTERN
        )

        # Configuration dictionary for convenience
        # self.config_dict: Dict[str, Any] = {}
        self.cycle_temp_params: Dict[str, Optional[str]] = {
            "completions_fname": None,
            "unique_smiles_fname": None,
            "generation_metrics_fname": None,
            "filtered_smiles_fname": None,
            "path_to_descriptors": None,
            "path_to_pca": None,
            "path_to_kmeans": None,
            "path_to_clusters": None,
            "path_to_sampled": None,
            "path_to_protein": None,
            "path_to_scored": None,
            "path_to_al_training_set": None,
        }
        self.prolif_weights: Optional[Dict[str, float]] = None
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
        al_path: Optional[str] = None,
        load_weight_path: Optional[str] = None,
        wandb_project_name: Optional[str] = None,
        wandb_runname: Optional[str] = None,
    ):
        if mode == "Pretraining":
            desc_path = (
                self.pretrain_desc_path + self.training_fname.split(".")[0] + ".yaml"
            )
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
            }
            self.model_config.generation_params["desc_path"] = desc_path

        elif mode == "Active Learning":
            if al_path is None:
                if (al_path := self.cycle_temp_params["path_to_al_training_set"]) is None:
                    raise ValueError(
                        "The name of the Active Learning Set isn't stored in current session, please provide through al_fname argument"
                    )
            desc_path = self.model_config.generation_params["desc_path"]
            assert (
                self.al_iteration >= 1
            ), "al_iteration cannot be less than 1 in Active Learning mode"
            if self.al_iteration == 1:
                load_path = self.pretrain_weight_path
            else:
                load_path = self.al_weight_path
            load_model_weights = (
                load_path + f"{self.cycle_prefix}_al{self.al_iteration-1}_{self.cycle_suffix}.pt"
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
            }
            self.model_config.generation_params["desc_path"] = desc_path

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
            row = "\n    {:<45} {:<80}"
            if (
                lpath := self.model_config.train_params["load_model_weight"]
            ) is not None:
                message += row.format(
                    "model weights will be loaded from:", self.rel_path(lpath)
                )
            if (
                spath := self.model_config.train_params["save_model_weight"]
            ) is not None:
                message += row.format(
                    "model weights will be saved to:", self.rel_path(spath)
                )
            message += row.format(
                "dataset descriptors will be loaded from:", self.rel_path(desc_path)
            )
            if wandb_project_name is None:
                message += f"\n  . note: wandb_project_name and wandb_runname were not provided, you can ignore this message if you don't plan to log runs to wandb"
            else:
                message += f"\n    runs will be stored in the {self.model_config.train_params['wandb_project_name']} wandb project"
                message += f"\n    under the name {self.model_config.train_params['wandb_runname']}"
            print(message)

    def set_generation_parameters(
        self,
        target_criterion: str,
        target_number: int = 10_000,
        batch_size: int = 64,
        temperature: float = 1.0,
        force_filters: Optional[str] = None,
        restricted_fgs: Optional[List[str]] = None,
        admet_criteria: Optional[AdmetDict] = None,
        load_model_weight: Optional[str] = None,
        dataset_desc_path: Optional[str] = None,
    ):
        assert (
            target_criterion in self.target_to_description
        ), f"Only {', '.join(self.target_to_description.keys())} are supported as target criterion"
        if force_filters is not None:
            assert (
                force_filters in self.filter_options
            ), f"Only {', '.join(self.filter_options)} are supported as force_filters"
        if target_criterion == "force_number_filtered":
            assert (
                force_filters is not None
            ), "force_filters must be specified for force_number_filtered target criterion"

        if restricted_fgs is None:
            restricted_fgs = RESTRICTED_FGS
        if admet_criteria is None:
            admet_criteria = FUNC_ADMET
        if load_model_weight is None:
            if (
                load_model_weight := self.model_config.train_params["save_model_weight"]
            ) is None:
                raise ValueError(
                    "No model weight path was provided, please provide through load_model_weight argument"
                )

        if dataset_desc_path is None:
            assert (
                "desc_path" in self.model_config.generation_params
            ), f"dataset_desc_path wasn't provided"
            dataset_desc_path = self.model_config.generation_params["desc_path"]
        else:
            self.model_config.generation_params["desc_path"] = dataset_desc_path
        self.cycle_temp_params["completions_fname"] = (
            self.generations_path
            + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}_completions.csv"
        )
        self.cycle_temp_params["unique_smiles_fname"] = (
            self.generations_path
            + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}_unique_smiles.csv"
        )
        self.cycle_temp_params["generation_metrics_fname"] = (
            self.generations_path
            + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}_metrics.txt"
        )
        if force_filters is not None:
            self.cycle_temp_params["filtered_smiles_fname"] = (
                self.generations_path
                + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}_filtered_smiles.csv"
            )
        self.model_config.generation_params.update(
            dict(
                context="!",
                target_criterion=target_criterion,
                target_number=target_number,
                batch_size=batch_size,
                temp=temperature,
                load_ckpt_path=load_model_weight,
                restricted_fgs=restricted_fgs,
                admet_criteria=admet_criteria,
                force_filters=force_filters,
            )
        )

        if self.verbose:
            message = f"""--- The following generation parameters were set:
    target number: {target_number} {self.target_to_description[target_criterion]}
    batch size: {batch_size} & temperature: {temperature}"""
            if force_filters is not None:
                message += (
                    f"\n    the following filters will be applied: {force_filters}"
                )
            row = "\n    {:<45} {:<80}"
            message += row.format(
                "model weights will be loaded from:", self.rel_path(load_model_weight)
            )
            message += row.format(
                "dataset descriptors will be loaded from:",
                self.rel_path(dataset_desc_path),
            )
            message += row.format(
                "generated completions will be saved to:",
                self.rel_path(cast(str, self.cycle_temp_params["completions_fname"])),
            )
            message += row.format(
                "unique canonic smiles will be saved to:",
                self.rel_path(cast(str, self.cycle_temp_params["unique_smiles_fname"])),
            )
            message += row.format(
                "generation metrics will be saved to:",
                self.rel_path(
                    cast(str, self.cycle_temp_params["generation_metrics_fname"])
                ),
            )
            if target_criterion == "force_number_filtered":
                message += row.format(
                    "filtered molecules will be saved to:",
                    self.rel_path(
                        cast(str, self.cycle_temp_params["filtered_smiles_fname"])
                    ),
                )
            if force_filters is not None and "ADMET" in force_filters:
                message += f"\n    The following ADMET filters will be enforced:"
                for descriptor, paramdic in admet_criteria.items():
                    message += f"\n    |    {descriptor} in range [{paramdic['lower']}, {paramdic['upper']}]"
            if force_filters is not None and "FGs" in force_filters:
                message += f"\n    The following functional groups will be restricted:"
                joined_str = ", ".join(restricted_fgs)
                message += f"\n    |    {textwrap.fill(joined_str, 80, subsequent_indent='    |    ')}"
            print(message)

    def set_previous_arrays(
        self,
        previously_scored_mols: Optional[List[str]] = None,
        previous_al_train_sets: Optional[List[str]] = None,
    ):
        if previously_scored_mols is None:
            previously_scored_mols = [
                self.scoring_score_path
                + f"{self.cycle_prefix}_al{i}_{self.cycle_suffix}.csv"
                for i in range(0, self.al_iteration)
            ]
        if previous_al_train_sets is None:
            previous_al_train_sets = [
                self.al_train_path
                + f"{self.cycle_prefix}_al{i}_{self.cycle_suffix}.csv"
                for i in range(0, self.al_iteration)
            ]
        self.previously_scored_mols = previously_scored_mols
        self.previous_al_train_sets = previous_al_train_sets
        if self.verbose:
            message = f"""--- The following previously scored molecules were set:"""
            for path in self.previously_scored_mols:
                rel_path = os.sep.join(path.split(os.sep)[self.base_sep_count :])
                message += f"\n     {rel_path}"
            message += f"""\n--- The following previously constructed Active Learning sets were set:"""
            for path in self.previous_al_train_sets:
                rel_path = os.sep.join(path.split(os.sep)[self.base_sep_count :])
                message += f"\n     {rel_path}"
            print(message)

    def set_sampling_parameters(
        self,
        n_clusters: int,
        samples_per_cluster: int,
        pca_fname: str,
        descriptors_mode: Optional[str] = None,
    ):
        if descriptors_mode is None:
            descriptors_mode = "mix"
        else:
            assert (
                descriptors_mode in self.supported_descriptors
            ), f"Only {', '.join(self.supported_descriptors)} are supported as descriptors_mode"
        n_samples = n_clusters * samples_per_cluster
        self.sampling_parameters = {
            "n_clusters": n_clusters,
            "samples_per_cluster": samples_per_cluster,
            "n_samples": n_samples,
            "descriptors_mode": descriptors_mode,
        }
        self.cycle_temp_params["path_to_descriptors"] = (
            self.sampling_desc_path
            + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}.pkl"
        )
        self.cycle_temp_params["path_to_pca"] = self.sampling_pca_path + pca_fname
        self.cycle_temp_params["path_to_kmeans"] = (
            self.sampling_kmeans_path
            + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}_k{n_clusters}.pkl"
        )
        self.cycle_temp_params["path_to_clusters"] = (
            self.sampling_clusters_path
            + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}_cluster_to_mols.pkl"
        )
        self.cycle_temp_params["path_to_sampled"] = (
            self.scoring_candidate_path
            + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}_sampled{n_samples}.csv"
        )
        if self.verbose:
            message = f"""--- The following sampling parameters were set:
    number of clusters: {n_clusters}
    samples per cluster: {samples_per_cluster}
    descriptors mode: {descriptors_mode}"""
            row = "\n    {:<50} {:<80}"
            message += row.format(
                "descriptors will be saved to:",
                self.rel_path(cast(str, self.cycle_temp_params["path_to_descriptors"])),
            )
            message += row.format(
                "PCA will be loaded from:",
                self.rel_path(cast(str, self.cycle_temp_params["path_to_pca"])),
            )
            message += row.format(
                "KMeans Objects will be saved to:",
                self.rel_path(cast(str, self.cycle_temp_params["path_to_kmeans"])),
            )
            message += row.format(
                "cluster to molecules mapping will be saved to:",
                self.rel_path(cast(str, self.cycle_temp_params["path_to_clusters"])),
            )
            message += row.format(
                "sampled molecules will be saved to:",
                self.rel_path(cast(str, self.cycle_temp_params["path_to_sampled"])),
            )
            print(message)

    def set_scoring_parameters(
        self, protein_path: str, interaction_weights: Optional[Dict[str, float]] = None
    ):
        prolif_weights = INTERACTION_WEIGHTS
        if interaction_weights is not None:
            for interaction, weight in interaction_weights.items():
                assert (
                    interaction in self.prolif_interactions
                ), f"{interaction=} is not counted by prolif, only {', '.join(self.prolif_interactions)} are supported"
                prolif_weights[interaction] = weight
        self.prolif_weights = prolif_weights
        self.cycle_temp_params["path_to_protein"] = (
            self.scoring_target_path + protein_path
        )
        self.cycle_temp_params["path_to_poses"] = (
            self.scoring_pose_path
            + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}" + os.sep
        )
        self.cycle_temp_params["path_to_scored"] = (
            self.scoring_score_path
            + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}.csv"
        )
        if self.verbose:
            row = "\n    {:<45} {:<80}"
            message = "--- The following scoring parameters were set:"
            message += row.format(
                "Reminder that docking poses will be written to",
                self.rel_path(self.scoring_pose_path),
            )
            message += row.format(
                "protein will be loaded from",
                self.rel_path(self.cycle_temp_params["path_to_protein"]),
            )
            message += row.format(
                "poses will be saved to",
                self.rel_path(self.cycle_temp_params["path_to_poses"]),
            )
            message += row.format(
                "and scored molecules will be saved to",
                self.rel_path(self.cycle_temp_params["path_to_scored"]),
            )
            message += f"\n    The following prolif interaction weights will be used:"
            joined_str = ", ".join([f"{i}: {w}" for i, w in prolif_weights.items()])
            message += f"\n    |    {textwrap.fill(joined_str, 80, subsequent_indent='    |    ')}"
            print(message)

    def set_active_learning_parameters(
        self,
        selection_mode: str,
        training_size: int,
        n_replicate: bool = True,
        probability_mode: str = "softmax",
        cluster_score_mode: str = "mean",
        softdiv_factor: Optional[float] = None,
        threshold: Optional[float] = None,
        percentile: Optional[float] = None,
    ):
        assert (
            selection_mode in self.selection_modes
        ), f"Only {', '.join(self.selection_modes)} are supported as selection modes"
        assert (
            probability_mode in self.probability_modes
        ), f"Only {', '.join(self.probability_modes)} are supported as probability modes"
        assert (
            training_size % 2 == 0
        ), f"Please provide an even value for the `training_size` so that it can be evenly split between top ligands and sampled_mols"
        self.alconstruction_parameters: Dict[str, Union[str, float, bool]] = {
            "selection_mode": selection_mode,
            "probability_mode": probability_mode,
            "training_size": training_size,
            "n_replicate": n_replicate,
            "cluster_score_mode": cluster_score_mode,
        }
        if probability_mode == "softdiv":
            assert isinstance(
                softdiv_factor, float
            ), "softdiv_factor must be provided when `probability_mode` is set to softdiv"
            self.alconstruction_parameters["softdiv_factor"] = softdiv_factor
        match selection_mode:
            case "threshold":
                assert isinstance(
                    threshold, (float, int)
                ), "you have to provide `threshold` value with selection mode `threshold`"
                self.alconstruction_parameters["threshold"] = threshold
            case "percentile":
                assert isinstance(
                    percentile, (float, int)
                ), "you have to provide `percentile` value with selection mode `percentile`"
                self.alconstruction_parameters["percentile"] = percentile

        self.cycle_temp_params["path_to_al_training_set"] = (
            self.al_train_path
            + f"{self.cycle_prefix}_al{self.al_iteration}_{self.cycle_suffix}.csv"
        )
        if self.verbose:
            message = (
                "--- The following AL training set construction parameters were set:"
            )
            message += f"\n    the training set will be constructed to have {training_size} molecules"
            message += f"\n    of which {training_size//2} will be selected from the top scoring molecules defined by the following parameters:"
            match selection_mode:
                case "threshold":
                    message += f"\n        molecules with score above {threshold} will be selected"
                case "percentile":
                    message += f"\n        molecules with score above the {percentile}th percentile will be selected"
            message += f"\n    the remaining {training_size//2} molecules will be selected from high-scoring clusters according to the following parameters:"
            message += f"\n        the following probability mode will be used: {probability_mode}"
            if probability_mode == "softdiv":
                message += f"\n        the following softdiv factor will be used: {softdiv_factor}"
            row = "\n    {:<45} {:<80}"
            message += row.format(
                "the training set will be saved to",
                self.rel_path(self.cycle_temp_params["path_to_al_training_set"]),
            )
            print(message)


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
        self.generation_params: Dict[str, Any] = {}

    def set_dataset_attributes(
        self,
        vocab_size: int,
        block_size: int,
        num_warmup_tokens: Optional[int] = None,
        total_num_tokens: Optional[int] = None,
        loss_ignore_index: Optional[int] = None,
    ):
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
