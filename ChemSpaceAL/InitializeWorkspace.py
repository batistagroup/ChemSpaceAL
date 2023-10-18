import os
from typing import Optional

FOLDER_STRUCTURE = {
    "1_Pretraining": {
        "dataset_folder": "datasets",
        "desc_folder": "datasets_descriptors",
        "weight_folder": "model_weights",
    },
    "2_Generation": None,
    "3_Sampling": {
        "desc_folder": "generations_descriptors",
        "pca_folder": "pca_weights",
        "kmeans_folder": "kmeans_objects",
        "clustering_folder": "clusterings",
    },
    "4_Scoring": {
        "target_folder": "binding_targets",
        "candidate_folder": "sampled_mols",
        "pose_folder": "binding_poses",
        "score_folder": "scored_dataframes",
    },
    "5_ActiveLearning": {
        "train_folder": "training_sets",
        "desc_folder": "trainingset_descriptors",
        "weight_folder": "model_weights",
    },
}


def create_folders(base_path: Optional[str] = None, folder_structure=None):
    if folder_structure is None:
        folder_structure = FOLDER_STRUCTURE
    if base_path is None:
        print(
            f"! WARNING: base_path is not provided, do you want to use the following path?"
        )
        response = input(f'... "{os.getcwd()}/"? Type Y/N: ')
        if response != "Y":
            raise ValueError(
                "base_path is not provided, default path rejected, aborting."
            )
        base_path = os.getcwd() + "/"
    print(f"    will create folders at {base_path=}")

    for folder, subfolders in folder_structure.items():
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        if subfolders is not None:
            for subfolder in subfolders.values():
                subfolder_path = os.path.join(folder_path, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)


if __name__ == "__main__":
    paper_runs = os.getcwd() + "/PaperRuns/"
    create_folders(base_path=paper_runs)
