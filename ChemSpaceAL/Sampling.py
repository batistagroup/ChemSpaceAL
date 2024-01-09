import pandas as pd
from tqdm import tqdm
import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import pickle
from sklearn.cluster import KMeans
import numpy as np

from ChemSpaceAL.Configuration import Config
from typing import Optional, Dict, List, Any, Union, Iterable, Sized


def calculate_descriptors(config: Config, smiles_fname: Optional[str] = None):
    """Extract descriptors for molecules from SMILES data.

    Args:
        config (dict): Configuration dictionary containing path keys.

    Returns:
        pd.DataFrame: DataFrame containing descriptors for each molecule.
    """
    if smiles_fname is None:
        gen_params = config.model_config.generation_params
        match gen_params["target_criterion"]:
            case "force_number_unique":
                assert (
                    config.cycle_temp_params["unique_smiles_fname"] is not None
                ), "please call .set_generation_parameters() or provide smiles_fname"
                smiles_fname = config.cycle_temp_params["unique_smiles_fname"]
            case "force_number_filtered":
                assert (
                    config.cycle_temp_params["filtered_smiles_fname"] is not None
                ), "please call .set_generation_parameters() or provide smiles_fname"
                smiles_fname = config.cycle_temp_params["filtered_smiles_fname"]
    match config.sampling_parameters["descriptors_mode"]:
        case "mix":
            func = rdkit.Chem.Descriptors.CalcMolDescriptors
        case "mqn":
            func = rdMolDescriptors.MQNs_

    smiles_set = set(pd.read_csv(smiles_fname)["smiles"])
    keySet = None
    smileToData: Dict[str, List[Any]] = {}
    pbar = tqdm(smiles_set, total=len(smiles_set))
    for smile in pbar:
        mol = rdkit.Chem.MolFromSmiles(smile)
        if mol is None:
            continue

        descriptors = func(mol)
        if keySet is None:
            keySet = set(descriptors.keys())
        for key in keySet:
            smileToData.setdefault(key, []).append(descriptors[key])
        smileToData.setdefault("smiles", []).append(smile)

    gpt_df = pd.DataFrame(smileToData)
    gpt_df.to_pickle(config.cycle_temp_params["path_to_descriptors"])


def project_into_pca_space(config: Config) -> np.ndarray:
    """Project molecules into PCA space.

    Args:
        config (dict): Configuration dictionary containing path keys.

    Returns:
        np.array: Array of PCA transformed molecules.
    """
    assert (
        config.cycle_temp_params["path_to_pca"] is not None
    ), "please provide path_to_pca through .set_sampling_parameters"
    scaler, pca = pickle.load(open(config.cycle_temp_params["path_to_pca"], "rb"))
    gptMols = pd.read_pickle(config.cycle_temp_params["path_to_descriptors"])
    return pca.transform(scaler.transform(gptMols[scaler.get_feature_names_out()]))


def _cluster_mols_by_mixed(
    mols: np.ndarray, n_clusters: int, n_iter: int, mixed_objective_loss_quantile: float
) -> KMeans:
    """Cluster molecules using experimental mixed method.

    Args:
        mols (np.array): Array of molecules.
        n_clusters (int): Number of clusters.
        n_iter (int): Number of iterations.
        mixed_objective_loss_quantile (float): Quantile for loss.

    Returns:
        object: Fitted KMeans clustering object.
    """
    inertias = []
    variances = []
    km_objs = []
    for _ in tqdm(range(n_iter), total=n_iter):
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", init="k-means++").fit(
            mols
        )
        inertias.append(kmeans.inertia_)
        counts = np.unique(kmeans.labels_, return_counts=True)[1]
        variances.append(np.var(counts))
        km_objs.append(kmeans)

    loss_var_kmeans_triples = sorted(
        zip(inertias, variances, km_objs), key=lambda x: x[0]
    )
    lowest_n = loss_var_kmeans_triples[
        : int(len(loss_var_kmeans_triples) * mixed_objective_loss_quantile)
    ]
    sorted_by_variance = sorted(lowest_n, key=lambda x: x[1])
    return sorted_by_variance[0][2]


def _cluster_mols_by_loss(mols: np.ndarray, n_clusters: int, n_iter: int) -> KMeans:
    min_loss, best_kmeans = float("inf"), None
    for _ in tqdm(range(n_iter), total=n_iter):
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", init="k-means++").fit(
            mols
        )
        if kmeans.inertia_ < min_loss:
            min_loss = kmeans.inertia_
            best_kmeans = kmeans
    return best_kmeans


def _cluster_mols_by_variance(mols: np.ndarray, n_clusters: int, n_iter: int) -> KMeans:
    max_variance, best_kmeans = float("-inf"), None
    for _ in tqdm(range(n_iter), total=n_iter):
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", init="k-means++").fit(
            mols
        )
        counts = np.unique(kmeans.labels_, return_counts=True)[1]
        if (variance := np.var(counts)) > max_variance:
            max_variance = variance  # type: ignore
            best_kmeans = kmeans
    return best_kmeans


def _cluster_mols(
    mols: np.ndarray,
    n_clusters: int,
    save_path: str,
    n_iter: int = 1,
    objective: str = "loss",
    mixed_objective_loss_quantile=0.1,
) -> KMeans:
    if n_iter == 1:
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", init="k-means++").fit(
            mols
        )
    elif objective == "loss":
        kmeans = _cluster_mols_by_loss(mols, n_clusters, n_iter)
    elif objective == "variance":
        kmeans = _cluster_mols_by_variance(mols, n_clusters, n_iter)
    elif objective == "mixed":
        kmeans = _cluster_mols_by_mixed(
            mols, n_clusters, n_iter, mixed_objective_loss_quantile
        )
    else:
        raise ValueError(f"Unknown objective {objective}")

    pickle.dump(kmeans, open(save_path, "wb"))
    return kmeans


def cluster_and_sample(
    mols: np.ndarray,
    config: Config,
    n_iter: int = 100,
    objective: str = "mixed",
    load_kmeans=False,
) -> Dict[int, np.ndarray]:
    """Cluster the molecules and then sample from each cluster.

    Args:
        mols (np.array): Array of molecules.
        config (dict): Configuration dictionary containing path keys and other settings.
        n_clusters (int): Number of clusters.
        n_samples (int): Number of samples per cluster.
        ensure_correctness (bool, optional): Flag to check correctness of PCA transformation. Defaults to False.
        path_to_pca (str, optional): Path to the PCA model. Required if ensure_correctness is True.
        load_kmeans (bool, optional): Flag to load an existing KMeans model. Defaults to False.

    Returns:
        dict: Dictionary mapping each cluster to its sampled molecules.
    """

    # Assert checks to make sure input parameters are valid
    not_init_error = ".set_sampling_parameters() was not called"
    assert isinstance(
        (samples_per_cluster := config.sampling_parameters["samples_per_cluster"]), int
    ), not_init_error
    assert isinstance(
        (n_samples := config.sampling_parameters["n_samples"]), int
    ), not_init_error
    assert isinstance(
        (n_clusters := config.sampling_parameters["n_clusters"]), int
    ), not_init_error
    path_to_pca = config.cycle_temp_params["path_to_pca"]
    assert n_samples <= len(
        mols
    ), f"{n_samples} requested but only {len(mols)} molecules provided"

    assert isinstance(
        (kmeans_path := config.cycle_temp_params["path_to_kmeans"]), str
    ), not_init_error
    if load_kmeans:
        kmeans = pickle.load(open(kmeans_path, "rb"))
    else:
        kmeans = _cluster_mols(
            mols=mols,
            n_iter=n_iter,
            n_clusters=n_clusters,
            save_path=kmeans_path,
            objective=objective,
            mixed_objective_loss_quantile=0.05,
        )

    # Load SMILES strings
    assert isinstance(
        (desc_path := config.cycle_temp_params["path_to_descriptors"]), str
    ), not_init_error
    mols_smiles = pd.read_pickle(desc_path)["smiles"]
    assert len(kmeans.labels_) == len(
        mols_smiles
    ), "Number of labels differs from number of molecules"

    # Load previously scored molecules
    scored_smiles = set()
    assert isinstance(
        config.previously_scored_mols, list
    ), ".set_previous_arrays() wasn't called"
    for scored_file in config.previously_scored_mols:
        for smile in pd.read_csv(scored_file)["smiles"].values:
            scored_smiles.add(smile)
    print(f"Loaded {len(scored_smiles)} scored molecules")

    # Collect non-repeated molecules from clusters
    repeated = 0
    cluster_to_mols: Dict[int, List[str]] = {}
    for mol, label, smile in zip(mols, kmeans.labels_, mols_smiles):
        if smile in scored_smiles:
            repeated += 1
            continue
        cluster_to_mols.setdefault(label, []).append(smile)
    print(
        f"{repeated} generated molecules were already scored. Excluding them from sampling."
    )
    # What happens below is sampling from each cluster. All the extra code is to ensure that the number of samples requested from each cluster
    # doesn't exceed the total number of available molecules. This is done by calculating the average number of molecules per cluster and then
    # calculating the number of extra molecules that need to be sampled from each cluster. The extra molecules are then distributed among the
    # clusters uniformly. If the number of extra molecules is greater than the number of molecules in a cluster, all
    # molecules from that cluster are sampled.
    avg_len = np.mean([len(v) for v in cluster_to_mols.values()])
    cluster_to_samples: Dict[int, np.ndarray[str]] = {}
    extra_mols = (n_clusters - len(cluster_to_mols)) * samples_per_cluster
    left_to_sample = n_samples
    cluster_to_len = {cluster: len(mols) for cluster, mols in cluster_to_mols.items()}

    for i, (cluster, _) in enumerate(
        sorted(cluster_to_len.items(), key=lambda x: x[1], reverse=False)
    ):
        smiles = np.array(cluster_to_mols[cluster])
        if extra_mols > 0:
            cur_extra = int(
                1 + extra_mols / (len(cluster_to_mols) - i) * len(smiles) / avg_len
            )
            cur_samples = samples_per_cluster + cur_extra
            extra_mols -= cur_extra
        else:
            cur_samples = samples_per_cluster
        # print(f"{cluster=}, {left_to_sample=}, {cur_samples=}")
        if cur_samples > left_to_sample:
            cur_samples = left_to_sample
        if len(smiles) > cur_samples:
            cluster_to_samples[cluster] = np.random.choice(
                smiles, cur_samples, replace=False
            )
            left_to_sample -= cur_samples
        else:
            cluster_to_samples[cluster] = smiles
            left_to_sample -= len(smiles)
            extra_mols += cur_samples - len(smiles)

    assert (
        n_sampled := sum(len(vals) for vals in cluster_to_samples.values())
    ) == n_samples, f"Sampled {n_sampled} but were requested {n_samples}"

    # Save clusters and samples
    assert isinstance(
        (clusters_path := config.cycle_temp_params["path_to_clusters"]), str
    ), not_init_error
    pickle.dump(cluster_to_mols, open(clusters_path, "wb"))

    assert isinstance(
        (samples_path := config.cycle_temp_params["path_to_sampled"]), str
    ), not_init_error
    # Prepare the data for saving in csv format
    keyToData: Dict[str, List[Union[str, int]]] = {}
    for cluster, mols in cluster_to_samples.items():
        for mol in mols:
            keyToData.setdefault("smiles", []).append(mol)
            keyToData.setdefault("cluster_id", []).append(cluster)
    pd.DataFrame(keyToData).to_csv(samples_path)

    return cluster_to_samples
