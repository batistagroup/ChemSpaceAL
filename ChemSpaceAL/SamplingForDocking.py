import pandas as pd
from tqdm import tqdm
import rdkit
from rdkit.Chem import Descriptors
import pickle
from sklearn.cluster import KMeans
import numpy as np

from .Configuration import *


def get_descriptors(config):
    """Extract descriptors for molecules from SMILES data.

    Args:
        config (dict): Configuration dictionary containing path keys.

    Returns:
        pd.DataFrame: DataFrame containing descriptors for each molecule.
    """
    # Load molecules data
    gpt_mols = pd.read_csv(config["path_to_predicted"])
    smiles_set = set(gpt_mols["smiles"].to_list())

    keySet = None
    keyToData = {}
    pbar = tqdm(smiles_set, total=len(smiles_set))
    for smile in pbar:
        mol = rdkit.Chem.MolFromSmiles(smile)
        if not mol:
            continue
        # Calculate descriptors
        mol_data = rdkit.Chem.Descriptors.CalcMolDescriptors(mol)

        if keySet is None:
            keySet = set(mol_data.keys())

        for key in keySet:
            keyToData.setdefault(key, []).append(mol_data[key])
        keyToData.setdefault("smiles", []).append(smile)

    gpt_df = pd.DataFrame(keyToData)
    gpt_df.to_pickle(config["path_to_gen_mol_descriptors"])

    return gpt_df


def project_into_pca_space(config):
    """Project molecules into PCA space.

    Args:
        config (dict): Configuration dictionary containing path keys.

    Returns:
        np.array: Array of PCA transformed molecules.
    """
    scaler, pca = pickle.load(open(config["path_to_pca"], "rb"))
    gptMols = pd.read_pickle(config["path_to_gen_mol_descriptors"])
    return pca.transform(scaler.transform(gptMols[scaler.get_feature_names_out()]))


def _cluster_mols_experimental_mixed(
    mols, n_clusters, n_iter, mixed_objective_loss_quantile
):
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


def _cluster_mols_experimental(
    mols,
    n_clusters,
    save_path,
    n_iter=1,
    objective="loss",
    mixed_objective_loss_quantile=0.1,
):
    """Cluster molecules using experimental method and save the KMeans model.

    Args:
        mols (np.array): Array of molecules.
        n_clusters (int): Number of clusters.
        save_path (str): Path to save KMeans model.
        n_iter (int, optional): Number of iterations. Defaults to 1.
        objective (str, optional): Objective metric. Defaults to 'loss'.
        mixed_objective_loss_quantile (float, optional): Quantile for loss. Defaults to 0.1.

    Returns:
        object: Fitted KMeans clustering object.
    """
    kmeans = _cluster_mols_experimental_mixed(
        mols, n_clusters, n_iter, mixed_objective_loss_quantile
    )
    pickle.dump(kmeans, open(save_path, "wb"))
    return kmeans


def cluster_and_sample(
    mols,
    config,
    n_clusters,
    n_samples,
    ensure_correctness=False,
    path_to_pca=None,
    load_kmeans=False,
):
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
    assert n_clusters * n_samples <= len(
        mols
    ), f"{n_clusters=} * {n_samples=} = {n_clusters*n_samples} requested but only {len(mols)} molecules provided"

    if ensure_correctness:
        assert (
            path_to_pca is not None
        ), "path_to_pca must be provided to ensure correctness"
        scaler, pca = pickle.load(open(path_to_pca, "rb"))

    if load_kmeans:
        kmeans = pickle.load(open(config["kmeans_save_path"], "rb"))
    else:
        kmeans = _cluster_mols_experimental(
            mols=mols,
            n_iter=100,
            n_clusters=n_clusters,
            save_path=config["kmeans_save_path"],
            objective="mixed",
            mixed_objective_loss_quantile=0.05,
        )

    # Load SMILES strings
    mols_smiles = pd.read_pickle(config["path_to_gen_mol_descriptors"])["smiles"]
    assert len(kmeans.labels_) == len(
        mols_smiles
    ), "Number of labels differs from number of molecules"

    # Load previously scored molecules
    scored_smiles = set()
    for scored_file in config["diffdock_scored_path_list"]:
        for smile in pd.read_csv(scored_file)["smiles"].values:
            scored_smiles.add(smile)
    print(f"Loaded {len(scored_smiles)} scored molecules")

    # Collect non-repeated molecules from clusters
    repeated = 0
    cluster_to_mols = {}
    for mol, label, smile in zip(mols, kmeans.labels_, mols_smiles):
        if smile in scored_smiles:
            repeated += 1
            continue
        cluster_to_mols.setdefault(label, []).append(smile)
        if ensure_correctness:
            smile_features = pca.transform(
                scaler.transform(
                    pd.DataFrame(
                        {
                            k: [v]
                            for k, v in rdkit.Chem.Descriptors.CalcMolDescriptors(
                                rdkit.Chem.MolFromSmiles(smile)
                            ).items()
                        }
                    )[scaler.get_feature_names_out()]
                )
            )
            assert np.allclose(
                smile_features[0], mol
            ), "Features calculated from a smile string differ from features in the array"

    # Distribute extra molecules across clusters proportionally
    avg_len = np.mean([len(v) for v in cluster_to_mols.values()])
    cluster_to_samples = {}
    extra_mols = (100 - len(cluster_to_mols)) * 10
    left_to_sample = n_clusters * n_samples
    cluster_to_len = {cluster: len(mols) for cluster, mols in cluster_to_mols.items()}

    for i, (cluster, _) in enumerate(
        sorted(cluster_to_len.items(), key=lambda x: x[1], reverse=False)
    ):
        smiles = cluster_to_mols[cluster]
        if extra_mols > 0:
            cur_extra = int(
                1 + extra_mols / (len(cluster_to_mols) - i) * len(smiles) / avg_len
            )
            cur_samples = n_samples + cur_extra
            extra_mols -= cur_extra
        else:
            cur_samples = n_samples
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
    ) == n_clusters * n_samples, (
        f"Sampled {n_sampled} but were requested {n_clusters*n_samples}"
    )

    # Save clusters and samples
    pickle.dump(cluster_to_mols, open(config["clusters_save_path"], "wb"))
    pickle.dump(cluster_to_samples, open(config["samples_save_path"], "wb"))

    # Prepare the data for saving in csv format
    keyToData = {}
    for cluster, mols in cluster_to_samples.items():
        for mol in mols:
            keyToData.setdefault("smiles", []).append(mol)
            keyToData.setdefault("cluster_id", []).append(cluster)
    pd.DataFrame(keyToData).to_csv(config["diffdock_save_path"])

    return cluster_to_samples
