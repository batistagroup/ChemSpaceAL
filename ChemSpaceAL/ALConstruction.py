import pandas as pd
import numpy as np
import pickle

from ChemSpaceAL.Configuration import Config
from typing import Union, Dict, List, Callable, Optional, cast

Number = Union[float, int]


def find_top_ligands(config: Config) -> pd.DataFrame:
    sampled_mols = pd.read_csv(config.cycle_temp_params["path_to_scored"])
    match config.alconstruction_parameters["selection_mode"]:
        case "percentile":
            assert isinstance(
                (percentile := config.alconstruction_parameters["percentile"]),
                (float, int),
            )
            threshold = np.percentile(
                a=list(sampled_mols["score"].values()), q=percentile
            )
        case "threshold":
            assert isinstance(
                (threshold := config.alconstruction_parameters["threshold"]),
                (float, int),
            )
        case _:
            raise KeyError(
                "Only 'percentile' and 'threshold' selection modes are supported"
            )
    top_ligands = sampled_mols[sampled_mols["score"] >= threshold]
    return top_ligands


def compute_cluster_scores(config: Config) -> Dict[int, Number]:
    """
    Compute the average scores for each cluster.

    Parameters:
    - config (dict): Configuration dictionary containing paths and other settings.

    Returns:
    - dict: A dictionary with cluster IDs as keys and their average scores as values.
    """
    scored_mols = pd.read_csv(config.cycle_temp_params["path_to_scored"])
    cluster_to_scores: Dict[int, List[Union[float, int]]] = {}
    for _, row in scored_mols.iterrows():
        cluster_to_scores.setdefault(row["cluster_id"], []).append(row["score"])
    match config.alconstruction_parameters["cluster_score_mode"]:
        case "mean":
            aggregator: Callable = np.mean
        case "median":
            aggregator = np.median
    return {
        cluster_id: aggregator(scores)
        for cluster_id, scores in cluster_to_scores.items()
    }


def _preprocess_scores_linearly(
    scores: Dict[int, Number], do_negation: bool = False
) -> Dict[int, Number]:
    sign = -1 if do_negation else 1
    negated = {k: sign * v for k, v in scores.items()}
    total = sum(negated.values())
    normalized = {k: v / total for k, v in negated.items()}
    return normalized

def _preprocess_scores_softmax(
    scores: Dict[int, Number],
    do_negation: bool = False,
    divide: bool = True,
    divide_factor: Optional[float] = None,
) -> Dict[int, Number]:
    sign = -1 if do_negation else 1
    negated = {k: sign * v for k, v in scores.items()}
    max_value = max(negated.values())
    if divide:
        assert (
            divide_factor is not None
        ), "You have to specify a value p in (0, 1). Softmax is computed as e^[x/(p*max)]"
        exponentiate = {
            k: np.exp(v / (divide_factor * max_value)) for k, v in negated.items()
        }
    else:
        exponentiate = {k: np.exp(v - max_value) for k, v in negated.items()}
    total = sum(exponentiate.values())
    softmax = {k: v / total for k, v in exponentiate.items()}
    return softmax

def balance_cluster_to_n(
    cluster_to_n: Dict[int, int], cluster_to_len: Dict[int, int]
) -> Dict[int, int]:
    """
    Balances the target number of samples for each cluster to ensure it doesn't exceed the actual size of the cluster.

    The function first calculates the surplus (i.e., the excess of the target number over the actual size) for each cluster.
    Then, it distributes the total surplus proportionally among the clusters that have a deficit (i.e., the target number is less than the actual size).
    If after this distribution, there's still a deficit (i.e., the sum of target numbers is less than the sum of actual sizes), the function
    increases the target number of the largest clusters one by one until the sum of target numbers equals to the sum of actual sizes.

    Parameters
    ----------
    cluster_to_n : dict
        A dictionary mapping cluster identifiers to their target number of samples.

    cluster_to_len : dict
        A dictionary mapping cluster identifiers to the actual size of each cluster.

    Returns
    -------
    balanced : dict
        A dictionary mapping cluster identifiers to their balanced target number of samples.

    Raises
    ------
    AssertionError
        If the sum of target numbers before and after balancing don't match.

    """

    surplus = {
        key: cluster_to_n[key] - cluster_to_len[key]
        for key in cluster_to_n
        if cluster_to_n[key] > cluster_to_len[key]
    }
    balanced = {k: v for k, v in cluster_to_n.items()}
    n_to_cluster = {v: k for k, v in cluster_to_n.items()}

    for key in surplus:
        balanced[key] = cluster_to_len[key]

    total_surplus = sum(surplus.values())
    initial_n_sum = sum(n for key, n in cluster_to_n.items() if key not in surplus)

    for key in balanced:
        if key in surplus:
            continue
        surplus_to_add = total_surplus * cluster_to_n[key] / initial_n_sum
        new_n = int(cluster_to_n[key] + surplus_to_add)
        balanced[key] = min(new_n, cluster_to_len[key])

    deficit = sum(cluster_to_n.values()) - sum(balanced.values())
    while deficit > 0:
        for initial_n in sorted(n_to_cluster, reverse=True):
            if deficit == 0:
                break
            if (cluster := n_to_cluster[initial_n]) in surplus:
                continue
            if balanced[cluster] < cluster_to_len[cluster]:
                balanced[cluster] += 1
                deficit -= 1

    assert sum(cluster_to_n.values()) == sum(
        balanced.values()
    ), f"Before balancing had {sum(cluster_to_n.values())}, post balancing = {sum(balanced.values())}"
    return balanced


def sample_molecules_from_clusters(config: Config, n_samples: int):
    match config.alconstruction_parameters["probability_mode"]:
        case "uniform":
            probability_function = lambda x: x
        case "linear":
            probability_function = lambda x: _preprocess_scores_linearly(x)
        case "softmax":
            probability_function = lambda x: _preprocess_scores_softmax(
                x,
                divide=False,
            )
        case "softdiv":
            assert isinstance(
                (divide_factor := config.alconstruction_parameters["softdiv_factor"]),
                float,
            ), f".set_active_learning_parameters() wasn't called"
            probability_function = lambda x: _preprocess_scores_softmax(
                x,
                divide=True,
                divide_factor=cast(float, divide_factor),
            )
    assert isinstance(
        (cluster_path := config.cycle_temp_params["path_to_clusters"]), str
    ), f".set_sampling_parameters() wasn't called"
    cluster_to_mols = pickle.load(open(cluster_path, "rb"))
    docked_mols = set(
        pd.read_csv(config.cycle_temp_params["path_to_sampled"])["smiles"]
    )
    cluster_to_new_mols = {
        k: [smile for smile in set(v) if smile not in docked_mols]
        for k, v in cluster_to_mols.items()
    }

    cluster_to_scores = compute_cluster_scores(config=config)
    probabilities = probability_function(cluster_to_scores)
    cluster_to_n = {k: int(v * n_samples) for k, v in probabilities.items()}
    max_cluster_id, max_prob = None, 0
    for cluster, prob in probabilities.items():
        if prob > max_prob:
            max_cluster_id, max_prob = cluster, prob
    cluster_to_n[max_cluster_id] += n_samples - sum(cluster_to_n.values())

    cluster_to_len = {k: len(v) for k, v in cluster_to_new_mols.items()}
    balanced = balance_cluster_to_n(cluster_to_n, cluster_to_len)

    training: List[str] = []
    np.random.seed(42)
    for i, (cluster, n) in enumerate(balanced.items()):
        training.extend(
            np.random.choice(cluster_to_new_mols[cluster], n, replace=False)
        )
    assert len(training) == n_samples, f"{len(training)=} != {n_samples=}"
    return training


def construct_al_training_set(config: Config, do_sampling: bool = True) -> pd.DataFrame:
    size = cast(int, config.alconstruction_parameters["training_size"])
    top_ligands = find_top_ligands(config)
    keyToData: Dict[str, List[str]] = {"smiles": []}
    if do_sampling:
        sampled = sample_molecules_from_clusters(config, n_samples=size // 2)
        keyToData["smiles"].extend(sampled)
    if config.alconstruction_parameters["n_replicate"] is True:
        multiplier = int(np.ceil(size // 2 / len(top_ligands)))
        keyToData["smiles"].extend(top_ligands['smiles'].to_list() * multiplier)

    combined = pd.DataFrame(keyToData)
    combined.to_csv(config.cycle_temp_params["path_to_al_training_set"])
    return combined
