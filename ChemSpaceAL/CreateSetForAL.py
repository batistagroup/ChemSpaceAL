import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

from .Configuration import *


def compute_cluster_scores(config: dict) -> dict:
    """
    Compute the average scores for each cluster.

    Parameters:
    - config (dict): Configuration dictionary containing paths and other settings.

    Returns:
    - dict: A dictionary with cluster IDs as keys and their average scores as values.
    """
    good_data = pd.read_csv(config["path_to_scored"])
    cluster_to_scores = {}
    for _, row in good_data.iterrows():
        cluster_to_scores.setdefault(row["cluster_id"], []).append(row["score"])
    return {
        cluster_id: np.mean(scores) for cluster_id, scores in cluster_to_scores.items()
    }


def plot_probability_distribution(cluster_scores: dict, args_list=None, bin_size=0.001):
    """
    Plot the probability distribution of the given cluster scores.

    Parameters:
    - cluster_scores (dict): Dictionary with cluster IDs as keys and scores as values.
    - args_list (list, optional): List of arguments for processing scores. Default to None.
    - bin_size (float, optional): Size of bins for histograms. Default to 0.001.

    Returns:
    - plotly.graph_objects.Figure: A plotly figure object.
    """
    if args_list is None:
        args_list = [
            ("softmax", True, 1),
            ("softmax", True, 0.5),
            ("softmax", True, 0.25),
            ("softmax", False, np.nan),
        ]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=[1 / 100 for _ in range(100)],
            name="uniform",
            opacity=0.75,
            histnorm="probability",
            xbins=dict(start=-0.1, end=1, size=bin_size),
        )
    )

    keyToFunc = {"softmax": _preprocess_scores_softmax}
    for func, param_switch, param_value in args_list:
        suffix, extras = "", {}
        if func == "softmax":
            suffix = f" divf {param_value}" if param_switch else ""
            extras = dict(divide=param_switch, divide_factor=param_value)
        fig.add_trace(
            go.Histogram(
                x=list(keyToFunc[func](cluster_scores, **extras).values()),
                name=f"{func}{suffix}",
                opacity=0.75,
                histnorm="probability",
                xbins=dict(start=-0.1, end=1, size=bin_size),
            )
        )

    fig.update_layout(
        barmode="overlay",
        title="Probabilities of sampling from clusters based on mean scores",
        xaxis_title="probability",
        yaxis_title="rel. frequency",
    )
    return fig


def plot_mean_cluster_scores(cluster_scores: dict):
    """
    Plot the distribution of mean cluster scores.

    Parameters:
    - cluster_scores (dict): Dictionary with cluster IDs as keys and scores as values.

    Returns:
    - plotly.graph_objects.Figure: A plotly figure object.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=list(cluster_scores.values()), opacity=0.75, histnorm="probability"
        )
    )
    fig.update_layout(
        barmode="overlay",
        title="Distribution of cluster scores",
        xaxis_title="Mean cluster score",
        yaxis_title="rel. frequency",
    )
    return fig


def balance_cluster_to_n(cluster_to_n: dict, cluster_to_len: dict) -> dict:
    """
    Adjust the cluster counts to not exceed their available samples.

    Parameters:
    - cluster_to_n (dict): Initial cluster counts.
    - cluster_to_len (dict): Maximum available samples for each cluster.

    Returns:
    - dict: Adjusted cluster counts.
    """
    surplus = {
        key: cluster_to_n[key] - cluster_to_len.get(key, 0)
        for key in cluster_to_n
        if cluster_to_n[key] > cluster_to_len.get(key, 0)
    }
    balanced = cluster_to_n.copy()
    n_to_cluster = {v: k for k, v in cluster_to_n.items()}

    for key in surplus:
        balanced[key] = cluster_to_len.get(key, 0)

    total_surplus = sum(surplus.values())
    initial_n_sum = sum(n for key, n in cluster_to_n.items() if key not in surplus)

    for key in balanced:
        if key in surplus:
            continue
        surplus_to_add = total_surplus * cluster_to_n[key] / initial_n_sum
        new_n = int(cluster_to_n[key] + surplus_to_add)
        balanced[key] = min(new_n, cluster_to_len[key])

    deficit = sum(cluster_to_n.values()) - sum(balanced.values())
    max_iterations = 10_000
    current_iteration = 0
    while deficit > 0 and current_iteration < max_iterations:
        for initial_n in sorted(n_to_cluster, reverse=True):
            if deficit == 0:
                break
            if (cluster := n_to_cluster[initial_n]) in surplus:
                continue
            if balanced[cluster] < cluster_to_len[cluster]:
                balanced[cluster] += 1
                deficit -= 1
        current_iteration += 1

    if current_iteration == max_iterations:
        print(f"Warning: Could not balance cluster sizes. Deficit = {deficit}")
    return balanced


def _preprocess_scores_softmax(
    scores: dict, do_negation=False, divide=True, divide_factor=None
) -> dict:
    """
    Process scores using softmax.

    Parameters:
    - scores (dict): Raw scores dictionary.
    - do_negation (bool, optional): Whether to negate the scores. Default to False.
    - divide (bool, optional): Whether to divide by a factor. Default to True.
    - divide_factor (float, optional): The divisor. Default to None.

    Returns:
    - dict: Softmax-processed scores.
    """
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
    return {k: v / total for k, v in exponentiate.items()}


def sample_clusters_for_active_learning(
    cluster_to_scores: dict,
    config: dict,
    probability_type="softmax",
    divide=True,
    divide_factor=0.25,
) -> list:
    """
    Sample clusters based on their scores for active learning.

    Parameters:
    - cluster_to_scores (dict): Dictionary with cluster IDs as keys and scores as values.
    - config (dict): Configuration dictionary containing paths and other settings.
    - probability_type (str, optional): Probability distribution type. Default to 'softmax'.
    - divide (bool, optional): Whether to divide by a factor. Default to True.
    - divide_factor (float, optional): The divisor. Default to 0.25.

    Returns:
    - list: List of sampled clusters.
    """
    if probability_type == "softmax":
        probability_function = lambda x: _preprocess_scores_softmax(
            x, divide=divide, divide_factor=divide_factor
        )
    else:
        raise KeyError("Only uniform and softmax probabilities are supported")

    cluster_to_mols = pickle.load(open(config["clusters_save_path"], "rb"))
    cluster_to_samples = pickle.load(open(config["samples_save_path"], "rb"))
    docked_mols = {smile for smiles in cluster_to_samples.values() for smile in smiles}
    cluster_to_new_mols = {
        k: [smile for smile in set(v) if smile not in docked_mols]
        for k, v in cluster_to_mols.items()
    }

    probabilities = probability_function(cluster_to_scores)
    cluster_to_n = {k: int(v * 5000) for k, v in probabilities.items()}
    max_cluster_id = max(probabilities, key=probabilities.get)
    cluster_to_n[max_cluster_id] += 5000 - sum(cluster_to_n.values())

    cluster_to_len = {k: len(v) for k, v in cluster_to_new_mols.items()}
    balanced = balance_cluster_to_n(cluster_to_n, cluster_to_len)

    training = [
        np.random.choice(cluster_to_new_mols[cluster], n, replace=False)
        for cluster, n in balanced.items()
    ]
    return [item for sublist in training for item in sublist]


def combine_sampled_and_good_ligands(
    sampled: list, good_ligands: list, config: dict
) -> pd.DataFrame:
    """
    Combine sampled molecules with good ligands.

    Parameters:
    - sampled (list): Sampled molecules.
    - good_ligands (list): Good ligands.
    - config (dict): Configuration dictionary containing paths and other settings.

    Returns:
    - pd.DataFrame: Combined dataframe.
    """
    good_ligand_multiplier = int(np.ceil(5000 / len(good_ligands)))
    keyToData = {"smiles": []}

    if sampled:
        keyToData["smiles"].extend(sampled)
    keyToData["smiles"].extend(good_ligands * good_ligand_multiplier)

    combined = pd.DataFrame(keyToData)
    combined.to_csv(config["AL_set_save_path"])
    return combined
