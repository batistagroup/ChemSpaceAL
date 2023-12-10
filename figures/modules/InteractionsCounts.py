from scipy.stats import pearsonr, spearmanr
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde
from .Style import Style
from collections import defaultdict
import itertools
from typing import List
import pickle


def check_for_multiple_ligands(affinities, dataframes):
    for key, df in dataframes.items():
        unique_ligands = df["ligand"].nunique()
        if unique_ligands != 1:
            return True
    return False


def create_pdbbind_figure(
    affinities,
    scores,
    subplot_titles,
    bin_step: int = 2,
):
    fig = make_subplots(subplot_titles=subplot_titles, rows=1, cols=2)
    biscale = Style.biscale

    total_scores = []
    affinity_values = []

    for key in scores:
        total_score = scores[key]["score"].sum()
        affinity_value = affinities[key]

        total_scores.append(total_score)
        affinity_values.append(affinity_value)

    fig.add_trace(
        go.Scatter(
            x=total_scores,
            y=affinity_values,
            mode="markers",
            marker=dict(
                size=2, color="#0077b6", line=dict(width=0, color="DarkSlateGrey")
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fit = np.polyfit(total_scores, affinity_values, 1)
    pearson_corr, _ = pearsonr(total_scores, affinity_values)
    # spearman_corr, _ = spearmanr(total_scores, affinity_values)
    count_above_11 = sum(score >= 11 for score in total_scores)
    percentage_above_11 = (count_above_11 / len(total_scores)) * 100
    print(f"{percentage_above_11=}, {pearson_corr=}")
    line_of_best_fit = np.polyval(fit, total_scores)
    fig.add_trace(
        go.Scatter(
            x=total_scores,
            y=line_of_best_fit,
            mode="lines",
            line=dict(color="#03045e", width=1),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    density = gaussian_kde(total_scores)
    xs = np.linspace(np.min(total_scores), np.max(total_scores), 200)
    density.covariance_factor = lambda: 0.25
    density._compute_covariance()

    hist_vals, bin_edges = np.histogram(
        total_scores,
        bins=range(0, int(np.max(total_scores)) + 2, bin_step),
        density=True,
    )
    hist = go.Bar(
        x=bin_edges[:-1],
        y=hist_vals,
        showlegend=False,
        marker=dict(color="#0077b6"),
    )
    density_curve = go.Scatter(
        x=xs,
        y=density(xs),
        mode="lines",
        line=dict(color="#333", width=2),
        showlegend=False,
    )
    fig.add_trace(hist, row=1, col=2)
    fig.add_trace(density_curve, row=1, col=2)

    formula = "Binding Affinity (p<i>K</i><sub>d</sub>)"
    fig.update_yaxes(
        range=[0, 17.5],
        title=f"{formula}",
        title_standoff=1,
        title_font_size=18,
        row=1,
        col=1,
    )
    fig.add_shape(
        dict(
            type="line",
            x0=11,
            x1=11,
            y0=0,
            y1=0.027,
            yref="paper",  # refers to the entire plot for the y-dimension
            line=dict(color=biscale["dark_grey"][0](0.6), width=2, dash="dash"),
        ),
        row=1,
        col=2,
    )
    fig.add_annotation(
        dict(
            x=32 + 0,
            y=0.025 + 0,
            # xref="paper",
            yref="paper",
            yanchor="bottom",
            text=f"<b>Threshold: 11</b>",
            font_color=biscale["dark_grey"][0](0.6),
            showarrow=False,
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title="Relative Frequency", title_font_size=18, title_standoff=1, row=1, col=2
    )
    fig.update_xaxes(
        title="Attractive Interaction Score", title_font_size=18, row=1, col=1
    )
    fig.update_xaxes(
        title="Attractive Interaction Score", title_font_size=18, row=1, col=2
    )
    for col, text in [(1, "a"), (2, "b")]:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=-0.02,
            y=1.15,
            xanchor="right",
            yanchor="top",
            text=f"<b>[{text.upper()}]</b>",
            font=dict(size=24, family="Helvetica"),
            showarrow=False,
            row=1,
            col=col,
        )
    return fig


def rename_interactions(scores, mapping):
    copy = lambda x: x if isinstance(x, int) else x.copy()
    new_scores = {k: copy(v) for k, v in scores.items()}
    for key, df in new_scores.items():
        if isinstance(df, int):
            continue
        df["interaction"] = df["interaction"].replace(mapping)
    return new_scores


def count_interactions(dataframes, calc_method):
    # Calculate counts for individual interactions
    interaction_counts = defaultdict(int)
    interaction_pair_counts = defaultdict(int)
    unique_interactions = set()

    # perform counts
    for key in dataframes:
        if isinstance(dataframes[key], int):
            continue
        if calc_method == "once":
            interactions = set(dataframes[key]["interaction"])
            unique_interactions |= interactions
        elif calc_method == "each":
            interactions = dataframes[key]["interaction"]
            unique_interactions |= set(interactions)
        else:
            raise ValueError(f"Unknown calc_method: {calc_method}")

        for interaction in interactions:
            interaction_counts[interaction] += 1
        for inter1, inter2 in itertools.combinations(interactions, 2):
            pair = tuple(sorted([inter1, inter2]))
            interaction_pair_counts[pair] += 1
    unique_interactions = list(sorted(unique_interactions))
    return interaction_counts, interaction_pair_counts, unique_interactions


def create_1d_heatmap_array(
    interaction_counts,
    unique_interactions,
):
    heatmap_data = np.zeros(len(unique_interactions))
    total = sum(interaction_counts.values())
    for i, interaction in enumerate(unique_interactions):
        heatmap_data[i] = interaction_counts[interaction]
    return heatmap_data


def create_2d_heatmap_array(
    interaction_counts,
    interaction_pair_counts,
    unique_interactions,
    normalize: bool = True,
    sig_digs: int = 2,
):
    heatmap_data = np.zeros((len(unique_interactions), len(unique_interactions)))

    total = sum(interaction_counts.values())
    if interaction_pair_counts is not None:
        total_pairs = sum(interaction_pair_counts.values())

    # Fill the diagonal
    for i, interaction in enumerate(unique_interactions):
        if normalize:
            heatmap_data[i][i] = np.round(
                interaction_counts[interaction] / total, sig_digs
            )
        else:
            heatmap_data[i][i] = interaction_counts[interaction]

    # Fill the off-diagonal
    for i in range(len(unique_interactions)):
        for j in range(i + 1, len(unique_interactions)):
            pair = (unique_interactions[i], unique_interactions[j])
            if interaction_pair_counts is not None:
                if normalize:
                    heatmap_data[i][j] = np.round(
                        interaction_pair_counts[pair] / total_pairs, sig_digs
                    )
                else:
                    heatmap_data[i][j] = interaction_pair_counts[pair]
            else:
                heatmap_data[i][j] = np.round(
                    heatmap_data[i][i] * heatmap_data[j][j], sig_digs
                )
            heatmap_data[j][i] = 0  # interaction_pair_counts[pair]

    return heatmap_data


def create_interactions_heatmap_figure(
    scoring_path: str,
    fnames: List[str],
    calc_method: str,
    row_titles: List[str],
    labels,
    mapping,
    normalize_counts=True,
    control_dict=None,
):
    # Create subplots
    fig = go.Figure()

    score_dicts = [load_scores_dictionary(scoring_path, fname, mapping, rename=True) for fname in fnames]
    if control_dict is not None:
        if len(control_dict) > 1000:
            keys = list(control_dict.keys())
            np.random.seed(42)
            sampled_keys = np.random.choice(keys, 1000, replace=False)
            sampled_dict = {k: control_dict[k] for k in sampled_keys}
            score_dicts = [sampled_dict] + score_dicts
        else:
            score_dicts = [control_dict] + score_dicts

    heat_list = [
        create_1d_heatmap_array(count_interactions(dic, calc_method)[0], labels)
        for dic in score_dicts
    ]

    heats = np.vstack(heat_list)
    heats = heats[::-1, :]

    def format_text(ndarray):
        text = ndarray.astype(str)
        # text[np.tril_indices(ndarray.shape[0], k=-1)] = ""
        return text

    fig.add_trace(
        go.Heatmap(
            x=labels,
            y=row_titles[::-1],
            z=heats,
            text=format_text(heats),
            texttemplate="%{text}",
            showscale=False,
            zsmooth=False,
            zmax=600,
            zmin=0,
            colorscale="Blues",
        ),
    )

    fig.update_layout(
        uniformtext_minsize=8, 
        uniformtext_mode="hide", 
    )
    return fig


def load_scores_dictionary(scoring_path:str, fname:str, mapping, rename=True):
    scores = pickle.load(open(scoring_path+ fname + ".pkl", "rb"))
    if not rename:
        return scores
    return rename_interactions(scores, mapping)
