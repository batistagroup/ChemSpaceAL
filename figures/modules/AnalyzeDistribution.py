import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.loaders
import tools.projectors
from typing import List, Optional, Tuple, Union, Callable, Dict
import pandas as pd
import numpy as np
import scipy.stats
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

prepare_scored_fnames = tools.loaders.setup_scored_fname_generator(filters=True)
prepare_generations_fnames = tools.loaders.setup_generations_fname_generator(
    presuffix="temp1.0_processed", filters=True
)
prepare_altrains_fnames = tools.loaders.setup_altrains_fname_generator(filters=True)
prepare_diffusion_fnames = tools.loaders.setup_altrains_fname_generator(
    filters=True, no_score=True
)
prepare_loader = tools.loaders.prepare_loader
Number = Union[int, float]
np.random.seed(42)


def _calculate_descriptors_for_an_array(
    smiles: List[str], save_path: str, save_name: str, desc_mode: Optional[str] = "mix"
):
    df = tools.projectors.calculate_descriptors(smiles, desc_mode)
    df.to_pickle(os.path.join(save_path, save_name + ".pkl"))


def load_descriptors(load_path: str, load_fname: str) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(load_path, load_fname + ".pkl"))


def project_smiles(smiles: List[str], descriptors: pd.DataFrame) -> pd.DataFrame:
    return descriptors[descriptors["smiles"].isin(smiles)]


BoundaryTuple = Tuple[np.ndarray, np.ndarray]


def load_all_scores(fnames, loader, merge: bool = True):
    scored_dfs = [loader(fname) for fname in fnames]
    if merge:
        return pd.concat(scored_dfs).drop_duplicates(subset=["smiles"])
    else:
        return scored_dfs


def load_all_smiles(
    fnames, loader, sample: Optional[int] = None, verbose: bool = False
):
    if sample is None:
        smiles = [loader(fname) for fname in fnames]
    else:
        smiles = [
            np.random.choice(list(set(loader(fname))), size=sample, replace=False)
            for fname in fnames
        ]
    if verbose:
        unique_counts = [len(set(s)) for s in smiles]
        print(f"Unique counts: {unique_counts}")
    return smiles


def load_smiles_for_config(
    configs: List[Tuple[str, ...]],
    gen_loader: Callable,
    scored_loader: Callable,
    altrain_loader: Callable,
    save_path: str,
    save_fname: str,
):
    n_iters = 5
    scores_dfs = []
    generated_smiles = []
    altrains_smiles = []
    diffusion_smiles = []
    for prefix, target, filters, channel in configs:
        scored_fnames = prepare_scored_fnames(prefix, n_iters, channel, filters, target)
        generated_fnames = prepare_generations_fnames(
            prefix, n_iters, channel, filters, target
        )
        altrains_fnames, diffusion_fnames = [
            fname_prepper(
                prefix,
                n_iters,
                channel,
                filters,
                target,
                threshold=37,
                conversion_scheme="softmax_sub",
            )
            for fname_prepper in (
                prepare_altrains_fnames,
                prepare_diffusion_fnames,
            )
        ]
        generated_smiles.extend(
            load_all_smiles(generated_fnames, gen_loader, sample=10_000, verbose=True)
        )
        altrains_smiles.extend(
            load_all_smiles(altrains_fnames, altrain_loader, sample=5_000, verbose=True)
        )
        diffusion_smiles.extend(
            load_all_smiles(
                diffusion_fnames, altrain_loader, sample=5_000, verbose=True
            )
        )
        scores_dfs.extend(load_all_scores(scored_fnames, scored_loader, merge=False))
    pickle.dump(
        [scores_dfs, generated_smiles, altrains_smiles, diffusion_smiles],
        open(os.path.join(save_path, "pickles", save_fname + ".pkl"), "wb"),
    )
    return scores_dfs, generated_smiles, altrains_smiles, diffusion_smiles


def calculate_descriptors(containers_list, save_path: str, save_fname: str):
    """
    containers_list is a list of iterables that contain smile strings
    """
    smile_set = set()
    for container in containers_list:
        for smiles in container:
            smile_set |= set(smiles)
    print(f"Loaded {len(smile_set)} unique smiles")
    _calculate_descriptors_for_an_array(list(smile_set), save_path, save_fname)


def fit_reduction_on_smiles(
    containers_list,
    descriptors,
    reduction_config,
    save_path: str,
    save_fname: str,
    pca_path: str,
    pca_fname: str,
):
    smile_set = set()
    for container in containers_list:
        for smiles in container:
            smile_set |= set(smiles)
    projected = project_smiles(list(smile_set), descriptors)
    print(
        f"Projected {len(projected)} smiles, started from {len(smile_set)} unique smiles"
    )
    reduced = tools.projectors.reduce_dataframe(
        data=projected, pca_path=pca_path, pca_fname=pca_fname, **reduction_config
    )
    mapping = dict(zip(projected["smiles"], reduced))
    pickle.dump(mapping, open(save_path + save_fname + ".pkl", "wb"))
    return mapping


# ---------------------
# Visualizations Code


def get_data_boundaries(
    data_list: List[np.ndarray], round_to: int = 10
) -> BoundaryTuple:
    combined = np.vstack(
        data_list
    )  # Combine both datasets to get overall min and max values

    min_val = (
        np.floor(np.min(combined, axis=0) / round_to) * round_to
    )  # Round to the nearest number divisible by 10
    max_val = np.ceil(np.max(combined, axis=0) / round_to) * round_to
    return min_val, max_val


def discretize_data(data: np.ndarray, boundaries: BoundaryTuple, bin_size: Number):
    # Use 2D histogram to discretize data
    bins = [np.arange(boundaries[0][i], boundaries[1][i], bin_size) for i in range(2)]
    hist_data, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
    # Compute bin centers
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    return hist_data.T, xcenters, ycenters


def discretize_data_wscores(
    data: np.ndarray, boundaries: BoundaryTuple, bin_size: Number, scores: pd.Series
):
    bins = [np.arange(boundaries[0][i], boundaries[1][i], bin_size) for i in range(2)]
    hist_data, xedges, yedges, _ = scipy.stats.binned_statistic_2d(
        data[:, 0], data[:, 1], scores, statistic="mean", bins=bins
    )
    hist_data = np.nan_to_num(hist_data, nan=0)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    return hist_data.T, xcenters, ycenters


def create_heatmap_trace(
    data: np.ndarray,
    boundaries: BoundaryTuple,
    bin_size: Number,
    scores: Optional[pd.Series] = None,
    force_zmax: Optional[Number] = None,
    force_zmin: Optional[Number] = None,
    name: str = "Heatmap",
    showscale: bool = True,
    showlegend: bool = True,
    colorscale: str = "Electric",
    coloraxis=None,
    colorbar_dict: dict | None = None,
):
    if colorbar_dict is None:
        colorbar_dict = {}
    functor = (
        lambda x: discretize_data(*x[:-1])
        if scores is None
        else discretize_data_wscores(*x)
    )

    hist, xcenters, ycenters = functor([data, boundaries, bin_size, scores])
    zmax = hist.max()
    zmin = hist.min()
    if force_zmax:
        zmax = force_zmax
    if force_zmin:
        zmin = force_zmin

    return go.Heatmap(
        x=xcenters,
        y=ycenters,
        z=hist,
        name=name,
        zmin=zmin,
        zmax=zmax,
        showlegend=showlegend,
        colorscale=colorscale,
        showscale=showscale,
        visible=True,
        coloraxis=coloraxis,
        colorbar=colorbar_dict,
    )


def create_heatmap_trace_for_difference(
    minuend: np.ndarray,
    subtrahend: np.ndarray,
    boundaries: BoundaryTuple,
    bin_size: Number,
    scores: Optional[pd.Series] = None,
    force_zmax: Optional[Number] = None,
    force_zmin: Optional[Number] = None,
    showscale: bool = True,
    showlegend: bool = True,
    coloraxis: str = "coloraxis",
):
    hist_minuend, xcenters, ycenters = discretize_data(minuend, boundaries, bin_size)
    hist_subtrahend, _, _ = discretize_data(subtrahend, boundaries, bin_size)
    diff = hist_minuend - hist_subtrahend
    zmax = force_zmax if force_zmax else diff.max()
    zmin = force_zmin if force_zmin else diff.min()
    return go.Heatmap(
        x=xcenters,
        y=ycenters,
        z=diff,
        zmax=zmax,
        zmin=zmin,
        zmid=0,
        colorscale="RdBu",
        showlegend=showlegend,
        showscale=showscale,
        coloraxis=coloraxis,
    )


def create_traces_from_difference(
    minuend: List[List[str]],
    subtrahend: List[List[str]],
    mapping: dict,
    boundaries: BoundaryTuple,
    forced_subtrahend_index: int,
    bin_size: Number,
    force_zmax: Optional[Number] = None,
    force_zmin: Optional[Number] = None,
):
    minuend_reduced_list = [
        [mapping[smile] for smile in smile_list] for smile_list in minuend
    ]
    subtrahend_reduced = np.array([
        mapping[smile] for smile in subtrahend[forced_subtrahend_index]
    ])
    return [
        create_heatmap_trace_for_difference(
            minuend=np.array(minuend_reduced),
            subtrahend=subtrahend_reduced,
            boundaries=boundaries,
            bin_size=bin_size,
            force_zmax=force_zmax,
            force_zmin=force_zmin,
            showscale=i == 0,
            showlegend=False,
            coloraxis="coloraxis2",
        )
        for i, minuend_reduced in enumerate(minuend_reduced_list)
    ]


def prepare_scored_traces(
    scored_dfs: List[pd.DataFrame],
    tsne_mapping: Dict[str, np.ndarray],
    pca_mapping: Dict[str, np.ndarray],
    bin_size_factor: int = 2,
    force_zmax: Optional[Number] = None,
    force_zmin: Optional[Number] = None,
    colorscale: str = "Thermal",
):
    all_scores = pd.concat(scored_dfs).drop_duplicates(subset=["smiles"])
    pca_reduced = np.array([pca_mapping[smile] for smile in all_scores["smiles"]])
    tsne_reduced = np.array([tsne_mapping[smile] for smile in all_scores["smiles"]])

    pca_boundaries = get_data_boundaries([pca_reduced])
    tsne_boundaries = get_data_boundaries([tsne_reduced])

    pca_trace = create_heatmap_trace(
        data=pca_reduced,
        boundaries=pca_boundaries,
        scores=all_scores["score"],
        bin_size=0.3 * bin_size_factor,
        force_zmax=force_zmax,
        force_zmin=force_zmin,
        colorscale=colorscale,
        showscale=False,
        showlegend=False,
        # colorbar_title="Attractive<br>Interaction<br>Score",
    )
    tsne_trace = create_heatmap_trace(
        data=tsne_reduced,
        boundaries=tsne_boundaries,
        scores=all_scores["score"],
        bin_size=0.75 * bin_size_factor,
        force_zmax=force_zmax,
        force_zmin=force_zmin,
        colorscale=colorscale,
        showscale=True,
        showlegend=False,
        colorbar_dict=dict(
            title="Attractive<br>Interaction<br>Score", yanchor="top", len=1.1, y=1.07
        ),
        # dict(x=1.01, y=0.49, len=0.5, yanchor="top")
    )
    return [(pca_trace, 1, 1), (tsne_trace, 1, 2)]


def plot_heatmap_wsubplots(
    traces,
    subplot_titles: List[str],
    rows: int,
    cols: int,
    vertical_spacing: float = 0.05,
    horizontal_spacing: float = 0.08,
    shared_xaxes: bool = True,
    shared_yaxes: bool = True,
):
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )
    for trace, row, col in traces:
        fig.add_trace(trace, row=row, col=col)
    return fig
