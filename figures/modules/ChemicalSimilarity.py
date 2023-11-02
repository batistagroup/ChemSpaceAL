from rdkit import Chem
from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from rdkit.Chem import Mol
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Avalon import pyAvalonTools
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Union, List, Dict, cast, Callable, Optional, Any
import pprint
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.loaders

pp = pprint.PrettyPrinter(indent=2, width=100)

imatinib = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
nilotinib = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)C(=O)Nc4cc(cc(c4)n5cc(nc5)C)C(F)(F)F"
dasatinib = "Cc1cccc(c1NC(=O)c2cnc(s2)Nc3cc(nc(n3)C)N4CCN(CC4)CCO)Cl"
bosutinib = "Clc1c(OC)cc(c(Cl)c1)Nc4c(C#N)cnc3cc(OCCCN2CCN(CC2)C)c(OC)cc34"
ponatinib = "Cc1ccc(cc1C#Cc2cnc3n2nccc3)C(=O)Nc4ccc(c(c4)C(F)(F)F)CN5CCN(CC5)C"
bafetinib = "CC1=C(C=C(C=C1)NC(=O)C2=CC(=C(C=C2)CN3CC[C@@H](C3)N(C)C)C(F)(F)F)NC4=NC=CC(=N4)C5=CN=CN=C5"
ABL_BINDERS = {
    "imatinib": imatinib,
    "nilotinib": nilotinib,
    "dasatinib": dasatinib,
    "bosutinib": bosutinib,
    "ponatinib": ponatinib,
    "bafetinib": bafetinib,
}
ALL_SIMILARITY_METRICS = [
    "Tanimoto",
    "Dice",
    "Asymmetric",
    "Kulczynski",
    "Sokal",
    "Cosine",
]
ALL_FP_TYPES = [
    "ECFP",
    "FCFP",
    "MACCS",
    "RDKit FP",
    "Atom-Pair FP",
    "Topological Torsion FP",
    "Avalon FP",
]


def create_mol_from_smile(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def compute_fingerprint(
    mol: Mol, fingerprint_type: str = "ECFP", radius: int = 2, n_bits: int = 1024
) -> ExplicitBitVect:
    match fingerprint_type:
        case "ECFP":
            fpgen = AllChem.GetMorganGenerator(radius=radius)
            fp = fpgen.GetFingerprint(mol)
        case "FCFP":
            invgen = AllChem.GetMorganFeatureAtomInvGen()
            ffpgen = AllChem.GetMorganGenerator(
                radius=2, atomInvariantsGenerator=invgen
            )
            fp = ffpgen.GetFingerprint(mol)
        case "MACCS":
            fp = MACCSkeys.GenMACCSKeys(mol)
        case "RDKit FP":
            fpgen = AllChem.GetRDKitFPGenerator()
            fp = fpgen.GetFingerprint(mol)
        case "Atom-Pair FP":
            fpgen = AllChem.GetAtomPairGenerator()
            fp = fpgen.GetFingerprint(mol)
        case "Topological Torsion FP":
            fpgen = AllChem.GetTopologicalTorsionGenerator()
            fp = fpgen.GetFingerprint(mol)
        case "Avalon FP":
            fp = pyAvalonTools.GetAvalonFP(mol)
    return fp


def compute_similarity(
    fp1: ExplicitBitVect,
    fp2: ExplicitBitVect,
    similarity_type: str = "Tanimoto",
) -> float:
    match similarity_type:
        case "Tanimoto":
            score = DataStructs.TanimotoSimilarity(fp1, fp2)
        case "Dice":
            score = DataStructs.DiceSimilarity(fp1, fp2)
        case "Asymmetric":
            score = DataStructs.AsymmetricSimilarity(fp1, fp2)
        case "Russel":
            score = DataStructs.RusselSimilarity(fp1, fp2)
        case "Sokal":
            score = DataStructs.SokalSimilarity(fp1, fp2)
        case "Cosine":
            score = DataStructs.CosineSimilarity(fp1, fp2)
        case "Kulczynski":
            score = DataStructs.KulczynskiSimilarity(fp1, fp2)
        # case "Tversky": # needs sparse bit vectors
        #     score = DataStructs.TverskySimilarity(fp1, fp2)
    return score


def compute_abl_inhibitors_similarity_matrix(
    fp_type: str, sim_type: str
) -> pd.DataFrame:
    assert fp_type in ALL_FP_TYPES, f"{fp_type} is not a supported fingerprint type"
    assert (
        sim_type in ALL_SIMILARITY_METRICS
    ), f"{sim_type} is not a supported similarity metric"
    mols = [create_mol_from_smile(smile) for smile in ABL_BINDERS.values()]
    if fp_type == "ECFP" or fp_type == "FCFP":
        extra_params = dict(radius=2, n_bits=2048)
    else:
        extra_params = dict()
    fingerprints = [compute_fingerprint(mol, fp_type, **extra_params) for mol in mols]
    matrix: List[List[[Union[float, None]]]] = []
    for i, r_fp in enumerate(fingerprints):
        row: List[Union[float, None]] = []
        for j, c_fp in enumerate(fingerprints):
            # Only calculate similarity for the lower triangle, including diagonal
            if j >= i:
                row.append(compute_similarity(r_fp, c_fp, sim_type))
            else:
                row.append(None)  # Set the upper triangle values as None
        matrix.append(row)
    df = pd.DataFrame(
        matrix, columns=list(ABL_BINDERS.keys()), index=list(ABL_BINDERS.keys())
    )
    return df


def create_abl_trace(fp_type: str, sim_type: str) -> go.Heatmap:
    df = compute_abl_inhibitors_similarity_matrix(fp_type, sim_type)
    text = [
        [f"{val:.1f}" if not np.isnan(val) else "" for val in row] for row in df.values
    ]
    return go.Heatmap(
        z=df,
        x=df.columns,
        y=df.columns,
        colorscale="Magma",
        zmin=0,
        zmax=1,
        text=text,
        texttemplate="%{text:.1f}",
    )


def create_abl_inhibitors_heatmap_all_simtypes(
    fp_type: str,
    sim_types: List[str],
    n_cols: int = 3,
    h_space: float = 0.01,
    v_space: float = 0.1,
) -> go.Figure:
    n_rows = -(-len(sim_types) // n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=v_space,
        horizontal_spacing=h_space,
        subplot_titles=[f"<b>{sim}</b>" for sim in sim_types],
    )
    for i, sim_type in enumerate(sim_types):
        row, col = i // n_cols + 1, i % n_cols + 1
        fig.add_trace(create_abl_trace(fp_type, sim_type), row=row, col=col)

    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            if row != n_rows:
                fig.update_xaxes(visible=False, row=row, col=col)
            if col != 1:
                fig.update_yaxes(visible=False, row=row, col=col)
    return fig


def compare_smiles_to_single_abl_binder(
    smiles: List[str], abl_binder: str, fp_type: str, sim_type: str
) -> List[float]:
    assert abl_binder in ABL_BINDERS, f"{abl_binder} is not a supported ABL binder"
    assert fp_type in ALL_FP_TYPES, f"{fp_type} is not a supported fingerprint type"
    assert (
        sim_type in ALL_SIMILARITY_METRICS
    ), f"{sim_type} is not a supported similarity metric"
    abl_mol = create_mol_from_smile(ABL_BINDERS[abl_binder])
    abl_fp = compute_fingerprint(abl_mol, fp_type)
    scores = []
    for smile in tqdm(smiles):
        mol = create_mol_from_smile(smile)
        mol_fp = compute_fingerprint(mol, fp_type)
        score = compute_similarity(abl_fp, mol_fp, sim_type)
        scores.append(score)
    return scores


def compare_smiles_to_fingerprint(
    smiles: List[str], reference_fp: str, fp_type: str, sim_type: str
) -> List[float]:
    assert fp_type in ALL_FP_TYPES, f"{fp_type} is not a supported fingerprint type"
    assert (
        sim_type in ALL_SIMILARITY_METRICS
    ), f"{sim_type} is not a supported similarity metric"
    scores = []
    for smile in smiles:
        mol = create_mol_from_smile(smile)
        mol_fp = compute_fingerprint(mol, fp_type)
        score = compute_similarity(reference_fp, mol_fp, sim_type)
        scores.append(score)
    return scores


def analyze_scores(plain_scores: List[float]):
    scores: np.ndarray = np.array(plain_scores)
    # find Q1 and Q3
    q1, q3 = np.percentile(scores, [25, 75])
    # find 95th percentile
    p95 = np.percentile(scores, 95)
    print(f"Min: {scores.min():.3f}")
    print(f"Q1: {q1:.3f}")
    print(f"Median: {np.median(scores):.3f}")
    print(f"Mean: {scores.mean():.3f}")
    print(f"Q3: {q3:.3f}")
    print(f"95th percentile: {p95:.3f}")
    print(f"Max: {scores.max():.3f}")


def compare_smiles_to_abl_binders(
    smiles: List[str], fp_type: str, sim_type: str, metrics: List[str]
) -> Dict[str, List[float]]:
    assert fp_type in ALL_FP_TYPES, f"{fp_type} is not a supported fingerprint type"
    assert (
        sim_type in ALL_SIMILARITY_METRICS
    ), f"{sim_type} is not a supported similarity metric"
    metricToFunc = {
        "min": np.min,
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
    }
    assert all(
        metric in metricToFunc for metric in metrics
    ), f"{metrics} has a metric, which is not supported"
    abl_mols = [create_mol_from_smile(smile) for smile in ABL_BINDERS.values()]
    abl_fps = [compute_fingerprint(mol, fp_type) for mol in abl_mols]
    scores: Dict[str, List[float]] = {metric: [] for metric in metrics}
    for abl_fp in abl_fps:
        sim_scores = compare_smiles_to_fingerprint(smiles, abl_fp, fp_type, sim_type)
        for metric in metrics:
            func = cast(Callable, metricToFunc[metric])
            scores[metric].append(func(sim_scores))
    return scores


def create_similarity_al_trace(
    scores: List[List[float]],
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    showscale: bool = True,
    colorbar: Optional[dict] = None,
) -> List[go.Heatmap]:
    rev_scores = scores[::-1]
    if zmin is None:
        zmin = min(min(row) for row in rev_scores)
    if zmax is None:
        zmax = max(max(row) for row in rev_scores)
    update = {}
    if colorbar is not None:
        update["colorbar"] = colorbar
    return go.Heatmap(
        z=rev_scores,
        x=list(ABL_BINDERS.keys()),
        y=[f"AL{i}" for i in range(len(scores) - 1, -1, -1)],
        colorscale="Magma",
        zmin=zmin,
        zmax=zmax,
        text=rev_scores,
        texttemplate="%{text:.2f}",
        showscale=showscale,
        **update,
    )


def create_mean_max_similarity_figure(
    smiles_lists: List[List[str]],
    fp_type: str,
    sim_type: str,
    colorbars: List[Any],
    h_space: float = 0.01,
    v_space: float = 0.1,
    mean_zmin: Optional[float] = None,
    mean_zmax: Optional[float] = None,
    max_zmin: Optional[float] = None,
    max_zmax: Optional[float] = None,
) -> go.Figure:
    assert fp_type in ALL_FP_TYPES, f"{fp_type} is not a supported fingerprint type"
    assert (
        sim_type in ALL_SIMILARITY_METRICS
    ), f"{sim_type} is not a supported similarity metric"
    mean_lists, max_lists = [], []
    for smiles in smiles_lists:
        scores = compare_smiles_to_abl_binders(
            smiles, fp_type, sim_type, metrics=["mean", "max"]
        )
        mean_lists.append(scores["mean"])
        max_lists.append(scores["max"])
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["<b>Average Similarity</b>", "<b>Maximum Similarity</b>"],
        vertical_spacing=v_space,
        horizontal_spacing=h_space,
    )
    # zmin = min(min(row) for lists in (mean_lists, max_lists) for row in lists)
    # zmax = max(max(row) for lists in (mean_lists, max_lists) for row in lists)
    mean_trace = create_similarity_al_trace(mean_lists, colorbar=colorbars[0], zmin=mean_zmin, zmax=mean_zmax)
    max_trace = create_similarity_al_trace(max_lists, colorbar=colorbars[1], zmin=max_zmin, zmax=max_zmax)
    fig.add_trace(mean_trace, row=1, col=1)
    fig.add_trace(max_trace, row=1, col=2)
    return fig


prepare_scored_fnames = tools.loaders.setup_fname_generator("mix_k100")
prepare_loader = tools.loaders.prepare_loader
if __name__ == "__main__":
    fnames = prepare_scored_fnames(
        prefix="model2_hnh",
        n_iters=5,
        channel="admetfg_softsub",
        filters="ADMET+FGs",
        target="HNH",
    )
    pp.pprint(fnames)
    mol1 = create_mol_from_smile(imatinib)
    mol2 = create_mol_from_smile(dasatinib)
    # fp_type = "ECFP"
    # fp1 = compute_fingerprint(mol1, fp_type)
    # fp2 = compute_fingerprint(mol2, fp_type)
    # sim_type = "Russel"
    # print(len(fp1), len(fp2))
    # score = DataStructs.RusselSimilarity(fp1, fp1)
    # score = compute_similarity(fp1, fp1, sim_type)
    # print(score)
    # types = []
    # for fp_type in [
    #     "ECFP",
    #     "FCFP",
    #     "MACCS",
    #     "RDKit FP",
    #     "Atom-Pair FP",
    #     "Topological Torsion FP",
    #     "Avalon FP",
    # ]:
    #     fp1 = compute_fingerprint(mol1, fp_type)
    #     fp2 = compute_fingerprint(mol2, fp_type)
    #     types.append(type(fp1))
    #     types.append(type(fp2))
    #     for sim_type in [
    #         "Tanimoto",
    #         "Dice",
    #         "Asymmetric",
    #         "Russel",
    #         "Sokal",
    #         "Cosine",
    #         "Kulczynski",
    #     ]:
    #         score = compute_similarity(fp1, fp2, sim_type)
    #         print(f"{fp_type} {sim_type} score: {score:.3f}")
    # print(types)
    # df = compute_abl_inhibitors_similarity_matrix(fp_type="MACCS", sim_type="Tanimoto")
    pass
