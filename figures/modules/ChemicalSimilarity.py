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
from typing import Union, List

imatinib = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
nilotinib = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)C(=O)Nc4cc(cc(c4)n5cc(nc5)C)C(F)(F)F"
dasatinib = "Cc1cccc(c1NC(=O)c2cnc(s2)Nc3cc(nc(n3)C)N4CCN(CC4)CCO)Cl"
bosutinib = "Clc1c(OC)cc(c(Cl)c1)Nc4c(C#N)cnc3cc(OCCCN2CCN(CC2)C)c(OC)cc34"
ponatinib = "Cc1ccc(cc1C#Cc2cnc3n2nccc3)C(=O)Nc4ccc(c(c4)C(F)(F)F)CN5CCN(CC5)C"
bafetinib = "CC1=C(C=C(C=C1)NC(=O)C2=CC(=C(C=C2)CN3CC[C@@H](C3)N(C)C)C(F)(F)F)NC4=NC=CC(=N4)C5=CN=CN=C5"
BINDERS = [imatinib, nilotinib, dasatinib, bosutinib, ponatinib, bafetinib]
BINDER_NAMES = [
    "imatinib",
    "nilotinib",
    "dasatinib",
    "bosutinib",
    "ponatinib",
    "bafetinib",
]
ALL_SIMILARITY_METRICS = ["Tanimoto", "Dice", "Asymmetric", "Russel", "Sokal", "Cosine"]


def create_mol_from_smile(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def compute_fingerprint(
    mol: Mol, fingerprint_type: str = "ECFP", radius: int = 4, n_bits: int = 1024
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
        # case "Tversky": # needs sparse bit vectors
        #     score = DataStructs.TverskySimilarity(fp1, fp2)
    return score


def compute_abl_inhibitors_similarity_matrix(
    fp_type: str = "MACCS", sim_type: str = "Tanimoto"
) -> pd.DataFrame:
    mols = [create_mol_from_smile(smile) for smile in BINDERS]
    fingerprints = [compute_fingerprint(mol, fp_type) for mol in mols]
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
    df = pd.DataFrame(matrix, columns=BINDER_NAMES, index=BINDER_NAMES)
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


if __name__ == "__main__":
    # mol1 = create_mol_from_smile(imatinib)
    # mol2 = create_mol_from_smile(dasatinib)
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
    #     for sim_type in ["Tanimoto", "Dice", "Asymmetric", "Russel", "Sokal", "Cosine"]:
    #         score = compute_similarity(fp1, fp2, sim_type)
    #         print(f"{fp_type} {sim_type} score: {score:.3f}")
    # print(types)
    # df = compute_abl_inhibitors_similarity_matrix(fp_type="MACCS", sim_type="Tanimoto")
    pass
