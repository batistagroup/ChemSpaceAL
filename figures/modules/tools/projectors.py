from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from typing import Dict, Optional, Union, Set, List, Any
from tqdm import tqdm

Number = Union[int, float]
def calculate_descriptors(smiles_set: Set[str], desc_mode:str) -> pd.DataFrame:
    match desc_mode:
        case "mix":
            func = rdkit.Chem.Descriptors.CalcMolDescriptors
        case "mqn":
            func = rdMolDescriptors.MQNs_
    keySet = None
    smileToData: Dict[str, List[Number]] = {}
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
    df = pd.DataFrame(smileToData)
    return df


def reduce_dataframe(
    data: pd.DataFrame,
    reduction: str,
    pca_path: str,
    pca_fname: str,
    reduction_parameters: Optional[dict] = None,
) -> np.ndarray:
    assert reduction in {"PCA", "UMAP", "t-SNE"}
    if reduction_parameters is None:
        reduction_parameters = {}
    # scores = data[score_key].to_numpy()
    scaler, pca = pickle.load(open(pca_path + pca_fname + ".pkl", "rb"))
    features = scaler.get_feature_names_out()
    if reduction_parameters.get("refit_pca", False):
        scaler = StandardScaler()
        pca = PCA(n_components=reduction_parameters["n_components"])
        scaled = scaler.fit_transform(features)
        transformed = pca.fit_transform(scaled)
    else:
        transformed = pca.transform(scaler.transform(data[features]))
    match reduction:
        case "PCA":
            if reduction_parameters.get("reduce_to_2d", False):
                transformed = transformed[:, :2]
            return transformed
        case "UMAP":
            raise NotImplementedError
            # reducer = umap.UMAP(
            #     metric="euclidean",
            #     n_components=2,
            #     random_state=42,
            #     **reduction_parameters,
            # )
            # return reducer.fit_transform(transformed)
        case "t-SNE":
            tsne = TSNE(
                n_components=2,
                random_state=42,
                metric="euclidean",
                **reduction_parameters,
            )
            return tsne.fit_transform(transformed)
        case _:
            raise ValueError(f"Unknown reduction method: {reduction}")
