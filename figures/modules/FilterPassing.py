from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from typing import Dict, Union, Callable, List, cast, Tuple, Set, Optional
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.loaders

RDLogger.DisableLog("rdApp.*")

Number = Union[int, float]
AdmetDict = Dict[str, Dict[str, Union[Callable, Number]]]
Trace = Union[go.Scatterpolar]
FUNC_ADMET: AdmetDict = {
    "MW": {"func": lambda mol: Descriptors.MolWt(mol), "lower": 100, "upper": 600},
    "nHA": {
        "func": lambda mol: Descriptors.NumHAcceptors(mol),
        "lower": 0,
        "upper": 12,
    },
    "nHD": {"func": lambda mol: Descriptors.NumHDonors(mol), "lower": 0, "upper": 7},
    "nRot": {
        "func": lambda mol: Descriptors.NumRotatableBonds(mol),
        "lower": 0,
        "upper": 11,
    },
    "nRing": {
        "func": lambda mol: Descriptors.RingCount(mol),
        "lower": 0,
        "upper": 6,
    },  # AdMET recommends [0,6], QED recommends >0
    "nHet": {
        "func": lambda mol: Descriptors.NumHeteroatoms(mol),
        "lower": 1,
        "upper": 15,
    },
    "fChar": {"func": lambda mol: Chem.GetFormalCharge(mol), "lower": -4, "upper": 4},
    "TPSA": {"func": lambda mol: Descriptors.TPSA(mol), "lower": 0, "upper": 140},
    "logP": {
        "func": lambda mol: Descriptors.MolLogP(mol),
        "lower": -0.4,
        "upper": 6.5,  # TODO: specify which Abl binder has log.p 6.3
    },  # AdMET Lab recommends [0,3], [-0.4, 5.6] from Ghose
}


prepare_generation_fnames = tools.loaders.setup_generations_fname_generator(
    "temp1.0_completions", filters=True
)
prepare_generation_loader = tools.loaders.prepare_loader
# def prepare_generation_fnames(
#     prefix: str,
#     n_iters: int,
#     channel: str,
#     filters: str,
#     target: str,
# ) -> List[str]:
#     assert filters in {"ADMET", "ADMET+FGs"}
#     if "model2" in prefix:
#         fnames = [
#             f"{prefix[:6]}_baseline_{target.upper()}_temp1.0_completions_{filters}",
#             *(
#                 f"{prefix}_al{i}_{channel}_{target.upper()}_temp1.0_completions_{filters}"
#                 for i in range(1, n_iters + 1)
#             ),
#         ]
#     elif "model7" in prefix:
#         fnames = [
#             f"{prefix[:6]}_baseline_{target.upper()}_temp1.0_completions_{filters}",
#             *(
#                 f"{prefix}_{channel}_al{i}_{target.upper()}_temp1.0_completions_{filters}"
#                 for i in range(1, n_iters + 1)
#             ),
#         ]
#     return fnames


# def prepare_generation_loader(base_path: str):
#     def _load_generation(fname: str):
#         return pd.read_csv(base_path + fname + ".csv")["smiles"].to_list()

#     return _load_generation


def compute_admet_metrics(completions: List[str]) -> Dict[str, List[Number]]:
    filterToData: Dict[str, List[Number]] = {}
    pbar = tqdm(enumerate(completions), total=len(completions))
    n_filters = len(FUNC_ADMET)
    n_smiles = len(completions)
    for i, completion in pbar:
        if completion[0] == "!" and completion[1] == "~":
            completion = "!" + completion[2:]
        if "~" not in completion:
            continue
        smile = completion[1 : completion.index("~")]
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        for j, (filterName, filterDict) in enumerate(FUNC_ADMET.items()):
            pbar.set_description(
                f"{i}/{n_smiles} completions, {j}/{n_filters} filters processed"
            )
            assert callable(filterFunc := filterDict["func"])
            filterToData.setdefault(filterName, []).append(filterFunc(mol))
    return filterToData


def create_admet_metrics_traces(
    admet_metrics: Dict[str, List[Number]],
    showlegend: bool = True,
    ignored_metrics: Optional[Set[str]] = None,
    distribution_upper_percentile: float = 100,
) -> Tuple[List[go.Scatterpolar], Number]:
    if ignored_metrics is None:
        ignored_metrics = set()
    scaled_metrics = {
        metric: [value / cast(Number, FUNC_ADMET[metric]["upper"]) for value in values]
        for metric, values in admet_metrics.items()
        if metric not in ignored_metrics
    }
    metrics = list(scaled_metrics.keys()) + [list(scaled_metrics.keys())[0]]
    avg_values, min_values, upper_values = [], [], []
    for metric in metrics:
        avg_values.append(np.mean(scaled_metrics[metric]))
        min_values.append(min(scaled_metrics[metric]))
        if distribution_upper_percentile == 100:
            upper_values.append(np.max(scaled_metrics[metric]))
        else:
            upper_values.append(
                np.percentile(scaled_metrics[metric], distribution_upper_percentile)
            )
    traces = []

    traces.append(
        go.Scatterpolar(
            r=min_values,
            theta=metrics,
            name="Lower Bound",
            line_color="black",
            showlegend=showlegend,
        )
    )
    traces.append(
        go.Scatterpolar(
            r=avg_values,
            theta=metrics,
            fill="tonext",
            name="Average Value",
            line_color="#072ac8",
            showlegend=showlegend,
        )
    )
    if distribution_upper_percentile == 100:
        name = "Max. Value"
    else:
        name = f"{distribution_upper_percentile}% percentile"
    traces.append(
        go.Scatterpolar(
            r=upper_values,
            theta=metrics,
            name=name,
            fill="tonext",
            line_color="#2196f3",
            showlegend=showlegend,
        )
    )
    traces.append(
        go.Scatterpolar(
            r=[1 for _ in range(len(metrics))],
            theta=metrics,
            name="Upper Bound",
            line_color="#c9184a",
            showlegend=showlegend,
        )
    )
    return traces, max(upper_values)


def create_admet_progression_figure(
    traces_lists: List[List[go.Scatterpolar]],
    n_cols: int = 3,
    h_space: float = 0.01,
    v_space: float = 0.1,
    y_max: float = 1.0,
):
    n_rows = -(-len(traces_lists) // n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=v_space,
        horizontal_spacing=h_space,
        subplot_titles=[f"<b>Iteration {i}</b>" for i in range(0, len(traces_lists))],
        specs=[[{"type": "polar"}] * 3 for _ in range(n_rows)],
    )
    for i, traces in enumerate(traces_lists):
        row, col = i // n_cols + 1, i % n_cols + 1
        for trace in traces:
            fig.add_trace(trace, row=row, col=col)
    for i in range(len(traces_lists)):
        polar_key = f'polar{"" if i == 0 else i + 1}'
        fig.update_layout(
            {
                polar_key: dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, y_max + 0.1],
                        gridcolor="#cecece",
                        showticklabels=False,
                    ),
                    angularaxis=dict(gridcolor="#cecece"),
                    bgcolor="white",
                )
            }
        )
    return fig


if __name__ == "__main__":
    pass
    # prefix = "model2_hnh"
    # n_iters = 5
    # channel = "admetfg_softsub"
    # filters = "ADMET+FGs"
    # target = "HNH"
    # ignored: Set[str] = set()  # {"fChar"}
    # fnames = prepare_generation_fnames(
    #     prefix=prefix, n_iters=n_iters, channel=channel, filters=filters, target=target
    # )
    # load_generation = prepare_generation_loader(
    #     base_path="/Users/morgunov/batista/Summer/pipeline/2. Generation/smiles/"
    # )
    # traces_lists = []
    # traces_lists = pickle.load(open("traces_lists.pkl", "rb"))
    # # print(len(traces_lists))
    # max_val = -float("inf")
    # for i, fname in enumerate(fnames):
    #     if i < 5:
    #         continue
    #     smiles = load_generation(fname)
    #     filtToData = compute_admet_metrics(smiles[:10])
    #     traces, i_max_val = create_admet_metrics_traces(
    #         filtToData, showlegend=i == 0, ignored_metrics=ignored
    #     )
    #     max_val = max(max_val, i_max_val)
    #     traces_lists.append(traces)
    #     # pickle.dump(traces_lists, open("traces_lists.pkl", "wb"))
