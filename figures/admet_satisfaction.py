import modules.secret
import modules.Graph as Graph
import modules.FilterPassing as flt_pass
import pickle
import os
from typing import List

GENERATIONS_PATH = modules.secret.PRODUCTION_RUNS_PATH + "2. Generation/smiles/"
EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "admet_satisfaction", "")

n_iters = 5
ignored = {"fChar"}
configs = [
    ("model7_hnh_admet", "HNH", "ADMET", "softsub"),
    ("model7_hnh_admetfg", "HNH", "ADMET+FGs", "softsub"),
    ("model2_hnh", "HNH", "ADMET+FGs", "admetfg_softsub"),
    ("model7_1iep_admet", "1IEP", "ADMET", "softsub"),
    ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "softsub"),
    ("model2_1iep", "1IEP", "ADMET+FGs", "admetfg_softsub"),
]
rerun_admet = False
max_val = -float("inf")
for prefix, target, filters, channel in configs:
    print(prefix, target, filters, channel)
    if rerun_admet:
        fnames = flt_pass.prepare_generation_fnames(
            prefix=prefix,
            n_iters=n_iters,
            channel=channel,
            filters=filters,
            target=target,
        )
        load_generation = flt_pass.prepare_generation_loader(base_path=GENERATIONS_PATH)
        traces_lists: List[flt_pass.Trace] = []
        filtered_dicts = []
        max_val = -float("inf")
        for i, fname in enumerate(fnames):
            smiles = load_generation(fname)
            filtToData = flt_pass.compute_admet_metrics(smiles)
            filtered_dicts.append(filtToData)
            pickle.dump(
                filtered_dicts,
                open(EXPORT_PATH + f"{prefix}_{filters}_{target}_dicts.pkl", "wb"),
            )
            traces, i_max_val = flt_pass.create_admet_metrics_traces(
                filtToData,
                showlegend=i == 0,
                ignored_metrics=ignored,
                distribution_upper_percentile=100,
            )
            max_val = max(max_val, i_max_val)
            traces_lists.append(traces)
    else:
        filtered_dicts = pickle.load(
            open(EXPORT_PATH + f"{prefix}_{filters}_{target}_dicts.pkl", "rb")
        )
        traces_lists = []
        for i, filtToData in enumerate(filtered_dicts):
            traces, i_max_val = flt_pass.create_admet_metrics_traces(
                filtToData,
                showlegend=i == 0,
                ignored_metrics=ignored,
                distribution_upper_percentile=95,
            )
            max_val = max(max_val, i_max_val)
            traces_lists.append(traces)
    max_val = 1.116
    fig = flt_pass.create_admet_progression_figure(
        traces_lists, v_space=0.1, h_space=0.08, y_max=max_val + 0.05
    )
    graph = Graph.Graph()
    graph.update_parameters(dict(width=1000, height=700, annotation_size=28))
    graph.style_figure(fig)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            x=1.0,
            y=0.5,
            font=dict(size=18),
        ),
    )
    graph.save_figure(
        figure=fig, path=EXPORT_PATH, fname=f"{prefix}_{filters}_{target}"
    )
    # print(max_val)