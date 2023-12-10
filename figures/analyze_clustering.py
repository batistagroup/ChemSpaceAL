import modules.secret
import os
import modules.ClusterAnalysis as ca

SAMPLING_PATH = modules.secret.PRODUCTION_RUNS_PATH + "3. Sampling/kmeans_objects/"
EXPORT_PATH = os.path.join(
    os.getcwd(), "figures", "exports", "cluster_distribution", ""
)

import pickle


configs = [
    ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "softsub"),
    # ("model2_1iep", "1IEP", "ADMET+FGs", "admetfg_softsub"),
    # ("model7_1iep_admet", "1IEP", "ADMET", "softsub"),
]
n_iters = 5
from modules.Graph import Graph

for prefix, target, filters, channel in configs:
    loader = ca.prepare_kmeans_fnames(filters="ADMET+FGs")
    fnames = loader(prefix, n_iters, channel, filters, target)
    fig = ca.plot_cluster_size_evolution(
        path=SAMPLING_PATH, fnames=fnames, n_rows=2, n_cols=3
    )
    gr = Graph()
    gr.save_figure(
        figure=fig,
        path=EXPORT_PATH,
        fname=f"test_cluster_size_{prefix}_{channel}_{target}_{filters}",
        html=True
    )
