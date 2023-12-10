import modules.secret
from modules.Graph import Graph
import modules.ModelMemorization as mm
import os

GENERATIONS_PATH = modules.secret.PRODUCTION_RUNS_PATH + "2. Generation/smiles/"
EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "model_quality", "")

# # ------------------------------
# # Vaidity, Uniqueness, Novelty
# fig = mm.create_unique_valid_novel_figure(generation_path=GENERATIONS_PATH)
# graph = Graph()
# graph.update_parameters(
#     dict(
#         width=900,
#         height=300,
#         t_margin=40,
#         b_margin=40,
#         yrange=[74, 100],
#     )
# )
# graph.style_figure(fig)
# graph.save_figure(
#     figure=fig, path=EXPORT_PATH, fname="validity_uniqueness_novelty", svg=True
# )
# # ------------------------------

# # ------------------------------
# # Memorization Table
# metrics = [
#     lambda i: f"% repetitions (from AL{i} training set)",
#     lambda i: f"% repetitions (from scored from round {i})",
#     lambda i: f"% fraction of AL{i} training set in generated",
#     lambda i: f"% fraction of scored from round {i} in generated",
# ]
# format_ch = (
#     lambda ch: "random"
#     if ch == "random"
#     else "uniform"
#     if ch == "diffusion"
#     else f"{ch}-based"
# )
# # for channel in ["random", "diffusion", "linear", "softdiv", "softsub"]:
# #     fnames = mm.prepare_nofilters_fnames(channel)
# #     fig_fname = f"memorization_table_with_{format_ch(channel)}"

# configs = [
#     ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "softsub"),
#     ("model2_1iep", "1IEP", "ADMET+FGs", "admetfg_softsub"),
#     ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "random"),
#     ("model7_1iep_admet", "1IEP", "ADMET", "softsub"),
#     ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "randomwsampling"),
#     ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "diffusion"),
# ]
# n_iters = 5
# for prefix, target, filters, channel in configs:
#     fnames = mm.prepare_generated_fnames(prefix, n_iters, channel, filters, target)
#     fig_fname = f"{prefix}_{channel}_{target}_{filters}"
#     print(f"Preparing {fig_fname}")
#     print(fnames)
#     traces = mm.prepare_quality_table(
#         generations_path=GENERATIONS_PATH,
#         fnames=fnames,
#         metric_functions=metrics,
#         n_rows=2,
#         n_cols=2,
#     )
#     fig = mm.create_memorization_heatmap_figure(
#         traces,
#         n_rows=2,
#         n_cols=2,
#         subplot_titles=[
#             "<b>Percentage of generated (x) molecules<br>repeated from AL training set (y)</b>",
#             "<b>Percentage of generated (x) molecules<br>repeated from scored molecules (y)</b>",
#             "<b>Percentage of molecules in AL training<br>set (y) repeated in generated (x)</b>",
#             "<b>Percentage of scored molecules (y)<br>repeated in generated (x)</b>",
#         ],
#     )
#     graph = Graph()
#     graph.update_parameters(
#         dict(
#             width=700,
#             height=350,
#             # title=f"<b>Model memorization with {format_ch(channel)} selection</b>",
#             t_margin=40,
#             b_margin=0,
#         )
#     )
#     graph.style_figure(fig)
#     fig.update_layout(
#         coloraxis1=dict(
#             colorscale="Reds",
#             cmax=2.2,
#             cmin=0,
#             colorbar=dict(x=1.01, y=1.08, len=0.5, yanchor="top", title="Percentage"),
#         ),
#         coloraxis2=dict(
#             colorscale="Reds",
#             cmax=100,
#             cmin=0,
#             colorbar=dict(x=1.01, y=0.45, len=0.5, yanchor="top", title="Percentage"),
#         ),
#     )
#     graph.save_figure(
#         figure=fig,
#         path=EXPORT_PATH,
#         fname=fig_fname,
#     )
# # ------------------------------

configs = [
    ("model7_hnh_admet", "HNH", "ADMET", "softsub"),
    ("model7_hnh_admetfg", "HNH", "ADMET+FGs", "softsub"),
    # ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "softsub"),
    # ("model2_1iep", "1IEP", "ADMET+FGs", "admetfg_softsub"),
    # ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "random"),
    # ("model7_1iep_admet", "1IEP", "ADMET", "softsub"),
    # ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "randomwsampling"),
    # ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "diffusion"),
]
n_iters = 5
import plotly.graph_objects as go

fig = go.Figure()
colors = ["#080708", "#2DD881", "#A40E4C", "#FF8552", "#A288E3", "#009DDC"]
i = 0
for prefix, target, filters, channel in configs:
    fnames = mm.prepare_generated_fnames(prefix, n_iters, channel, filters, target)
    fig_fname = prefix.split("_")[0] + "_" + prefix.split("_")[-1] + "_" + channel

    # for channel in ["random", "diffusion", "linear", "softdiv", "softsub"]:
    #     fnames = mm.prepare_nofilters_fnames(channel)

    num_generated = mm.prepare_num_generations(GENERATIONS_PATH, fnames)
    fig.add_trace(
        go.Scatter(
            x=[f"Iter. {i}" for i in range(6)],
            y=num_generated,
            mode="markers+lines",
            marker=dict(color="black"),
            line_color=colors[i % len(colors)],
            line_width=4,
            name=fig_fname,
            showlegend=True,
        )
    )
    i += 1

# fig.show()
fig.update_layout(legend=dict(x=0.7, y=0.95))  # , orientation="h"
gr = Graph()
gr.update_parameters(
    dict(
        width=900,
        height=400,
        l_margin=0,
        b_margin=0,
        t_margin=40,
        r_margin=0,
        # title=f"<b>Number of </b>",
    )
)
gr.style_figure(fig)
# Update the legend position)
# gr.save_figure(fig, EXPORT_PATH, "num_generations_for_hnh")
gr.save_figure(fig, EXPORT_PATH, "num_generations_for_hnh_wfilters")
# gr.save_figure(fig, EXPORT_PATH, "num_generations_for_1iep")
# fig.write_image(os.path.join(EXPORT_PATH, "num_generations.png"))
