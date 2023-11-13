from modules.Graph import Graph
import modules.TrainingCurves as tc
import modules.secret
import os
import numpy as np
import wandb

EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "training_curves", "")
WANDB_LOGS = modules.secret.PRODUCTION_RUNS_PATH + "1. Pretraining/wandb_logs/"

# ------------------ Pretraining ------------------
# traces = tc.prepare_pretraining_traces(WANDB_LOGS)
# fig = tc.plot_scatter2d_wsubplots(
#     traces=traces,
#     subplot_titles=[
#         "<b>Training loss</b>",
#         "<b>Validation loss</b>",
#         "<b>Learning rate</b>",
#     ],
#     n_rows=1,
#     n_cols=3,
# )

# graph = Graph()
# graph.update_parameters(
#     dict(
#         width=1000,
#         height=300,
#         xdticks=5,
#         xrange=[0, 30],
#         xaxis_title="Epochs",
#         l_margin=0,
#         b_margin=60,
#         axis_title_size=16,
#         annotation_size=18,
#     )
# )
# direct_yaxes = [
#     dict(
#         title="Loss",
#         title_font=dict(size=18, color="#333", family="Helvetica"),
#         title_standoff=0,
#         range=[0.48, 0.75],
#         row=1,
#         col=1,
#     ),
#     dict(range=[0.48, 0.75], row=1, col=2),
# ]
# for config in direct_yaxes:
#     fig.update_yaxes(**config)
# graph.style_figure(fig)
# fig.update_layout(showlegend=False)
# graph.save_figure(
#     fig,
#     path=EXPORT_PATH,
#     fname="pretraining=train_val_lr",
#     svg=True,
# )
# ------------------ Pretraining ------------------


# ------------------ AL losses ------------------

annotations = [
    dict(
        x=30 + 10 * i + 5,
        y=0.5,
        text=f"<b>Active Learning {i+1}</b>",
        showarrow=False,
        font=dict(size=12, color="DarkSlateGrey"),
    )
    for i in range(5)
]
shapes = [
    dict(
        type="line",
        x0=30 + 10 * i,
        y0=0.18,
        x1=30 + 10 * i,
        y1=0.50,
        line=dict(color="black", width=3, dash="dash"),
    )
    for i in range(1, 6)
]

# traces = tc.prepare_al_traces()
# fig = tc.plot_al_losses(
#     traces=traces,
#     shapes=shapes,
#     annotations=annotations,
# )
# graph = Graph()
# graph.update_parameters(dict(
#         width=1000,
#         height=500,
#         xaxis_title="Epochs",
#         yaxis_title="Loss",
#         yrange=[0.18, 0.52],
#         axis_title_size=24,
#         title_size=32,
#         title="<b>Training step losses throughout Active Learning cycles</b>",
#     ))
# graph.style_figure(fig)
# fig.update_layout(legend=dict(
#         x=0.85,
#         y=0.7,
#         xanchor="left",
#         yanchor="middle",
#         font=dict(size=18),
#     ))
# graph.save_figure(
#     fig,
#     path=EXPORT_PATH,
#     fname="AL_losses",
#     svg=True,
# )
# ------------------ AL losses ------------------
