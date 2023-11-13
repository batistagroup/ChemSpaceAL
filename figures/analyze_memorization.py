import modules.secret
from modules.Graph import Graph
import modules.ModelMemorization as mm
import os

GENERATIONS_PATH = modules.secret.PRODUCTION_RUNS_PATH + "2. Generation/smiles/"
EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "model_quality", "")

# ------------------------------
# Vaidity, Uniqueness, Novelty
fig = mm.create_unique_valid_novel_figure(generation_path=GENERATIONS_PATH)
graph = Graph()
graph.update_parameters(
    dict(
        width=900,
        height=300,
        t_margin=40,
        b_margin=40,
        yrange=[74, 100],
    )
)
graph.style_figure(fig)
graph.save_figure(
    figure=fig, path=EXPORT_PATH, fname="validity_uniqueness_novelty", svg=True
)
# ------------------------------

# ------------------------------
# Memorization Table
metrics = [
    lambda i: f"% repetitions (from AL{i} training set)",
    lambda i: f"% repetitions (from scored from round {i})",
    lambda i: f"% fraction of AL{i} training set in generated",
    lambda i: f"% fraction of scored from round {i} in generated",
]
format_ch = (
    lambda ch: "random"
    if ch == "random"
    else "uniform"
    if ch == "diffusion"
    else f"{ch}-based"
)
for channel in ["random", "diffusion", "linear", "softdiv", "softsub"]:
    traces = mm.prepare_quality_table(
        generations_path=GENERATIONS_PATH,
        metric_functions=metrics,
        channel=channel,
        n_rows=2,
        n_cols=2,
    )
    fig = mm.create_memorization_heatmap_figure(
        traces,
        n_rows=2,
        n_cols=2,
        subplot_titles=[
            "<b>Fraction of generated (x) molecules<br>repeated from AL training set (y)</b>",
            "<b>Fraction of generated (x) molecules<br>repeated from scored molecules (y)</b>",
            "<b>Fraction of molecules in AL training<br>set (y) repeated in generated (x)</b>",
            "<b>Fraction of scored molecules (y)<br>repeated in generated (x)</b>",
        ],
    )
    graph = Graph()
    graph.update_parameters(
        dict(
            width=700,
            height=400,
            title=f"<b>Model memorization with {format_ch(channel)} selection</b>",
            t_margin=80,
        )
    )
    graph.style_figure(fig)
    fig.update_layout(
            coloraxis1=dict(
                colorscale="Reds",
                cmax=2,
                cmin=0,
                colorbar=dict(
                    x=1.01, y=1.08, len=0.5, yanchor="top", title="Percentage"
                ),
            ),
            coloraxis2=dict(
                colorscale="Reds",
                cmax=100,
                cmin=0,
                colorbar=dict(
                    x=1.01, y=0.45, len=0.5, yanchor="top", title="Percentage"
                ),
            ),
        )
    graph.save_figure(figure=fig, path=EXPORT_PATH, fname=f"memorization_table_with_{format_ch(channel)}")
# ------------------------------
