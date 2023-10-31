import numpy as np
from modules.Graph import Graph
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

step_to_time = {
    "Pretraining": 23 * 30,
    "Generation": 30,
    "Calculating<br>Descriptors": 30,
    "Clustering<br>& Sampling": 8,
    "Docking": 75 / 60 * 1000,
    "Scoring": 6,
    "Active<br>Learning": 0.2 * 10,
}

x = [f"<b>{key}</b>" for key in step_to_time.keys()]
y = list(step_to_time.values())

# Define where the y-axis will break
cut_interval = [40, 500]

genml = "#5a189a"
desc = "#072ac8"
dock = "#a4133c"
colors = [genml, genml, desc, desc, dock, dock, genml]
bar = go.Bar(x=x, y=y, name="Steps", marker_color=colors)


fig = make_subplots(
    rows=2,
    cols=1,
    vertical_spacing=0.05,
    shared_xaxes=True,
)

bar.showlegend = False
fig.append_trace(bar, row=1, col=1)
fig.append_trace(bar, row=2, col=1)


graph = Graph()
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_yaxes(range=[0, cut_interval[0]], row=2, col=1)
graph.update_parameters(dict(width=900, tick_font_size=18))
graph.style_figure(fig)
fig.add_annotation(
    text="Wall Time (min)",
    xref="paper",
    yref="paper",
    x=-0.10,
    y=0.4,
    showarrow=False,
    textangle=-90,  # Rotates the text
    font=dict(
        size=20, family="Helvetica", color="#333333"
    ),  # Adjust the size of the text
)
fig.update_yaxes(range=[cut_interval[1], max(y) * 1.1], row=1, col=1, title="")
EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "")
fig.write_image(EXPORT_PATH + "pipeline_walltime.svg", scale=1.0)
fig.write_image(EXPORT_PATH + "pipeline_walltime.jpg", scale=2.0)
