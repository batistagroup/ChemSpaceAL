import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional
import pandas as pd
import numpy as np
import wandb


def load_losses_manually(logs_path: str):
    loader = lambda fname: pd.read_csv(logs_path + fname + ".csv")
    train = loader("epoch_train_loss")["model7_baseline - epoch_train_loss"].to_numpy()
    train_step = loader("step_train_loss")[
        "model7_baseline - step_train_loss"
    ].to_numpy()
    valid = loader("epoch_valid_loss")["model7_baseline - epoch_valid_loss"].to_numpy()
    lrs = loader("learning_rate")["model7_baseline - learning_rate"].to_numpy()
    return train, train_step, valid, lrs


def prepare_scatter_trace(xVals, yVals, extra_params: Optional[dict] = None):
    if extra_params is None:
        extra_params = dict(mode="lines", showlegend=False)

    return go.Scatter(
        x=xVals,
        y=yVals,
        visible=True,
        **extra_params,
    )


def prepare_pretraining_traces(logs_path: str):
    train_epoch, train_step, valid, lrs = load_losses_manually(logs_path)
    lr_steps_per_epoch = len(lrs) / len(train_epoch)
    lrs_x = np.arange(len(lrs)) / lr_steps_per_epoch
    coloring = dict(line=dict(color="#023e8a", width=4))
    return [
        prepare_scatter_trace(
            np.arange(1, len(train_epoch) + 1), train_epoch, coloring
        ),
        prepare_scatter_trace(np.arange(1, len(valid) + 1), valid, coloring),
        prepare_scatter_trace(lrs_x, lrs, coloring),
    ]


def find_runid_mapping():
    api = wandb.Api()
    runs = api.runs("Production Pipeline")
    return {run.name: run.id for run in runs}


def load_losses_from_wandb(wandb_api, group: str, project: str, run_id: str, key: str):
    df = wandb_api.run(f"{group}/{project}/{run_id}").history()
    return df[key].to_numpy()


def prepare_loss_trace(prefix: str, channel: str, n_iters: int):
    wandb_mapping = find_runid_mapping()
    fnames = [f"{prefix}_{channel}_al{i}" for i in range(1, n_iters + 1)]
    wandb_ids = [wandb_mapping[fname] for fname in fnames]
    loss_arrays = [
        load_losses_from_wandb(
            wandb.Api(),
            "generative_ml",
            "Production Pipeline",
            wandb_id,
            "step_train_loss",
        )
        for wandb_id in wandb_ids
    ]
    loss = np.hstack(loss_arrays)
    n_epochs = 10

    shifted = []
    for i, loss_array in enumerate(loss_arrays):
        n_steps = len(loss_array)
        steps_per_epoch = n_steps / n_epochs
        step_array = np.arange(n_steps) / steps_per_epoch
        step_array += i * n_epochs + 30
        shifted.append(step_array)
    xVals = np.hstack(shifted)
    # print(epoch_steps)
    return xVals, loss


def prepare_al_traces():
    traces = []
    colors = ["#252323", "#43AA8B", "#4895ef", "#560bad", "#f72585"]
    for i, channel in enumerate(["random", "diffusion", "linear", "softdiv", "softsub"]):
        prefix = "model7" if channel == "random" else "model7_mix100"
        xVals, loss = prepare_loss_trace(prefix, channel, 5)
        trace_params = dict(
            mode="lines", name=channel, showlegend=True, line=dict(color=colors[i], width=2)
        )
        traces.append(prepare_scatter_trace(xVals, loss, trace_params))
    return traces


def plot_scatter2d_wsubplots(
    traces,
    subplot_titles,
    n_rows: int,
    n_cols: int,
):
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.03,
    )
    for i, trace in enumerate(traces):
        row = i // n_cols + 1
        col = i % n_cols + 1
        fig.add_trace(trace, row=row, col=col)

    for col, text in [(1, "a"), (2, "b"), (3, "c")]:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=-0.02,
            y=1.25,
            xanchor="right",
            yanchor="top",
            text=f"<b>[{text.upper()}]</b>",
            font=dict(size=24, family="Helvetica"),
            showarrow=False,
            row=1,
            col=col,
        )

    return fig


def plot_al_losses(
    traces,
    shapes,
    annotations,
):
    fig = go.Figure()
    components = list(range(1, 121))
    for trace in traces:
        fig.add_trace(trace)
    for shape in shapes:
        fig.add_shape(shape)
    for annotation in annotations:
        fig.add_annotation(annotation)
    return fig
