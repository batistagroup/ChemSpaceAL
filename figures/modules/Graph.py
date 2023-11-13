import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List

# import plotly.io as pio
# pio.kaleido.scope.mathjax = None


class Graph:
    def __init__(self):
        self.title_size = 20
        self.axis_title_size = 14
        self.tick_font_size = 12
        self.annotation_size = 14
        self.text_color = "#333333"
        self.annotation_color = "#333333"
        self.background = "white"
        self.grid_color = "#e2e2e2"
        self.line_color = "#000000"
        self.font_family = "Helvetica"
        self.title_font_family = "Helvetica"
        self.showlegend = True
        self.width = 600
        self.height = 400
        self.title = ""
        self.xaxis_title = ""
        self.yaxis_title = ""
        self.show_xgrid = False
        self.show_ygrid = False
        self.show_yzero = False
        self.show_xzero = False
        self.yrange = None
        self.xrange = None
        self.xgrid_width = 1
        self.ygrid_width = 1
        self.xline_width = 1
        self.yline_width = 1
        self.xmirror = False
        self.ymirror = False
        self.xshowticklabels = True
        self.yshowticklabels = True
        self.xdticks = None
        self.ydticks = None
        self.title_ycoord = 0.95
        self.title_xcoord = 0.5
        self.yaxis_standoff = 0
        self.b_margin = 50
        self.t_margin = 50
        self.l_margin = 50
        self.r_margin = 50
        self.xtick_width = 1
        self.ytick_width = 1
        self.xtick_len = 5
        self.ytick_len = 5
        self.tick_position = "outside"
        self.xtick_vals = None
        self.ytick_vals = None

    def create_folders(self, base_path: str, folder_lists: List[List[str]]):
        if not folder_lists:
            return
        for parent_folder in folder_lists[0]:
            new_path = os.path.join(base_path, parent_folder)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            self.create_folders(new_path, folder_lists[1:])

    def update_parameters(self, params):
        for key, val in params.items():
            setattr(self, key, val)

    def style_figure(self, figure, force_annotations=True):
        layout_dict = dict(
            showlegend=self.showlegend,
            margin=dict(
                t=self.t_margin, b=self.b_margin, l=self.l_margin, r=self.r_margin
            ),
            plot_bgcolor=self.background,
            paper_bgcolor=self.background,
            title=dict(
                text=self.title,
                y=self.title_ycoord,
                x=self.title_xcoord,
                xanchor="center",
                yanchor="top",
                font=dict(
                    size=self.title_size,
                    color=self.text_color,
                    family=self.title_font_family,
                ),
            ),
            height=self.height,  # Set fixed size ratio 3:4
            width=self.width,
            font=dict(
                family=self.font_family,
                size=self.tick_font_size,
                color=self.text_color,
            ),
            legend=dict(
                font=dict(
                    family=self.font_family,
                    size=self.tick_font_size,
                    color=self.text_color,
                ),
            ),
        )
        xaxis_dict = dict(
            side="bottom",
            title=self.xaxis_title,
            title_font=dict(
                size=self.axis_title_size,
                color=self.text_color,
                family=self.font_family,
            ),
            tickfont=dict(
                size=self.tick_font_size,
                color=self.text_color,
                family=self.font_family,
            ),
            showgrid=self.show_xgrid,
            zeroline=self.show_xzero,
            gridwidth=self.xgrid_width,
            gridcolor=self.grid_color,
            linecolor=self.line_color,  # make x axis line visible
            linewidth=self.xline_width,
            mirror=self.xmirror,
            showticklabels=self.xshowticklabels,
            ticks=self.tick_position,
            tickwidth=self.xtick_width,
            ticklen=self.xtick_len,
        )

        yaxis_dict = dict(
            title=self.yaxis_title,
            title_standoff=self.yaxis_standoff,
            title_font=dict(
                size=self.axis_title_size,
                color=self.text_color,
                family=self.font_family,
            ),
            tickfont=dict(
                size=self.tick_font_size,
                color=self.text_color,
                family=self.font_family,
            ),
            showgrid=self.show_ygrid,
            zeroline=self.show_yzero,
            gridwidth=self.ygrid_width,
            gridcolor=self.grid_color,
            linecolor=self.line_color,  # make y axis line visible
            linewidth=self.yline_width,
            mirror=self.ymirror,
            showticklabels=self.yshowticklabels,
            ticks=self.tick_position,
            tickwidth=self.ytick_width,
            ticklen=self.ytick_len,
        )
        if self.xrange is not None:
            xaxis_dict["range"] = self.xrange
        if self.yrange is not None:
            yaxis_dict["range"] = self.yrange
        if self.xdticks is not None:
            xaxis_dict["dtick"] = self.xdticks
        if self.ydticks is not None:
            yaxis_dict["dtick"] = self.ydticks

        if self.xtick_vals is not None:
            xaxis_dict["tickvals"] = self.xtick_vals
        if self.ytick_vals is not None:
            yaxis_dict["tickvals"] = self.ytick_vals

        figure.update_layout(layout_dict)
        figure.update_xaxes(xaxis_dict)
        figure.update_yaxes(yaxis_dict)
        if force_annotations:
            for annotation in figure["layout"]["annotations"]:
                annotation["font"] = dict(
                    size=self.annotation_size,
                    family=self.font_family,
                    color=self.annotation_color,
                )
        return figure

    def save_figure(
        self,
        figure,
        path,
        fname,
        jpg=True,
        svg=False,
        pdf=False,
        html=False,
        png=False,
        eps=False,
        scale: float = 8.0,
    ):
        if html:
            figure.write_html(f"{path}html/{fname}.html", include_plotlyjs="cdn")
        if jpg:
            figure.write_image(f"{path}jpg/{fname}.jpg", scale=scale)
        if png:
            figure.write_image(f"{path}jpg/{fname}.png", scale=scale)
        if svg:
            figure.write_image(f"{path}svg/{fname}.svg")
        if pdf:
            figure.write_image(f"{path}pdf/{fname}.pdf")
        if eps:
            figure.write_image(f"{path}pdf/{fname}.eps")


if __name__ == "__main__":
    base_path = os.path.join(os.getcwd(), "figures")
    gr = Graph()
    gr.create_folders(
        base_path=base_path,
        folder_lists=[
            ["exports"],
            [
                "ligand_distribution",
                "cluster_distribution",
                "admet_satisfaction",
                "chemical_similarity",
                "distribution_analysis",
                "conversion_schemes",
                "training_curves",
                "model_quality",
                # "pca",
                # "pdbbind",
                # "generation_distribution",
            ],
            ["html", "svg", "jpg"],
        ],
    )
