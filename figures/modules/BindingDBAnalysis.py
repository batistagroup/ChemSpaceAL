from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import plotly.graph_objects as go

def load_kd_containing_from_bindingdb(datasets_path: str):
    columns = ["Ligand SMILES", "Ki (nM)", "IC50 (nM)", "Kd (nM)", "Target Name"]
    db = pd.read_csv(
        datasets_path + "BindingDB_all.tsv",
        sep="\t",
        usecols=columns,
    )
    print(f"Loaded {len(db)=} entries")
    db.dropna(subset=["Kd (nM)"], inplace=True)
    print(f"Removed entries with missing Kd, {len(db)=} entries left")
    # db.drop_duplicates(inplace=True)
    # print(f"Removed duplicates, {len(db)=} entries left")

    # Loaded len(db)=2781616 entries
    # Removed entries with missing Kd, len(db)=105832 entries left
    db.to_csv(datasets_path + "bdb_with_kd_values.csv")
    return db[["Ligand SMILES", "Target Name", "Kd (nM)"]]


def find_good_binders(datasets_path: str, cutoff: float = 100) -> pd.DataFrame:
    def parse_kd(value):
        if isinstance(value, (float, int)):
            return value
        elif isinstance(value, str):
            try:
                if ">" in value:
                    return float(value[1:])
                elif "<" in value:
                    return (
                        float(value[1:]) - 1
                    )  # or other logic if you want to handle the '<' differently
                return float(value)
            except Exception as e:
                print(f"Error parsing {value=}: {e}")
                return float("inf")  # or another value that will be filtered out
        else:
            raise TypeError(f"{value=} is of type {type(value)=}")

    df = pd.read_csv(datasets_path + "bdb_with_kd_values.csv")
    # Apply the custom function to the "Kd (nM)" column
    df["Kd (nM)"] = df["Kd (nM)"].apply(parse_kd)
    good_binders = df[df["Kd (nM)"] <= cutoff]
    return good_binders


def analyze_repeated_targets(good_binders: pd.DataFrame) -> None:
    # Group by "Ligand SMILES" and count the unique "Target Name" values for each group
    target_counts = good_binders.groupby("Ligand SMILES")["Target Name"].nunique()

    # Count how many SMILES strings are associated with each number of unique targets
    repetition_counts = target_counts.value_counts().sort_index()

    print("Number of unique targets for each SMILES string:")
    print(repetition_counts)


def find_smiles_with_n_targets(good_binders, n):
    target_counts = good_binders.groupby("Ligand SMILES")["Target Name"].nunique()
    smiles_with_n_targets = target_counts[target_counts == n].index

    # Get the rows from the original DataFrame that match the SMILES strings with n targets
    rows_with_n_targets = good_binders[
        good_binders["Ligand SMILES"].isin(smiles_with_n_targets)
    ]

    # Group by "Ligand SMILES" again and get the list of unique target names for each
    target_names_by_smiles = rows_with_n_targets.groupby("Ligand SMILES")[
        "Target Name"
    ].unique()

    return target_names_by_smiles


def calculate_descriptors_for_bindingdb(kd_cutoff, target_freq):
    df = pd.read_csv(
        f"{BASE_PATH}analysis/no_rare_targets_Kd<{kd_cutoff}_freq>{target_freq}.csv"
    )
    smiles = set(df["Ligand SMILES"].values)
    calculate_descriptors(
        smiles, "mix", f"bindingdb_mix_Kd<{kd_cutoff}_freq>{target_freq}"
    )
    calculate_descriptors(
        smiles, "mqn", f"bindingdb_mqn_Kd<{kd_cutoff}_freq>{target_freq}"
    )


def plot_cumulative_histogram_of_smiles_distribution(good_binders, suffix=""):
    # graph = Graph(f"{PLOT_PATH}bindingdb_analysis/")
    # graph.create_folders([["html", "svg", "jpg"]])
    smiles_counts_per_target = good_binders.groupby("Target Name")[
        "Ligand SMILES"
    ].nunique()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=smiles_counts_per_target,
            cumulative=dict(enabled=True, direction="decreasing"),
            xbins=dict(start=0, end=smiles_counts_per_target.max(), size=1),
            marker=dict(opacity=0.7),
        )
    )
    thresholds = [50, 100, 200, 300, 500]
    counts = {
        threshold: sum(smiles_counts_per_target >= threshold)
        for threshold in thresholds
    }

    for i, (threshold, count) in enumerate(counts.items()):
        fig.add_annotation(
            x=0.95,  # Relative horizontal position in the figure
            y=0.95 - 0.05 * i,  # Relative vertical position, you can adjust as needed
            xref="paper",  # Position is relative to the figure
            yref="paper",  # Position is relative to the figure
            text=f"{count} targets have {threshold} or more SMILES",
            showarrow=False,
            align="right",
        )
    # graph.update_parameters(
    #     dict(
    #         title="Cumulative Histogram of SMILES Distribution Across Targets",
    #         xaxis_title="Number of SMILES Strings or Fewer",
    #         yaxis_title="Number of Targets",
    #         showlegend=False,
    #     )
    # )
    # graph.style_figure(fig)
    # graph.save_figure(
    #     fig, f"{PLOT_PATH}bindingdb_analysis/", "smiles_distribution" + suffix
    # )
    return fig


# def remove_rare_targets(kd_cutoff=100, target_freq=200):
#     good_binders = find_good_binders(load_bindingdb(), kd_cutoff)
#     print(f"{len(good_binders)=}")
#     good_binders.drop_duplicates(subset=["Ligand SMILES"], inplace=True)
#     print(f"{len(good_binders)=} after dropping duplicates")
#     smiles_counts_per_target = good_binders.groupby("Target Name")[
#         "Ligand SMILES"
#     ].nunique()
#     targets_to_keep = smiles_counts_per_target[
#         smiles_counts_per_target >= target_freq
#     ].index
#     no_rare_targets = good_binders[good_binders["Target Name"].isin(targets_to_keep)]
#     print(f"Post filtering we have {len(no_rare_targets)=}")
#     # no_rare_targets.drop_duplicates(subset=["Ligand SMILES"], inplace=True)
#     # print(f"Post dropping {len(no_rare_targets)=}")
#     no_rare_targets.to_csv(
#         f"{BASE_PATH}analysis/no_rare_targets_Kd<{kd_cutoff}_freq>{target_freq}.csv",
#         index=False,
#     )


# def find_unique_targets(kd_cutoff, target_freq):
#     df = pd.read_csv(
#         f"{BASE_PATH}analysis/no_rare_targets_Kd<{kd_cutoff}_freq>{target_freq}.csv"
#     )
#     unique_targets = df["Target Name"].unique()
#     colors = [
#         "#e6194B",
#         "#3cb44b",
#         "#ffe119",
#         "#4363d8",
#         "#f58231",
#         "#911eb4",
#         "#46f0f0",
#         "#f032e6",
#         "#42d4f4",
#         "#bfef45",
#         "#fabebe",
#         "#469990",
#         "#e6beff",
#         "#9A6324",
#         "#800000",
#         "#aaffc3",
#     ]
#     target_to_color = {target: color for target, color in zip(unique_targets, colors)}
#     # safe this dictionary to a yaml file
#     with open(f"{BASE_PATH}analysis/target_to_color_16.yaml", "w") as file:
#         yaml.dump(target_to_color, file)
#     return target_to_color
