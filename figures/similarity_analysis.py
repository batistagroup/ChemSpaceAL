from modules.Graph import Graph
import modules.secret
import modules.ChemicalSimilarity as chem_sim
import os

EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "chemical_similarity", "")


def analyze_abl_binders():
    for fp_type in [
        "ECFP",
        "FCFP",
        "MACCS",
        "RDKit FP",
        "Atom-Pair FP",
        "Topological Torsion FP",
        "Avalon FP",
    ]:
        fig = chem_sim.create_abl_inhibitors_heatmap_all_simtypes(
            fp_type=fp_type, sim_types=chem_sim.ALL_SIMILARITY_METRICS
        )
        gr = Graph()
        gr.update_parameters(
            dict(
                width=800,
                height=500,
                annotation_size=24,
                title=f"<b>Fingerprint Type: {fp_type}</b>",
                t_margin=80,
            )
        )
        gr.style_figure(fig)
        # fig.show()
        gr.save_figure(fig, path=EXPORT_PATH, fname=f"abl_inhibitors_{fp_type}")


if __name__ == "__main__":
    fnames = chem_sim.prepare_scored_fnames(
        prefix="model2_hnh",
        n_iters=5,
        channel="admetfg_softsub",
        filters="ADMET+FGs",
        target="HNH",
    )
    print(fnames)
    pass
