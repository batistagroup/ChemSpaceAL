import os
import modules.secret
import modules.DatasetAnalysis as da
import numpy as np
import pickle
import modules.ChemicalSimilarity as chem_sim

PRETRAINING_PATH = modules.secret.PRODUCTION_RUNS_PATH + "1. Pretraining/"
EXPORT_PATH = os.path.join(os.getcwd(), "figures", "exports", "datasets_analysis", "")

# combined_smiles = da.load_raw_dataset(
#     PRETRAINING_PATH + "datasets/converted/", "combined"
# )
training_smiles = da.load_training_smiles(PRETRAINING_PATH + "datasets/", "combined", sample=None)
from tqdm import tqdm
import rdkit.Chem as Chem

canon_set = set()
for smile in tqdm(training_smiles):
    canon_set.add(Chem.MolToSmiles(Chem.MolFromSmiles(smile)))
pickle.dump(canon_set, open(EXPORT_PATH + "canon_set_freq1000_block133.pkl", "wb"))
# canon_set = pickle.load(open(EXPORT_PATH + "canon_set.pkl", "rb"))
# print(
#     f"Number of unique smiles: {len(canon_set)}, total smiles: {len(combined_smiles)}"
# )
# train_set = set(combined_smiles)
print(chem_sim.ABL_BINDERS)
for name, abl_smile in chem_sim.ABL_BINDERS.items():
    present = abl_smile in canon_set
    present_canon = Chem.MolToSmiles(Chem.MolFromSmiles(abl_smile)) in canon_set
    print(f"{name}: is in training set? {present}, {present_canon}")

# combined_smiles = np.random.choice(combined_smiles, 100_000, replace=False)
# token_to_freq, block_sizes = da.analyze_vocabulary_and_length(combined_smiles)
# pickle.dump((token_to_freq, block_sizes), open(EXPORT_PATH + "combined_token_to_freq_and_block_sizes_sam100k.pkl", "wb"))

# token_to_freq, block_sizes = pickle.load(
#     open(EXPORT_PATH + "combined_token_to_freq_and_block_sizes.pkl", "rb")
# )

from modules.Graph import Graph

# graph = Graph()
# fig = da.plot_block_sizes_and_vocabulary_frequency(token_to_freq, block_sizes)
# graph.update_parameters(
#     dict(
#         width=1200,
#         height=600,
#         annotation_size=18,
#         axis_title_size=18,
#         tick_font_size=15,
#     )
# )
# graph.style_figure(fig, force_annotations=True)
# fig.layout.annotations[0].update(font_size=28)
# fig.layout.annotations[1].update(font_size=28)
# fig.update_xaxes(title_text="Block Size", row=1, col=1)
# fig.update_xaxes(title_text="Token", row=1, col=2)
# fig.update_yaxes(title_text="Frequency", row=1, col=1)
# fig.update_yaxes(title_text="Frequency", row=1, col=2)
# graph.save_figure(figure=fig, path=EXPORT_PATH, fname="block_size_and_token_frequency")

# load_mw = lambda fname: pickle.load(
#     open(PRETRAINING_PATH + f"descriptors/{fname}_mol_wts.pkl", "rb")
# )

# combined_mw = load_mw("combined")
# guacamol_mw = load_mw("guac")
# moses_mw = load_mw("moses")

# fig = da.plot_molecular_weight_distribution(moses_mw, guacamol_mw, combined_mw)
# graph = Graph()
# graph.update_parameters(
#     dict(
#         width=1200,
#         height=400,
#         annotation_size=28,
#         axis_title_size=18,
#         tick_font_size=15,
#     )
# )
# graph.style_figure(fig, force_annotations=True)
# fig.update_xaxes(range=[190, 410], title_text="Molecular Weight", row=1, col=1)
# fig.update_yaxes(title_text="Frequency", row=1, col=1)
# fig.update_xaxes(title_text="Molecular Weight", row=1, col=2)
# fig.update_xaxes(title_text="Molecular Weight", row=1, col=3)

# # Show the figure
# # fig.show()
# fig.write_image(EXPORT_PATH + "molecular_weight_distribution.jpg", scale=4.0)
# graph.save_figure(figure=fig, path=EXPORT_PATH, fname="molecular_weight_distribution")
