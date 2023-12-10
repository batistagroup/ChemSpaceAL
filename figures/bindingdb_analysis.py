import modules.BindingDBAnalysis as bdb
import modules.secret

DATASETS_PATH = (
    modules.secret.PRODUCTION_RUNS_PATH + "1. Pretraining/datasets/original_files/"
)

# df = bdb.load_kd_containing_from_bindingdb(DATASETS_PATH)
good_binders = bdb.find_good_binders(DATASETS_PATH, cutoff=100)
print(len(good_binders))
# bdb.analyze_repeated_targets(good_binders)

o = bdb.find_smiles_with_n_targets(good_binders, 180)
print(o)
# for cut in [1, 10, 100, 1000, 10_000, 100_000]:
#     fig=bdb.plot_cumulative_histogram_of_smiles_distribution(
#         bdb.find_good_binders(DATASETS_PATH, cutoff=cut), suffix=f"_cutoff{cut}"
#     )
#     fig.show()
