from ChemSpaceAL.Configuration import Config
import os

def test_config():
    base_path = os.getcwd() + "/PaperRuns/"
    config = Config(
        base_path=base_path,
        cycle_prefix="model0",
        al_iteration=0,
        cycle_suffix="ch1",
        training_fname="moses_train.csv.gz",
        validation_fname="moses_test.csv.gz",
        slice_data=1_000,
        verbose=True,
    )
    assert True
