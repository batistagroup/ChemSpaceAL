import ChemSpaceAL.InitializeWorkspace as iw
import os

def test_config():
    base_path = os.getcwd() + "/PaperRuns/"
    assert isinstance(iw.FOLDER_STRUCTURE, dict)
    assert True
