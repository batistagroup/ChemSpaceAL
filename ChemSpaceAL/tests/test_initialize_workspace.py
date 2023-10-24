import os
import tempfile
from unittest.mock import patch
from ChemSpaceAL import InitializeWorkspace as iw


def test_create_default_folders():
    # Using a temporary directory for testing
    with tempfile.TemporaryDirectory() as tempdir:
        iw.create_folders(base_path=tempdir)

        # Check if the main folders are created
        assert os.path.exists(os.path.join(tempdir, "1_Pretraining"))
        assert os.path.exists(os.path.join(tempdir, "2_Generation"))

        # Check if subfolders are created
        assert os.path.exists(os.path.join(tempdir, "1_Pretraining", "datasets"))
        # ... (add more assertions for other subfolders)
