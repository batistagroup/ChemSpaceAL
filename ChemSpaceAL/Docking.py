try:
    import biopandas
except:
    message = """pip install pyg==0.7.1 pyyaml==6.0 scipy==1.7.3 networkx==2.6.3 biopython==1.79 rdkit-pypi==2022.03.5 e3nn==0.5.0 spyrmsd==0.5.2 pandas==1.5.3 biopandas==0.4.1"""
    raise ImportError(
        f"biopandas is not found, you can install it by running\n" + message
    )

import torch

print(torch.__version__)

try:
    import torch_geometric
except ModuleNotFoundError:
    message = f"""pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git  --quiet"""
    raise ImportError(
        f"torch_geometric is not found, you can install it by running\n" + message
    )

from rdkit import Chem
import shutil
import os
import pandas as pd
from tqdm import tqdm

base_path = os.getcwd()
dd_path = os.path.join(base_path, "DiffDock")

if not os.path.exists(dd_path):
    message = """Please don't forget to install DiffDock by:
    1. Creating a new temprorary `DiffDock` folder
    2. Running `git clone https://github.com/gcorso/DiffDock.git`
    3. Enter `DiffDock` and run `git checkout a6c5275` (you can update this for a more up to date code)"""
    raise ImportError(message)

# clone ESM repository
esm_path = os.path.join(dd_path, "esm")
if not os.path.exists(esm_path):
    message = """Please don't forget to download ESM for protein folding. To do that:
    1. Go to your temporary `DiffDock` folder
    2. Run `git clone https://github.com/facebookresearch/esm`
    3. Enter `esm` and run `git checkout ca8a710`
    4. Run `sudo pip install -e .`"""
    raise ImportError(message)


def get_top_poses(ligands_csv: str, protein_pdb_path: str, save_pose_path: str):
    data = pd.read_csv(ligands_csv)

    os.environ["HOME"] = "esm/model_weights"
    os.environ["PYTHONPATH"] = f'{os.environ.get("PYTHONPATH", "")}:{esm_path}'
    pbar = tqdm(range(len(data)), total=len(data))
    results_path = os.path.join(dd_path, "results")
    for i in pbar:
        smiles = data["smiles"][i]

        with open(os.path.join(base_path, "input_protein_ligand.csv"), "w") as out:
            out.write("protein_path,ligand\n")
            out.write(f"{protein_pdb_path},{smiles}\n")

        # Clear out old results if running multiple times
        shutil.rmtree(results_path, ignore_errors=True)

        # ESM Embedding Preparation
        os.chdir(dd_path)
        !python /content/DiffDock/datasets/esm_embedding_preparation.py --protein_ligand_csv /content/input_protein_ligand.csv --out_file /content/DiffDock/data/prepared_for_esm.fasta

        # ESM Extraction
        !python /content/DiffDock/esm/scripts/extract.py esm2_t33_650M_UR50D /content/DiffDock/data/prepared_for_esm.fasta /content/DiffDock/data/esm2_output --repr_layers 33 --include per_tok --truncation_seq_length 30000

        # Inference
        !python /content/DiffDock/inference.py --protein_ligand_csv /content/input_protein_ligand.csv --out_dir /content/DiffDock/results/user_predictions_small --inference_steps 20 --samples_per_complex 10 --batch_size 6

        # Move results
        for root, dirs, files in os.walk(
            os.path.join(dd_path, "results", "user_predictions_small")
        ):
            for file in files:
                if file.startswith("rank1_confidence"):
                    shutil.move(
                        os.path.join(root, file),
                        os.path.join(save_pose_path, f"complex{i}.sdf"),
                    )
