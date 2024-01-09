# Available files

The generative model was pretrained on combined dataset formed from a combination of [ChemBL 33](https://chembl.gitbook.io/chembl-interface-documentation/downloads), [GuacaMol v1](https://github.com/BenevolentAI/guacamol/), [MOSES](https://github.com/molecularsets/moses), and [BindingDB (08-2023)](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp). The dataset was processed to exclude all SMILES strings containing more than 133 tokens and which contain tokens that occurr less than 1000 times in the dataset. The combined dataset contains 5 539 765 unique and valid SMILES strings, which are split into:

- [Training Partition (5 262 776 entries; 95%)](https://files.ischemist.com/ChemSpaceAL/publication_runs/1_Pretraining/datasets/combined_train.csv.gz)
- [Validation Partition (276 989 entries; 5%)](https://files.ischemist.com/ChemSpaceAL/publication_runs/1_Pretraining/datasets/combined_valid.csv.gz)

If you wish to use our pretrained model, you can download the model weights and dataset descriptors (these are internal parameters required for generation). Note that if you use [our Jupyter Notebook](../ChemSpaceAL.ipynb), it has a special cell which will download these files and put them into appropriate folders for you!

- [Pretrained model weights](https://files.ischemist.com/ChemSpaceAL/publication_runs/1_Pretraining/datasets_descriptors/combined_train.yaml)
- [Pretrained model descriptors](https://files.ischemist.com/ChemSpaceAL/publication_runs/1_Pretraining/model_weights/model7_al0_ch1.pt)

You can also download the PCA weights fitted on the combined dataset:

- [scaler,pca tuple stored as a pickle](https://files.ischemist.com/ChemSpaceAL/publication_runs/3_Sampling/pca_weights/scaler_pca_combined_n120.pkl)
