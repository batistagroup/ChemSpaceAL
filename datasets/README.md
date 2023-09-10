# Pretraining the model

The generative model was pretrained on combined dataset formed from a combination of [ChemBL 33](https://chembl.gitbook.io/chembl-interface-documentation/downloads), [GuacaMol v1](https://github.com/BenevolentAI/guacamol/), [MOSES](https://github.com/molecularsets/moses), and [BindingDB (08-2023)](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp). The dataset was processed to exclude all SMILES strings containing more than 133 tokens and which contain tokens that occurr less than 1000 times in the dataset. The combined dataset contains 5 539 765 unique and valid SMILES strings, which are split into:

- [Training Partition (5 262 776 entries; 95%)](datasets/combined_processed_freq1000_block133_train.csv.gz)
- [Validation Partition (276 989 entries; 5%)](datasets/combined_processed_freq1000_block133_val.csv.gz)
