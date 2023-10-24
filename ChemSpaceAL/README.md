# ChemSpaceAL internals

## TO-DO

- [ ] fix the types in `Sampling.py` for `cluster_to_samples: Dict[int, Any] = {}`
- [ ] cite the DiffDock repo and cite it's license
- [ ] get rid of the `try/except` loop in `scoring/score_protein_ligand_pose.py`
- [ ] we need to possibly change AL construction procedure s.t. if `n_replicate` is False, more molecules are sampled from the training set until `training_size` is satisfied
