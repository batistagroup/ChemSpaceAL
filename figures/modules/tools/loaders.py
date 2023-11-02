from typing import List, Callable
import pandas as pd

def prepare_loader(base_path: str):
    def _load_smiles(fname: str):
        return pd.read_csv(base_path + fname + ".csv")["smiles"].to_list()
    return _load_smiles


def setup_fname_generator(presuffix: str) -> Callable:
    def prepare_scored_fnames(
        prefix: str,
        n_iters: int,
        channel: str,
        filters: str,
        target: str,
    ) -> List[str]:
        assert filters in {"ADMET", "ADMET+FGs"}
        if "model2" in prefix:
            fnames = [
                f"{prefix[:6]}_baseline_{target.upper()}_{presuffix}_{filters}",
                *(
                    f"{prefix}_al{i}_{channel}_{target.upper()}_{presuffix}_{filters}"
                    for i in range(1, n_iters + 1)
                ),
            ]
        elif "model7" in prefix:
            fnames = [
                f"{prefix[:6]}_baseline_{target.upper()}_{presuffix}_{filters}",
                *(
                    f"{prefix}_{channel}_al{i}_{target.upper()}_{presuffix}_{filters}"
                    for i in range(1, n_iters + 1)
                ),
            ]
        return fnames

    return prepare_scored_fnames
