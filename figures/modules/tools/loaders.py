from typing import List, Callable, Optional, Union
import pandas as pd
import pprint

pp = pprint.PrettyPrinter(indent=2, width=100)

Number = Union[float, int]


def prepare_loader(
    base_path: str, return_full: bool = False, extension: str = "csv"
) -> Callable:
    def _load_smiles(fname: str) -> List[str]:
        assert extension in {"csv"}, f"Extension {extension} not supported."
        return pd.read_csv(base_path + fname + "." + extension)["smiles"].to_list()

    def _load_df(fname: str) -> pd.DataFrame:
        match extension:
            case "csv":
                return pd.read_csv(base_path + fname + ".csv")
            case "pkl":
                return pd.read_pickle(base_path + fname + ".pkl")

    if return_full:
        return _load_df
    else:
        return _load_smiles


def setup_scored_fname_generator(filters: bool) -> Callable:
    def prepare_scored_fnames_wfilters(
        prefix: str,
        n_iters: int,
        channel: str,
        filters: str,
        target: str,
    ) -> List[str]:
        assert filters in {"ADMET", "ADMET+FGs"}
        presuffix = "mix_k100"
        if "model2" in prefix:
            fnames = [
                f"{prefix[:6]}_baseline_{target.upper()}_{presuffix}_{filters}",
                *(
                    f"{prefix}_al{i}_{channel}_{target.upper()}_{presuffix}_{filters}"
                    for i in range(1, n_iters + 1)
                ),
            ]
        elif "model7" in prefix:
            if "random" in channel:
                fnames = [
                    f"{prefix[:6]}_baseline_{target.upper()}_{channel}_{filters}",
                    *(
                        f"{prefix}_{channel}_al{i}_{target.upper()}_{channel}_{filters}"
                        for i in range(1, n_iters + 1)
                    ),
                ]
            else:
                fnames = [
                    f"{prefix[:6]}_baseline_{target.upper()}_{presuffix}_{filters}",
                    *(
                        f"{prefix}_{channel}_al{i}_{target.upper()}_{presuffix}_{filters}"
                        for i in range(1, n_iters + 1)
                    ),
                ]
            # fnames = [
            #     f"{prefix[:6]}_baseline_{target.upper()}_{presuffix}_{filters}",
            #     *(
            #         f"{prefix}_{channel}_al{i}_{target.upper()}_{presuffix}_{filters}"
            #         for i in range(1, n_iters + 1)
            #     ),
            # ]
        return fnames

    def prepare_scored_fnames_nofilters(
        prefix: str,
        n_iters: int,
        channel: str,
        filters: str,
        target: str,
        descriptors_type: str = "mix",
        n_clusters: Optional[int] = 100,
    ) -> List[str]:
        if n_clusters is not None:
            # original runs
            fnames = [
                f"{prefix}_baseline_{descriptors_type}_k{n_clusters}",
                *(
                    f"{prefix}_{descriptors_type}{n_clusters}_{channel}_al{i}_{descriptors_type}_k{n_clusters}"
                    for i in range(1, n_iters + 1)
                ),
            ]
        else:
            # random runs
            fnames = [
                f"{prefix}_baseline_{channel}",
                *(f"{prefix}_{channel}_al{i}_{channel}" for i in range(1, n_iters + 1)),
            ]
        return fnames

    if filters:
        return prepare_scored_fnames_wfilters
    else:
        return prepare_scored_fnames_nofilters


def setup_generations_fname_generator(presuffix: str, filters: bool) -> Callable:
    def prepare_generations_fnames_wfilters(
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

    def prepare_generations_fnames_nofilters(
        prefix: str,
        n_iters: int,
        channel: str,
        filters: str,
        target: str,
        descriptors_type: str = "mix",
        n_clusters: Optional[int] = 100,
    ) -> List[str]:
        if n_clusters is not None:
            # original runs
            fnames = [
                f"{prefix}_baseline_{presuffix}",
                *(
                    f"{prefix}_{descriptors_type}{n_clusters}_{channel}_al{i}_{presuffix}"
                    for i in range(1, n_iters + 1)
                ),
            ]
        else:
            # random runs
            fnames = [
                f"{prefix}_baseline_{presuffix}",
                *(
                    f"{prefix}_{channel}_al{i}_{presuffix}"
                    for i in range(1, n_iters + 1)
                ),
            ]
        return fnames

    if filters:
        return prepare_generations_fnames_wfilters
    else:
        return prepare_generations_fnames_nofilters


def setup_altrains_fname_generator(filters: bool, no_score: bool = False) -> Callable:
    if no_score:
        suffix = "_noscore"
    else:
        suffix = ""

    def prepare_altrains_fnames_wfilters(
        prefix: str,
        n_iters: int,
        channel: str,
        filters: str,
        target: str,
        threshold: Number,
        conversion_scheme: str,
    ) -> List[str]:
        assert filters in {"ADMET", "ADMET+FGs"}
        presuffix = "mix_k100"
        if "model2" in prefix:
            fnames = [
                f"{prefix[:6]}_baseline_{target.upper()}_{presuffix}_threshold{threshold}_{conversion_scheme}_{filters}{suffix}",
                *(
                    f"{prefix}_al{i}_{channel}_{target.upper()}_{presuffix}_threshold{threshold}_{conversion_scheme}_{filters}{suffix}"
                    for i in range(1, n_iters + 1)
                ),
            ]
        elif "model7" in prefix:
            fnames = [
                f"{prefix[:6]}_baseline_{target.upper()}_{presuffix}_threshold{threshold}_{conversion_scheme}_{filters}{suffix}",
                *(
                    f"{prefix}_{channel}_al{i}_{target.upper()}_{presuffix}_threshold{threshold}_{conversion_scheme}_{filters}{suffix}"
                    for i in range(1, n_iters + 1)
                ),
            ]
        return fnames

    def prepare_altrains_fnames_nofilters(
        prefix: str,
        n_iters: int,
        channel: str,
        filters: str,
        target: str,
        threshold: Number,
        conversion_scheme: str,
        descriptors_type: str = "mix",
        n_clusters: Optional[int] = 100,
    ) -> List[str]:
        if n_clusters is not None:
            # original runs
            fnames = [
                f"{prefix}_baseline_{descriptors_type}_k{n_clusters}_threshold{threshold}_{conversion_scheme}{suffix}",
                *(
                    f"{prefix}_{descriptors_type}{n_clusters}_{channel}_al{i}_{descriptors_type}_k{n_clusters}_threshold{threshold}_{conversion_scheme}{suffix}"
                    for i in range(1, n_iters + 1)
                ),
            ]
        else:
            # random runs
            fnames = [
                f"{prefix}_baseline_{channel}_threshold{threshold}",
                *(
                    f"{prefix}_{channel}_al{i}_{channel}_threshold{threshold}"
                    for i in range(1, n_iters + 1)
                ),
            ]
        return fnames

    if filters:
        return prepare_altrains_fnames_wfilters
    else:
        return prepare_altrains_fnames_nofilters


if __name__ == "__main__":
    filter_configs = [
        ("model7_hnh_admet", "HNH", "ADMET", "softsub"),
        # ("model7_hnh_admetfg", "HNH", "ADMET+FGs", "softsub"),
        # ("model2_hnh", "HNH", "ADMET+FGs", "admetfg_softsub"),
        # ("model7_1iep_admet", "1IEP", "ADMET", "softsub"),
        # ("model7_1iep_admetfg", "1IEP", "ADMET+FGs", "softsub"),
        # ("model2_1iep", "1IEP", "ADMET+FGs", "admetfg_softsub"),
    ]
    filter_run_generator = setup_scored_fname_generator(filters=True)
    filter_run_generator = setup_altrains_fname_generator(filters=True)
    n_iters = 5
    threshold = 11
    conversion_scheme = "softmax_sub"
    for prefix, target, filters, channel in filter_configs:
        fnames = filter_run_generator(
            prefix, n_iters, channel, filters, target, threshold, conversion_scheme
        )
        pp.pprint(fnames)

    original_configs = [
        ("mix", 100, 5, "softsub"),
        # ("mix", 100, 5, "softdiv"),
        # ("mix", 100, 5, "linear"),
        # ("mix", 100, 5, "diffusion"),
        # ("mix", 10, 5, "softsub"),
        # ("mqn", 100, 5, "softsub"),
        # ("mqn", 100, 5, "diffusion"),
        # ("mqn", 10, 5, "softsub"),
        ("", None, 5, "random"),
    ]
    run_generator = setup_scored_fname_generator(filters=False)
    run_generator = setup_generations_fname_generator(
        "temp1.0_completions", filters=False
    )
    run_generator = setup_altrains_fname_generator(filters=False)
    prefix = "model7"
    threshold = 11
    conversion_scheme = "softmax_sub"
    for descriptors_type, n_clusters, n_iters, channel in original_configs:
        fnames = run_generator(
            prefix,
            descriptors_type,
            n_iters,
            channel,
            threshold,
            conversion_scheme,
            n_clusters,
        )
        pp.pprint(fnames)
