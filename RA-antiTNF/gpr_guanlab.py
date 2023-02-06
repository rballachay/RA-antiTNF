import copy
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from caching import cachewrapper
from datareader import create_clinical_data, get_dosage_data
from globals import TEST_INDEX
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import resample

RANDOM_STATE = 0
META_PATH = "metadata/guanlab/"
SNP_BY_DRUG = {
    "adalumimab": "Guanlab_adalimumab_snps",
    "etanercept": "Guanlab_etanercept_snps",
    "infliximab": "Guanlab_infliximab_snps",
}
PARAMS = {"alpha": 4e-1, "kernel": RBF(), "random_state": RANDOM_STATE}
SAMPLES = 10


def main(params=PARAMS):
    gpr = GaussianProcessRegressor(**params)
    raw_results = run_cross_validation(gpr)
    stat_results = produce_statistics(raw_results)
    f = sns.violinplot(data=stat_results, x="drug_model", y="pearson")
    fig = f.get_figure()
    fig.savefig("results.png")


def produce_statistics(raw_results: pd.DataFrame):
    def calc_r(group):
        drugmodel = group["drug_model"].values[0]
        sample = group["samples"].values[0]
        pearson = np.corrcoef(group["y_real"], group["y_hat"])[0, 1]
        return pd.DataFrame(
            [{"drug_model": drugmodel, "sample": sample, "pearson": pearson}]
        )

    stat_results = (
        raw_results.groupby(["drug_model", "samples"], as_index=False)
        .apply(calc_r)
        .reset_index(drop=True)
    )
    return stat_results


@cachewrapper("cache", "guanlab_exp_results")
def run_cross_validation(gpr):
    results = []
    for drug, X_train, X_test, y_train, y_test in data_generator():
        for sample in range(SAMPLES):
            # copy model
            gpr_i = copy.deepcopy(gpr)
            _X_train, _y_train = resample(X_train, y_train)
            _X_test, _y_test = resample(X_test, y_test)

            # fit + predict model
            gpr_i.fit(_X_train, _y_train)
            y_hat = gpr_i.predict(_X_test)

            # prepare results
            drug_list = [drug] * len(y_hat)
            samples = [sample + 1] * len(y_hat)
            _results_i = list(zip(drug_list, y_hat, _y_test, samples))
            results.extend(_results_i)
    results_df = pd.DataFrame(
        results, columns=["drug_model", "y_hat", "y_real", "samples"]
    )
    return results_df


def data_generator(path=Path(META_PATH), snp_dict=SNP_BY_DRUG):
    # either create data or read from cache
    data = create_drug_df(path, snp_dict)
    engineered = feature_engineering(data)
    for drug in engineered["TNF-drug"].unique():
        _df = data[data["TNF-drug"] == drug]
        _df = _df.dropna(axis=1)
        _train = _df.iloc[:TEST_INDEX]
        _test = _df.iloc[TEST_INDEX:]
        yield drug, *split_dfs(_train, _test)


def feature_engineering(data):
    # convert drugs to one-hot encoding
    onehot_drugs = pd.get_dummies(data["Drug"])
    data = data.drop(["Drug"], axis=1)
    data = pd.concat([data, onehot_drugs], axis=1)
    return data


def split_dfs(train_df, test_df):
    y_cols = ["Response.deltaDAS"]
    not_X_cols = [
        "Response.deltaDAS",
        "Cohort",
        "Batch",
        "Response.EULAR",
        "Response.NonResp",
        "ID",
        "TNF-drug",
    ]

    X_train = train_df.drop(not_X_cols, axis=1).values
    X_test = test_df.drop(not_X_cols, axis=1).values
    y_train = train_df[y_cols].values.flatten()
    y_test = test_df[y_cols].values.flatten()
    return X_train, X_test, y_train, y_test


@cachewrapper("cache", "guanlab_fulldata")
def create_drug_df(path, snp_dict):
    """returns a dictionary with three keys, one for each drug. each of
    these drugs then have two subkeys, one for training and one for testing.
    """

    clinical_data = create_clinical_data()
    snp_map = _read_snps(path, snp_dict)

    dfs = []
    for drug, releveant_snps in snp_map.items():
        # this gives us a dataframe like [chromosome,marker,....patient1,...patient2706]
        #  we want columns to be titled by marker id, drop other columns
        relevant_dosage = get_dosage_data(releveant_snps)
        clinical_data["TNF-drug"] = drug
        drug_features = pd.concat([clinical_data, relevant_dosage], axis=1)
        dfs.append(drug_features)

    return pd.concat(dfs)


def _read_snps(path, snp_dict) -> dict:
    """read in the text files from guanlab that describe which snps belong"""
    out_dict = {}
    for drug, file in snp_dict.items():
        snppath = path / file
        with open(snppath, "r") as txtfile:
            lines = txtfile.read()
            snps = list(filter(lambda x: x, lines.split("\t")))

        out_dict[drug] = snps
    return out_dict


if __name__ == "__main__":
    main()
