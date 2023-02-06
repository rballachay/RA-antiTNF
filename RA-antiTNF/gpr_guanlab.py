import copy
from pathlib import Path

import pandas as pd
from caching import cachewrapper
from datareader import create_clinical_data, get_dosage_data
from globals import TEST_INDEX
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

META_PATH = "metadata/guanlab/"
SNP_BY_DRUG = {
    "adalumimab": "Guanlab_adalimumab_snps",
    "etanercept": "Guanlab_etanercept_snps",
    "infliximab": "Guanlab_infliximab_snps",
}
PARAMS = {"alpha": 4, "kernel": RBF(100000.0), "random_state": 0}


def main(params=PARAMS):
    gpr = GaussianProcessRegressor(**params)

    results = []
    for drug, X_train, X_test, y_train, y_test in data_generator():
        gpr_i = copy.deepcopy(gpr)
        gpr_i.fit(X_train, y_train)
        y_hat = gpr_i.predict(X_test)

        drug_list = [drug] * len(y_hat)
        _results_i = list(zip(drug_list, y_hat, y_test))
        results.extend(_results_i)
    results_df = pd.DataFrame(results, colummns=["drug_model", "y_hat", "y_real"])
    print(results_df)


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
