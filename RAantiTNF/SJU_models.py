import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
import seaborn as sns
from caching import cachewrapper
from datareader import create_clinical_data, get_dosage_data
from globals import TEST_INDEX
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.utils import resample
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from utils import NonResponseRA

RANDOM_STATE = 0
META_PATH = "metadata/SJU_plink/"
SNP_BY_DRUG = {
    "adalumimab": "plink_LD_SNPs",
    "etanercept": "plink_LD_SNPs",
    "infliximab": "plink_LD_SNPs",
}
HYPER_DICT = {
    Lasso: {"alpha": Real(1e-3, 1, prior="log-uniform")},
    SVR: {
        "kernel": Categorical(["linear", "rbf", "poly"]),
        "C": Real(1e-3, 10, prior="log-uniform"),
        "epsilon": Real(1e-3, 10, prior="log-uniform"),
    },
    RandomForestRegressor: {"n_estimators": Integer(10, 1000, prior="log-uniform")},
}


THRESHOLD = 2.5
SAMPLES = 10


def main(
    models=(SVR, RandomForestRegressor, Lasso),
    hyper_dict=HYPER_DICT,
    random_state=RANDOM_STATE,
):
    # create our model with all default parameters
    raw_results = run_cross_validation(models, hyper_dict, random_state)
    stat_results = produce_statistics(raw_results)

    sns.set_theme()
    f = sns.violinplot(data=stat_results, x="drug_model", y="pearson")
    fig = f.get_figure()
    fig.savefig("pearson_results.png")

    plt.clf()
    f = sns.scatterplot(data=raw_results, x="y_real", y="y_hat", hue="drug_model")
    f.set_xlim((-2, 7))
    f.set_ylim((-2, 7))
    fig = f.get_figure()
    fig.savefig("scatterplot_results.png")

    plt.clf()
    f = sns.violinplot(data=stat_results, x="drug_model", y="auroc")
    fig = f.get_figure()
    fig.savefig("auroc_results.png")

    plt.clf()
    f = sns.violinplot(data=stat_results, x="drug_model", y="accuracy")
    fig = f.get_figure()
    fig.savefig("accuracy_results.png")

    plt.clf()
    a = raw_results[["cat_hat"]].values
    b = 1 - raw_results[["cat_hat"]].values
    cat = np.concatenate([a, b], axis=1)
    f = skplt.metrics.plot_roc_curve(raw_results["cat_real"].values, cat)
    fig = f.get_figure()
    fig.savefig("roccurve_results.png")


def choose_best_model(X, y, model, hyper, random_state):
    opt = BayesSearchCV(
        model(),
        hyper,
        n_iter=2,
        random_state=random_state,
        verbose=10,
        n_points=1,
        cv=5,
    )
    opt.fit(X, y)

    # return best estimator, aka best model over all hyperparameters
    return opt.best_estimator_


def produce_statistics(raw_results: pd.DataFrame):
    def calc_r(group):
        drugmodel = group["drug_model"].values[0]
        sample = group["samples"].values[0]
        pearson = pearsonr(group["y_real"], group["y_hat"])[0]
        auroc = roc_auc_score(group["cat_real"].values, group["cat_hat"].values)
        accuracy = accuracy_score(group["cat_real"].values, group["cat_hat"].values)
        return pd.DataFrame(
            [
                {
                    "drug_model": drugmodel,
                    "sample": sample,
                    "pearson": pearson,
                    "auroc": auroc,
                    "accuracy": accuracy,
                }
            ]
        )

    stat_results = (
        raw_results.groupby(["drug_model", "samples"], as_index=False)
        .apply(calc_r)
        .reset_index(drop=True)
    )
    return stat_results


@cachewrapper("cache", "SJU_plink_results")
def run_cross_validation(models, hyper_dict, random_state):
    results = []
    for (
        drug,
        X_train,
        X_test,
        y_train,
        y_test,
        das_test,
        cat_test,
        yscaler,
    ) in data_generator():
        for model in models:
            best_model = choose_best_model(
                X_train, y_train, model, hyper_dict[model], random_state
            )

            for sample in range(SAMPLES):
                # copy model
                best_model_i = copy.deepcopy(best_model)
                _X_train, _y_train = resample(X_train, y_train)
                _X_test, _y_test, _das_test, _cat_test = resample(
                    X_test, y_test, das_test, cat_test
                )

                # fit + predict model
                best_model_i.fit(_X_train, _y_train)

                # predict + expand dims
                preds = best_model_i.predict(_X_test)

                if preds.ndim == 1:
                    preds = preds[:, None]

                y_hat = yscaler.inverse_transform(preds).flatten()
                _y_test = yscaler.inverse_transform(_y_test).flatten()

                # prepare results
                drug_list = [drug] * len(y_hat)
                samples = [sample + 1] * len(y_hat)

                # predicted response
                # cat_hat = list(map(NonResponseRA(), _das_test, y_hat))
                cat_hat = y_hat < THRESHOLD

                assert len(cat_hat) == len(_cat_test)

                _results_i = list(
                    zip(
                        model.__name__,
                        drug_list,
                        y_hat,
                        _y_test,
                        cat_hat,
                        _cat_test,
                        samples,
                    )
                )
                results.extend(_results_i)
    results_df = pd.DataFrame(
        results,
        columns=["model", "drug," "y_hat", "y_real", "cat_hat", "cat_real", "samples"],
    )
    return results_df


def data_generator(path=Path(META_PATH), snp_dict=SNP_BY_DRUG, test_index=TEST_INDEX):
    # either create data or read from cache
    data = create_drug_df(path, snp_dict)
    engineered = feature_engineering(data)
    for drug in engineered["TNF-drug"].unique():
        _df = data[data["TNF-drug"] == drug].reset_index(drop=True)
        _df = _df.dropna(axis=1)

        X, y, das, cat = split_xy(_df)

        X, y, yscaler = scale_xy(X, y)

        X_train, X_test, y_train, y_test, das_test, cat_test = test_train_split(
            X, y, das, cat, test_index
        )

        yield drug, X_train, X_test, y_train, y_test, das_test, cat_test, yscaler


def scale_xy(X, y):
    yscaler = StandardScaler()
    xscaler = StandardScaler()
    X = xscaler.fit_transform(X)
    y = yscaler.fit_transform(y)
    return X, y, yscaler


def test_train_split(X, y, das, cat, test_index):
    return (
        X[:test_index],
        X[test_index:],
        y[:test_index],
        y[test_index:],
        das[test_index:],
        cat[test_index:],
    )


def feature_engineering(data):
    snp_cols = list(filter(lambda x: x.startswith("rs"), data.columns))
    to_impute = ["baselineDAS", "Age", "Gender", "Mtx"] + snp_cols
    knn = KNNImputer(n_neighbors=5)
    data[to_impute] = knn.fit_transform(data[to_impute])
    # convert drugs to one-hot encoding
    # onehot_drugs = pd.get_dummies(data["Drug"])
    data = data.drop(["Drug"], axis=1)
    # data = pd.concat([data, onehot_drugs], axis=1)
    return data


def split_xy(df):
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
    X = df.drop(not_X_cols, axis=1).values
    y = df[y_cols].values
    das = df["baselineDAS"].values
    cat = df["Response.NonResp"].values
    return X, y, das, cat


@cachewrapper("cache", "SJU_plink_fulldata")
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
            snps = [i for x in snps for i in x.split("\n")]
            snps = list(filter(lambda x: x, snps))

        out_dict[drug] = snps
    return out_dict


if __name__ == "__main__":
    main()
