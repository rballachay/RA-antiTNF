import copy
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from caching import cachewrapper
from datareader import create_clinical_data, get_dosage_data
from globals import TEST_INDEX
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils import resample
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

RANDOM_STATE = 0
META_PATH = "metadata/SJU_plink/"
SNP_BY_DRUG = {
    "all": "plink_LD_SNPs",
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
MODEL_DICT = {
    Lasso: Lasso(alpha=0.01),
    SVR: SVR(kernel="linear", C=0.1, epsilon=0.01),
    RandomForestRegressor: RandomForestRegressor(n_estimators=300),
}
PLOT_DICT = {
    "pearson": "pearson_results.png",
    "scatterplot": "scatterplot_results.png",
    "auroc": "auroc_results.png",
    "accuracy": "accuracy_results.png",
    "roccurve": "roccurve_results.png",
}

THRESHOLD = 2.5
SAMPLES = 10

RESULTS_PATH = Path("results/SJU_models/")


def main(
    models=(SVR, RandomForestRegressor, Lasso),
    hyper_dict=HYPER_DICT,
    random_state=RANDOM_STATE,
    results_path=RESULTS_PATH,
    plot_dict=PLOT_DICT,
):
    # create our model with all default parameters
    raw_results = run_cross_validation(models, hyper_dict, random_state)
    stat_results = produce_statistics(raw_results)
    roc_results = prep_roc_curve(raw_results)

    # assert results path exists
    results_path.mkdir(parents=True, exist_ok=True)

    # visualize results + plot
    plots = visualize_results(raw_results, stat_results, roc_results)

    # save to output directory
    for title, fig in plots.items():
        fig.savefig(str(results_path / plot_dict[title]))


def visualize_results(raw_results, stat_results, roc_results):
    plots = {}
    sns.set_theme()

    plt.figure()
    f = sns.violinplot(data=stat_results, x="model", y="pearson")
    plots["pearson"] = f.get_figure()

    plt.figure()
    f = sns.scatterplot(data=raw_results, x="y_real", y="y_hat", hue="model")
    f.set_xlim((-2, 7))
    f.set_ylim((-2, 7))
    plots["scatterplot"] = f.get_figure()

    plt.figure()
    f = sns.violinplot(data=stat_results, x="model", y="auroc")
    plots["auroc"] = f.get_figure()

    plt.figure()
    f = sns.violinplot(data=stat_results, x="model", y="accuracy", hue="model")
    plots["accuracy"] = f.get_figure()

    plt.figure(dpi=200)
    f = sns.lineplot(data=roc_results, x="fpr", y="tpr", hue="model")
    roc_auc = list(roc_results.groupby(["model"])[["roc_auc"]].mean().itertuples())
    labels = list(map(lambda x: f"{x[0]}, {x[1]:.2f}", roc_auc))
    plt.legend(title="Mean AUROC", loc="upper left", labels=labels)
    plt.plot(np.arange(0, 2), np.arange(0, 2), "k--")
    plots["roccurve"] = f.get_figure()

    return plots


def prep_roc_curve(raw_results):
    def _myagg(group, bins):
        groupslice = group.iloc[[0]].copy()
        to_append = []
        for bin_i in bins:
            if int(bin_i) not in set(group["cutter"]):
                newslice = groupslice.copy()
                newslice["fpr"] = bin_i
                newslice["tpr"] = None
                newslice["roc_auc"] = None
                to_append.append(newslice)

        if to_append:
            group = pd.concat([group, *to_append], axis=0)
        final = group.groupby(["cutter"], as_index=False).agg(
            {
                "tpr": np.nanmean,
                "roc_auc": np.nanmean,
                "model": "first",
                "samples": "first",
            }
        )
        final["fpr"] = final["cutter"]
        return final

    roc_results = {"fpr": [], "tpr": [], "roc_auc": [], "model": [], "samples": []}
    for title, group in raw_results.groupby(["model", "samples"]):
        cat_hat = group[["prob_hat"]].values
        cat_real = group[["cat_real"]].values
        fpr, tpr, _ = roc_curve(cat_real, cat_hat)
        roc_auc = auc(fpr, tpr)
        roc_results["fpr"].extend(fpr.tolist())
        roc_results["tpr"].extend(tpr.tolist())
        roc_results["roc_auc"].extend([roc_auc] * len(fpr))
        roc_results["model"].extend([title[0]] * len(fpr))
        roc_results["samples"].extend([title[1]] * len(fpr))

    roc_df = pd.DataFrame(roc_results)
    bins = np.arange(0 - 0.02, 1 + 0.02, 0.02)
    means = np.convolve(bins, [0.5, 0.5], "valid")
    roc_df["cutter"] = pd.cut(roc_df["fpr"], bins, labels=means)
    roc_new = (
        roc_df.groupby(["samples", "model"], as_index=False)
        .apply(partial(_myagg, bins=means))
        .reset_index(drop=True)
    )
    return roc_new


def choose_best_model(X, y, model, hyper, random_state):
    return MODEL_DICT[model]
    opt = BayesSearchCV(
        model(),
        hyper,
        n_iter=10,
        random_state=random_state,
        verbose=10,
        n_points=2,
        cv=5,
    )
    opt.fit(X, y)

    # return best estimator, aka best model over all hyperparameters
    return opt.best_estimator_


def produce_statistics(raw_results: pd.DataFrame):
    def calc_r(group):
        drugmodel = group["drug_model"].values[0]
        model = group["model"].values[0]
        sample = group["samples"].values[0]
        pearson = pearsonr(group["y_real"], group["y_hat"])[0]
        auroc = roc_auc_score(group["cat_real"].values, group["prob_hat"].values)
        accuracy = accuracy_score(group["cat_real"].values, group["cat_hat"].values)
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "drug_model": drugmodel,
                    "samples": sample,
                    "pearson": pearson,
                    "auroc": auroc,
                    "accuracy": accuracy,
                }
            ]
        )

    X = raw_results["y_hat"]
    raw_results["prob_hat"] = 1 - (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    raw_results["cat_hat"] = raw_results["y_hat"] > THRESHOLD
    stat_results = (
        raw_results.groupby(["drug_model", "samples", "model"], as_index=False)
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
                print(f"Sample {sample}, model {best_model}")
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
                model_list = [model.__name__] * len(y_hat)

                _results_i = list(
                    zip(
                        model_list,
                        drug_list,
                        y_hat,
                        _y_test,
                        _cat_test,
                        samples,
                    )
                )
                results.extend(_results_i)
    results_df = pd.DataFrame(
        results,
        columns=["model", "drug_model", "y_hat", "y_real", "cat_real", "samples"],
    )
    return results_df


def data_generator(path=Path(META_PATH), snp_dict=SNP_BY_DRUG, test_index=TEST_INDEX):
    # either create data or read from cache
    data = create_drug_df(path, snp_dict)
    engineered = feature_engineering(data)

    for drug in snp_dict:
        _df = engineered[engineered["TNF-drug"] == drug].reset_index(drop=True)
        # _df = _df.dropna(axis=1)

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
