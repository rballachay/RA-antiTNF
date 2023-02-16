import copy
import re
from functools import partial
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import resample
from skopt import BayesSearchCV
from skopt.space import Real
from utils import NonResponseRA

RANDOM_STATE = 0
META_PATH = "metadata/guanlab/"
SNP_BY_DRUG = {
    "adalumimab": "Guanlab_adalimumab_snps",
    "etanercept": "Guanlab_etanercept_snps",
    "infliximab": "Guanlab_infliximab_snps",
}
# kernels need to be defined as categorical because we cannot use the bayesian on kernels
# instead of model
_KERNELS = [
    RBF,
]  # RBF, ExpSineSquared]
_LENGTH_SPACE = np.logspace(-1, 4, num=6)
KERNEL_SPACE = [k(l) for k in _KERNELS for l in _LENGTH_SPACE]

# define global values to optimize
HYPER_SPACE = {"alpha": Real(1e-6, 1e2, prior="log-uniform")}
SAMPLES = 10
THRESHOLD = 2.5

PLOT_DICT = {
    "pearson": "pearson_results.png",
    "scatterplot": "scatterplot_results.png",
    "auroc": "auroc_results.png",
    "accuracy": "accuracy_results.png",
    "roccurve": "roccurve_results.png",
}
RESULTS_PATH = Path("results/GPR_guanlab/")


def main(
    model=GaussianProcessRegressor,
    hyper_space=HYPER_SPACE,
    kernels=KERNEL_SPACE,
    random_state=RANDOM_STATE,
    results_path=RESULTS_PATH,
    plot_dict=PLOT_DICT,
):
    # create our model with all default parameters
    raw_results = run_cross_validation(model, hyper_space, kernels, random_state)
    stat_results = produce_statistics(raw_results)

    roc_results = prep_roc_curve(raw_results)

    # visualize results + plot
    plots = visualize_results(raw_results, stat_results, roc_results)

    # save to output directory
    for title, fig in plots.items():
        fig.savefig(str(results_path / plot_dict[title]))


def visualize_results(raw_results, stat_results, roc_results):
    plots = {}
    sns.set_theme()

    plt.figure()
    f = sns.violinplot(data=stat_results, x="feature", y="pearson", hue="drug_model")
    plots["pearson"] = f.get_figure()

    plt.figure()
    f = sns.scatterplot(data=raw_results, x="y_real", y="y_hat", hue="drug_model")
    f.set_xlim((-2, 7))
    f.set_ylim((-2, 7))
    plots["scatterplot"] = f.get_figure()

    plt.figure()
    f = sns.violinplot(data=stat_results, x="feature", y="auroc", hue="drug_model")
    plots["auroc"] = f.get_figure()

    plt.figure()
    f = sns.violinplot(data=stat_results, x="feature", y="accuracy", hue="drug_model")
    plots["accuracy"] = f.get_figure()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    palettes = [("green", "orange", "blue"), ("red", "purple", "cyan")]

    for ax, (feature, roc_group), palette in zip(
        axes, roc_results.groupby("feature"), palettes
    ):

        f = sns.lineplot(
            data=roc_group, x="fpr", y="tpr", hue="drug_model", ax=ax, palette=palette
        )
        ax.plot(np.arange(0, 2), np.arange(0, 2), "k--")
        ax.title.set_text(f"{feature} Model")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        roc_auc = list(
            roc_group.groupby(["drug_model"])[["roc_auc"]].mean().itertuples()
        )
        labels = list(map(lambda x: f"{x[0]}, {x[1]:.2f}", roc_auc))
        ax.legend(title="Mean AUROC", loc="upper left", labels=labels)

    plots["roccurve"] = fig

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
                "drug_model": "first",
                "samples": "first",
                "feature": "first",
            }
        )
        final["fpr"] = final["cutter"]
        return final

    roc_results = {
        "fpr": [],
        "tpr": [],
        "roc_auc": [],
        "drug_model": [],
        "samples": [],
        "feature": [],
    }
    for title, group in raw_results.groupby(["drug_model", "samples", "feature"]):
        cat_hat = group[["prob_hat"]].values
        cat_real = group[["cat_real"]].values
        fpr, tpr, _ = roc_curve(cat_real, cat_hat)
        roc_auc = auc(fpr, tpr)
        roc_results["fpr"].extend(fpr.tolist())
        roc_results["tpr"].extend(tpr.tolist())
        roc_results["roc_auc"].extend([roc_auc] * len(fpr))
        roc_results["drug_model"].extend([title[0]] * len(fpr))
        roc_results["samples"].extend([title[1]] * len(fpr))
        roc_results["feature"].extend([title[2]] * len(fpr))

    roc_df = pd.DataFrame(roc_results)
    bins = np.arange(0 - 0.02, 1 + 0.02, 0.02)
    means = np.convolve(bins, [0.5, 0.5], "valid")
    roc_df["cutter"] = pd.cut(roc_df["fpr"], bins, labels=means)
    roc_new = (
        roc_df.groupby(["samples", "drug_model", "feature"], as_index=False)
        .apply(partial(_myagg, bins=means))
        .reset_index(drop=True)
    )
    return roc_new


def choose_best_model(X, y, model, hyper_space, kernels, random_state):
    return GaussianProcessRegressor(kernel=RBF(1000), alpha=4)
    kernel_best = []
    for kernel in kernels:
        opt = BayesSearchCV(
            model(kernel=kernel),
            hyper_space,
            n_iter=10,
            random_state=random_state,
            verbose=10,
            n_points=10,
            cv=5,
        )
        opt.fit(X, y)
        kernel_best.append(opt)
    # sort by best score
    sorted_models = sorted(kernel_best, key=lambda x: x.best_score_, reverse=True)

    # return best estimator, aka best model over all hyperparameters
    return sorted_models[0].best_estimator_


def produce_statistics(raw_results: pd.DataFrame):
    def calc_r(group):
        feature = group["feature"].values[0]
        drugmodel = group["drug_model"].values[0]
        sample = group["samples"].values[0]
        pearson = pearsonr(group["y_real"], group["y_hat"])[0]
        auroc = roc_auc_score(group["cat_real"].values, group["prob_hat"].values)
        accuracy = accuracy_score(group["cat_real"].values, 1 - group["cat_hat"].values)
        return pd.DataFrame(
            [
                {
                    "drug_model": drugmodel,
                    "sample": sample,
                    "pearson": pearson,
                    "auroc": auroc,
                    "accuracy": accuracy,
                    "feature": feature,
                }
            ]
        )

    raw_results["prob_hat"] = 1 - raw_results["y_hat"] / THRESHOLD
    stat_results = (
        raw_results.groupby(["drug_model", "samples", "feature"], as_index=False)
        .apply(calc_r)
        .reset_index(drop=True)
    )
    return stat_results


@cachewrapper("cache", "guanlab_exp_results")
def run_cross_validation(model, hyper_space, kernels, random_state):
    results = []
    for (
        drug,
        feature,
        X_train,
        X_test,
        y_train,
        y_test,
        das_test,
        cat_test,
        yscaler,
    ) in data_generator():
        gpr = choose_best_model(
            X_train, y_train, model, hyper_space, kernels, random_state
        )

        for sample in range(SAMPLES):
            # copy model
            gpr_i = copy.deepcopy(gpr)
            _X_train, _y_train = resample(X_train, y_train)
            _X_test, _y_test, _das_test, _cat_test = resample(
                X_test, y_test, das_test, cat_test
            )

            # fit + predict model
            gpr_i.fit(_X_train, _y_train)
            y_hat = yscaler.inverse_transform(gpr_i.predict(_X_test)).flatten()
            _y_test = yscaler.inverse_transform(_y_test).flatten()

            # prepare results
            drug_list = [drug] * len(y_hat)
            samples = [sample + 1] * len(y_hat)
            features = [feature] * len(y_hat)

            # predicted response
            # cat_hat = list(map(NonResponseRA(), _das_test, y_hat))
            cat_hat = y_hat < THRESHOLD

            assert len(cat_hat) == len(_cat_test)

            _results_i = list(
                zip(drug_list, features, y_hat, _y_test, cat_hat, _cat_test, samples)
            )
            results.extend(_results_i)
    results_df = pd.DataFrame(
        results,
        columns=[
            "drug_model",
            "feature",
            "y_hat",
            "y_real",
            "cat_hat",
            "cat_real",
            "samples",
        ],
    )
    return results_df


def data_generator(path=Path(META_PATH), snp_dict=SNP_BY_DRUG, test_index=TEST_INDEX):
    REGEX_PATTERNS = {"SNPS": r"^rs", "FULL": r"."}
    SIT_OUT_COLMNS = {
        "Response.deltaDAS",
        "Cohort",
        "Batch",
        "Response.EULAR",
        "Response.NonResp",
        "ID",
        "TNF-drug",
    }
    Y_FEATURE = ["Response.deltaDAS"]

    # either create data or read from cache
    data = create_drug_df(path, snp_dict)
    engineered = feature_engineering(data)
    for drug in engineered["TNF-drug"].unique():

        for patt_name, pattern in REGEX_PATTERNS.items():
            _df = engineered[engineered["TNF-drug"] == drug].reset_index(drop=True)

            das = _df["baselineDAS"].values
            cat = _df["Response.NonResp"].values

            re_cols = set(filter(re.compile(pattern).match, engineered.columns)).union(
                SIT_OUT_COLMNS
            )
            _df = _df[list(re_cols)]

            drop_cols = SIT_OUT_COLMNS.intersection(set(re_cols))

            sample = _df.copy()

            _df = _df.dropna(axis=1)

            dropped = list(
                filter(
                    lambda x: not x.startswith("rs"),
                    set(sample.columns) - set(_df.columns),
                )
            )
            assert not dropped

            X = _df.drop(drop_cols, axis=1).values
            y = _df[Y_FEATURE].values

            X, y, yscaler = scale_xy(X, y)

            X_train, X_test, y_train, y_test, das_test, cat_test = test_train_split(
                X, y, das, cat, test_index
            )

            yield drug, patt_name, X_train, X_test, y_train, y_test, das_test, cat_test, yscaler


def scale_xy(X, y):
    yscaler = StandardScaler()
    xscaler = StandardScaler()
    X = xscaler.fit_transform(X)
    y = yscaler.fit_transform(y)
    return X, y, yscaler


def test_train_split(X, y, das, cat, test_index, len_data=2706):
    assert X.shape[0] == len_data
    assert y.shape[0] == len_data
    return (
        X[:test_index],
        X[test_index:],
        y[:test_index],
        y[test_index:],
        das[test_index:],
        cat[test_index:],
    )


def feature_engineering(data):
    to_impute = ["baselineDAS", "Age", "Gender", "Mtx"]
    knn = KNNImputer(n_neighbors=5)
    data[to_impute] = knn.fit_transform(data[to_impute])

    # convert drugs to one-hot encoding
    onehot_drugs = pd.get_dummies(data["Drug"])
    data = data.drop(["Drug"], axis=1)

    data = pd.concat([data, onehot_drugs], axis=1)
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
