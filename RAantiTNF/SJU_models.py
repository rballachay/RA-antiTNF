import copy
import re
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
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
    "all": "plink_LD_SNPs_best",
}
HYPER_DICT = {
    Lasso: {"alpha": Real(1e-3, 1, prior="log-uniform")},
    SVR: {
        "kernel": Categorical(["linear", "rbf", "poly"]),
        "C": Real(1e-3, 10, prior="log-uniform"),
        "epsilon": Real(1e-3, 10, prior="log-uniform"),
    },
    RandomForestRegressor: {"n_estimators": Integer(10, 1000, prior="log-uniform")},
    GaussianProcessRegressor: {"alpha": Real(1e-1, 1e2, prior="log-uniform")},
}
MODEL_DICT = {
    # Lasso: Lasso(alpha=0.01),
    SVR: SVR(kernel="linear", C=0.1, epsilon=0.01),
    RandomForestRegressor: RandomForestRegressor(n_estimators=300),
    # GaussianProcessRegressor: GaussianProcessRegressor(kernel=RBF(), alpha=0.5),
}
PLOT_DICT = {
    "pearson": "pearson_results.png",
    "scatterplot": "scatterplot_results.png",
    "auroc": "auroc_results.png",
    "accuracy": "accuracy_results.png",
    "roccurve": "roccurve_results.png",
    "xfeatures": "feature_importance.png",
}
RESULTS_PATH = Path("results/SJU_models/exp_2_feb22")
REGEX_PATTERNS = {
    "Full": r".",
    "Baseline": r"baselineDAS",
    "AGM": r"Age|Gender|Mtx",
    "G-free": r"^(?!Gender).*$",
    "Genetic": r"^rs",
}
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

THRESHOLD = 2.5
SAMPLES = 10
USE_CACHE = False


def main(
    models=(
        SVR,
        RandomForestRegressor,
    ),  # Lasso, GaussianProcessRegressor),
    hyper_dict=HYPER_DICT,
    random_state=RANDOM_STATE,
    results_path=RESULTS_PATH,
    plot_dict=PLOT_DICT,
):
    # create our model with all default parameters
    raw_results, xfeats = run_cross_validation(models, hyper_dict, random_state)
    stat_results = produce_statistics(raw_results)
    roc_results = prep_roc_curve(raw_results)

    # assert results path exists
    results_path.mkdir(parents=True, exist_ok=True)

    # visualize results + plot
    plots = visualize_results(raw_results, stat_results, roc_results, xfeats)

    # save to output directory
    for title, fig in plots.items():
        fig.savefig(str(results_path / plot_dict[title]))


def visualize_results(raw_results, stat_results, roc_results, xfeats):
    plots = {}
    sns.set_theme()

    palette = sns.color_palette("hls", 5)

    fig, axes = plt.subplots(stat_results["model"].nunique(), 1, figsize=(5, 12))
    for (title, group), ax in zip(stat_results.groupby("model", as_index=False), axes):
        f = sns.barplot(
            data=group,
            x="feature",
            y="pearson",
            palette=palette,
            order=["Full", "Baseline", "AGM", "G-free", "Genetic"],
            ax=ax,
        )
        f.set_title(f"??DAS Pearson Correlation {title}")
        f.set_ylim((0, 1))
        for container in f.containers:
            f.bar_label(container, label_type="center", fmt="%.3f")
    fig.tight_layout()
    plots["pearson"] = f.get_figure()

    plt.figure()
    f = sns.scatterplot(data=raw_results, x="y_real", y="y_hat", hue="model")
    f.set_xlim((-2, 7))
    f.set_ylim((-2, 7))
    plots["scatterplot"] = f.get_figure()

    plt.figure()
    fig, axes = plt.subplots(stat_results["model"].nunique(), 1, figsize=(5, 12))
    for (title, group), ax in zip(stat_results.groupby("model", as_index=False), axes):
        f = sns.barplot(
            data=group,
            x="feature",
            y="auroc",
            palette=palette,
            order=["Full", "Baseline", "AGM", "G-free", "Genetic"],
            ax=ax,
        )
        f.set_title(f"EULAR Response AUROC {title}")
        f.set_ylim((0, 1))
        for container in f.containers:
            f.bar_label(container, label_type="center", fmt="%.3f")
    fig.tight_layout()
    plots["auroc"] = f.get_figure()

    plt.figure()
    f = sns.violinplot(data=stat_results, x="feature", y="accuracy", hue="model")
    plots["accuracy"] = f.get_figure()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    nun = roc_results["model"].nunique()
    palettes = sns.color_palette("hls", nun * 2)
    palettes = palettes[nun:], palettes[:nun]

    roc_ok = roc_results[roc_results["feature"].isin(["Full", "Genetic", "ASM"])]
    for ax, (feature, roc_group), palette in zip(
        axes, roc_ok.groupby("feature"), palettes
    ):

        f = sns.lineplot(
            data=roc_group, x="fpr", y="tpr", hue="model", ax=ax, palette=palette
        )
        ax.plot(np.arange(0, 2), np.arange(0, 2), "k--")
        ax.title.set_text(f"{feature} Model")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        roc_auc = list(roc_group.groupby(["model"])[["roc_auc"]].mean().itertuples())
        labels = list(map(lambda x: f"{x[0]}, {x[1]:.2f}", roc_auc))
        ax.legend(
            title="Mean AUROC",
            loc="upper left",
            labels=labels,
            handles=ax.get_legend().legendHandles,
        )

    plots["roccurve"] = fig

    xfeats = xfeats[xfeats["feature"].isin(["Full", "Genetic"])]
    fig, axes = plt.subplots(
        xfeats["feature"].nunique(), xfeats["model"].nunique(), figsize=(20, 8)
    )
    all_best_features = (
        xfeats.groupby("label", as_index=False)
        .max()
        .sort_values(by="importance", ascending=False)
    )["label"]
    all_labels = all_best_features.unique()
    palette = dict(
        zip(all_labels, sns.color_palette("Spectral", n_colors=len(all_labels)))
    )
    xfeats["SNP"] = xfeats["label"].apply(lambda x: x.startswith("rs"))
    xfeats["importance"] = xfeats["importance"].abs()
    for i, feature in enumerate(xfeats.feature.unique()):
        for j, model in enumerate(xfeats.model.unique()):
            _xfeatsi = xfeats[
                (xfeats["feature"] == feature) & (xfeats["model"] == model)
            ]
            best_features = (
                _xfeatsi.groupby("label", as_index=False)
                .mean()
                .sort_values(by="importance", ascending=False)
            )["label"].iloc[:10]
            _xfeatsi = _xfeatsi[_xfeatsi["label"].isin(best_features)].sort_values(
                by="label",
                key=lambda column: column.map(lambda e: list(best_features).index(e)),
            )

            bar = sns.barplot(
                data=_xfeatsi, x="label", y="importance", ax=axes[i, j], palette=palette
            )
            for item in bar.get_xticklabels():
                item.set_rotation(20)

            axes[i, j].title.set_text(f"{model} Model, {feature} Features")
            axes[i, j].set_xlabel("Top 10 Features")
            axes[i, j].set_ylabel("Coefficient Weight")

    fig.tight_layout()
    plots["xfeatures"] = fig

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
                "feature": "first",
            }
        )
        final["fpr"] = final["cutter"]
        return final

    roc_results = {
        "fpr": [],
        "tpr": [],
        "roc_auc": [],
        "model": [],
        "samples": [],
        "feature": [],
    }
    for title, group in raw_results.groupby(["model", "samples", "feature"]):
        cat_hat = group[["prob_hat"]].values
        cat_real = group[["cat_real"]].values
        fpr, tpr, _ = roc_curve(cat_real, cat_hat)
        roc_auc = auc(fpr, tpr)
        roc_results["fpr"].extend(fpr.tolist())
        roc_results["tpr"].extend(tpr.tolist())
        roc_results["roc_auc"].extend([roc_auc] * len(fpr))
        roc_results["model"].extend([title[0]] * len(fpr))
        roc_results["samples"].extend([title[1]] * len(fpr))
        roc_results["feature"].extend([title[2]] * len(fpr))

    roc_df = pd.DataFrame(roc_results)
    bins = np.arange(0 - 0.02, 1 + 0.02, 0.02)
    means = np.convolve(bins, [0.5, 0.5], "valid")
    roc_df["cutter"] = pd.cut(roc_df["fpr"], bins, labels=means)
    roc_new = (
        roc_df.groupby(["samples", "model", "feature"], as_index=False)
        .apply(partial(_myagg, bins=means))
        .reset_index(drop=True)
    )
    return roc_new


def choose_best_model(X, y, model, hyper, random_state):
    return MODEL_DICT[model]
    opt = BayesSearchCV(
        model(),
        hyper,
        n_iter=2,
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
        feature = group["feature"].values[0]
        model = group["model"].values[0]
        sample = group["samples"].values[0]
        pearson = pearsonr(group["y_real"], group["y_hat"])[0]
        auroc = roc_auc_score(group["cat_real"].values, group["prob_hat"].values)
        accuracy = accuracy_score(group["cat_real"].values, group["cat_hat"].values)
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "feature": feature,
                    "samples": sample,
                    "pearson": np.nan_to_num(pearson),
                    "auroc": auroc,
                    "accuracy": accuracy,
                }
            ]
        )

    X = raw_results["y_hat"]
    raw_results["prob_hat"] = 1 - raw_results["y_hat"]
    raw_results["cat_hat"] = raw_results["y_hat"] > THRESHOLD
    stat_results = (
        raw_results.groupby(["feature", "samples", "model"], as_index=False)
        .apply(calc_r)
        .reset_index(drop=True)
    )
    return stat_results


@cachewrapper("cache", ("SJU_plink_results", "SJU_coeff_results"), USE_CACHE)
def run_cross_validation(models, hyper_dict, random_state):
    results = []
    xfeats = []
    for (
        feature,
        X_train,
        X_test,
        y_train,
        y_test,
        das_test,
        cat_test,
        yscaler,
        x_labels,
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
                print(f"Sample {sample}, model {best_model}")

                # predict + expand dims
                preds = best_model_i.predict(_X_test)

                if preds.ndim == 1:
                    preds = preds[:, None]

                y_hat = yscaler.inverse_transform(preds).flatten()
                _y_test = yscaler.inverse_transform(_y_test).flatten()

                print("roc=", roc_auc_score(_cat_test, 1 - y_hat))
                print("pearson=", pearsonr(_y_test, y_hat)[0])

                # prepare results
                feature_list = [feature] * len(y_hat)
                samples = [sample + 1] * len(y_hat)
                model_list = [model.__name__] * len(y_hat)

                _results_i = list(
                    zip(
                        model_list,
                        feature_list,
                        y_hat,
                        _y_test,
                        _cat_test,
                        samples,
                    )
                )
                results.extend(_results_i)

                if hasattr(best_model_i, "coef_"):
                    importance = best_model_i.coef_.flatten()
                elif hasattr(best_model_i, "feature_importances_"):
                    importance = best_model_i.feature_importances_.flatten()
                else:
                    importance = None

                if importance is not None:
                    _xfeats_i = list(
                        zip(
                            [model.__name__] * len(importance),
                            [feature] * len(importance),
                            importance,
                            x_labels,
                        )
                    )
                    xfeats.extend(_xfeats_i)

    results_df = pd.DataFrame(
        results,
        columns=["model", "feature", "y_hat", "y_real", "cat_real", "samples"],
    )
    xfeats_df = pd.DataFrame(
        xfeats, columns=["model", "feature", "importance", "label"]
    )
    return results_df, xfeats_df


def data_generator(path=Path(META_PATH), snp_dict=SNP_BY_DRUG, test_index=TEST_INDEX):

    # either create data or read from cache
    data = create_drug_df(path, snp_dict)
    engineered = feature_engineering(data)

    for drug in snp_dict:

        for patt_name, pattern in REGEX_PATTERNS.items():
            _df = engineered[engineered["TNF-drug"] == drug].reset_index(drop=True)
            _df["tt_split"] = ["train"] * test_index + ["test"] * (
                len(_df) - test_index
            )
            # _df = _df[_df["Drug"] == "TNF-drug"]
            train_indices = _df["tt_split"].apply(lambda x: x == "train").values

            das = _df["baselineDAS"].values
            cat = _df["Response.NonResp"].values

            re_cols = set(filter(re.compile(pattern).match, engineered.columns)).union(
                SIT_OUT_COLMNS
            )
            _df = _df[list(re_cols)]

            drop_cols = SIT_OUT_COLMNS.intersection(set(re_cols))

            _df_x = _df.drop(drop_cols, axis=1)
            X = _df_x.values
            y = _df[Y_FEATURE].values

            X, y, yscaler = scale_xy(X, y)

            X_train, X_test, y_train, y_test, das_test, cat_test = test_train_split(
                X, y, das, cat, train_indices
            )

            yield patt_name, X_train, X_test, y_train, y_test, das_test, cat_test, yscaler, _df_x.columns


def scale_xy(X, y):
    yscaler = StandardScaler()
    xscaler = StandardScaler()
    X = xscaler.fit_transform(X)
    y = yscaler.fit_transform(y)
    return X, y, yscaler


def test_train_split(X, y, das, cat, train_indices):
    return (
        X[train_indices],
        X[~train_indices],
        y[train_indices],
        y[~train_indices],
        das[~train_indices],
        cat[~train_indices],
    )


def feature_engineering(data):
    to_impute = ["baselineDAS", "Age", "Gender", "Mtx"]
    knn = KNNImputer(n_neighbors=5)
    data[to_impute] = knn.fit_transform(data[to_impute])
    # convert drugs to one-hot encoding
    # onehot_drugs = pd.get_dummies(data["Drug"])
    data = data.drop(["Drug"], axis=1)
    # data = pd.concat([data, onehot_drugs], axis=1)
    return data


@cachewrapper("cache", "SJU_plink_fulldata", USE_CACHE)
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
