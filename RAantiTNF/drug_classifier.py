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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

RANDOM_STATE = 0
CACHE_NAME = "drug_classifier"
META_PATH = "metadata/drug_models/"
SNP_BY_DRUG = {
    "all": "top_100_alldrugs",
}
HYPER_DICT = {
    LogisticRegression: {
        "penalty": Categorical(["l2", "none"]),
        "C": Real(1e-2, 1e1, prior="log-uniform"),
    },
    RandomForestClassifier: {"n_estimators": Integer(50, 500)},
}
MODEL_DICT = {
    LogisticRegression: LogisticRegression(C=1),
    RandomForestClassifier: RandomForestClassifier(n_estimators=250),
}
PLOT_DICT = {
    "pearson": "pearson_results.png",
    "scatterplot": "scatterplot_results.png",
    "auroc": "auroc_results.png",
    "accuracy": "accuracy_results.png",
    "roccurve": "roccurve_results.png",
    "xfeatures": "feature_importance.png",
    "confusion": "confusion_matrix.png",
}
RESULTS_PATH = Path("results/drug_models/exp_3_mar2")
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
    "Drug",
}
Y_FEATURE = ["Drug"]

THRESHOLD = 2.5
SAMPLES = 10
USE_CACHE = True


def main(
    models=(LogisticRegression, RandomForestClassifier),
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

    raw_results = raw_results.copy()

    fig, axes = plt.subplots(
        stat_results["model"].nunique(),
        stat_results["feature"].nunique(),
        figsize=(35, 12),
    )
    for (titles, subdf), ax in zip(
        raw_results.groupby(["model", "feature"]), axes.flatten()
    ):
        y_hat = np.argmax(
            subdf[["adalimumab", "etanercept", "infliximab"]].values, axis=1
        )
        y_real = subdf["y_real"].values
        confusion = pd.DataFrame(
            confusion_matrix(y_real, y_hat),
            columns=["adalimumab", "etanercept", "infliximab"],
            index=["adalimumab", "etanercept", "infliximab"],
        )
        sns.heatmap(confusion, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("True Drug")
        ax.set_ylabel("Predicted Drug")
        ax.set_title(
            f'{", ".join(titles)}, Accuracy = {accuracy_score(y_real,y_hat):.2f}'
        )
    fig.tight_layout()
    plots["confusion"] = fig.get_figure()

    palette = sns.color_palette("hls", 5)

    plt.figure()
    fig, axes = plt.subplots(stat_results["model"].nunique(), 1, figsize=(5, 12))
    if not isinstance(axes, np.ndarray):
        axes = [
            axes,
        ]
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

    roc_ok = roc_results[roc_results["feature"].isin(["Full", "Genetic", "AGM"])]

    nun = roc_ok["feature"].nunique()
    drug = roc_ok["drug"].nunique()
    palettes = [list(sns.color_palette("hls", drug))] * nun * roc_ok["model"].nunique()

    fig, axes = plt.subplots(
        roc_ok["feature"].nunique(), roc_ok["model"].nunique(), figsize=(15, 10)
    )

    for ax, (feature, roc_group), palette in zip(
        axes.flatten(), roc_ok.groupby(["feature", "model"]), palettes
    ):
        f = sns.lineplot(
            data=roc_group, x="fpr", y="tpr", hue="drug", ax=ax, palette=palette
        )
        ax.plot(np.arange(0, 2), np.arange(0, 2), "k--")
        ax.title.set_text(f"{', '.join(feature)} Model")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        roc_auc = list(roc_group.groupby(["drug"])[["roc_auc"]].mean().itertuples())
        labels = list(map(lambda x: f"{x[0]}, {x[1]:.2f}", roc_auc))
        ax.legend(
            title="Mean AUROC",
            loc="upper left",
            labels=labels,
            handles=ax.get_legend().legendHandles,
        )

    fig.tight_layout()
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
                "drug": "first",
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
        "drug": [],
    }

    for i, drug in enumerate(["adalimumab", "etanercept", "infliximab"]):
        for title, group in raw_results.groupby(["model", "samples", "feature"]):
            cat_hat = group[[drug]].values
            cat_real = (group[["y_real"]] == i).values
            fpr, tpr, _ = roc_curve(cat_real, cat_hat)
            roc_auc = auc(fpr, tpr)
            roc_results["fpr"].extend(fpr.tolist())
            roc_results["tpr"].extend(tpr.tolist())
            roc_results["roc_auc"].extend([roc_auc] * len(fpr))
            roc_results["model"].extend([title[0]] * len(fpr))
            roc_results["samples"].extend([title[1]] * len(fpr))
            roc_results["feature"].extend([title[2]] * len(fpr))
            roc_results["drug"].extend([drug] * len(fpr))

    roc_df = pd.DataFrame(roc_results)
    bins = np.arange(0 - 0.02, 1 + 0.02, 0.02)
    means = np.convolve(bins, [0.5, 0.5], "valid")
    roc_df["cutter"] = pd.cut(roc_df["fpr"], bins, labels=means)
    roc_new = (
        roc_df.groupby(["drug", "samples", "model", "feature"], as_index=False)
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
        feature = group["feature"].values[0]
        model = group["model"].values[0]
        sample = group["samples"].values[0]
        auroc = roc_auc_score(
            pd.get_dummies(group["y_real"]).values,
            group[["adalimumab", "etanercept", "infliximab"]].values,
            multi_class="ovr",
        )
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "feature": feature,
                    "samples": sample,
                    "auroc": auroc,
                }
            ]
        )

    stat_results = (
        raw_results.groupby(["feature", "samples", "model"], as_index=False)
        .apply(calc_r)
        .reset_index(drop=True)
    )
    return stat_results


@cachewrapper(
    "cache", (f"{CACHE_NAME}_results", f"{CACHE_NAME}_coeff_results"), USE_CACHE
)
def run_cross_validation(models, hyper_dict, random_state):
    results = []
    xfeats = []
    for (
        feature,
        X_train,
        X_test,
        y_train,
        y_test,
        X_features,
        y_features,
    ) in data_generator():
        for model in models:
            best_model = choose_best_model(
                X_train, y_train, model, hyper_dict[model], random_state
            )
            for sample in range(SAMPLES):
                # copy model
                best_model_i = copy.deepcopy(best_model)
                _X_train, _y_train = resample(X_train, y_train)
                _X_test, _y_test = resample(X_test, y_test)
                print(f"Sample {sample}, model {best_model}")
                # fit + predict model
                best_model_i.fit(_X_train, _y_train)

                # predict + expand dims
                y_hat = best_model_i.predict_proba(_X_test)

                # prepare results
                feature_list = [feature] * len(y_hat)
                samples = [sample + 1] * len(y_hat)
                model_list = [model.__name__] * len(y_hat)

                _results_i = list(
                    zip(
                        model_list,
                        feature_list,
                        _y_test,
                        samples,
                        y_hat[:, 0],
                        y_hat[:, 1],
                        y_hat[:, 2],
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
                            X_features,
                        )
                    )
                xfeats.extend(_xfeats_i)

    results_df = pd.DataFrame(
        results,
        columns=[
            "model",
            "feature",
            "y_real",
            "samples",
            y_features[0],
            y_features[1],
            y_features[2],
        ],
    )
    xfeats_df = pd.DataFrame(
        xfeats, columns=["model", "feature", "importance", "label"]
    )
    return results_df, xfeats_df


def data_generator(path=Path(META_PATH), snp_dict=SNP_BY_DRUG, test_index=TEST_INDEX):

    # either create data or read from cache
    data = create_drug_df(path, snp_dict)
    engineered = feature_engineering(data)

    for drug in engineered["TNF-drug"].unique():
        for patt_name, pattern in REGEX_PATTERNS.items():
            _df = engineered[engineered["TNF-drug"] == drug].reset_index(drop=True)

            _df["tt_split"] = ["train"] * test_index + ["test"] * (
                len(_df) - test_index
            )
            _df = _df[_df["Drug"].notnull()]
            train_indices = _df["tt_split"].apply(lambda x: x == "train").values

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
                    set(sample.columns) - set(_df.columns.tolist()),
                )
            )
            assert not dropped

            X = _df.drop(drop_cols, axis=1)
            X_features = X.columns.tolist()
            X = X.values
            y_features, y = np.unique(_df[Y_FEATURE].values, return_inverse=True)

            X, y = scale_xy(X, y)

            X_train, X_test, y_train, y_test = test_train_split(X, y, train_indices)

            yield patt_name, X_train, X_test, y_train, y_test, X_features, y_features


def scale_xy(X, y):
    xscaler = StandardScaler()
    X = xscaler.fit_transform(X)
    return X, y


def test_train_split(X, y, train_indices):
    return (
        X[train_indices],
        X[~train_indices],
        y[train_indices],
        y[~train_indices],
    )


def feature_engineering(data):
    to_impute = ["baselineDAS", "Age", "Gender", "Mtx"]
    knn = KNNImputer(n_neighbors=5)
    data[to_impute] = knn.fit_transform(data[to_impute])
    return data


@cachewrapper("cache", f"{CACHE_NAME}_fulldata", USE_CACHE)
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
