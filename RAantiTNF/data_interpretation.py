from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datareader import create_clinical_data, get_dosage_data
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SNP_BY_DRUG = [
    "top_100_alldrugs",
    "plink_LD_SNPs_best",
    "Guanlab_adalimumab_snps",
    "Guanlab_etanercept_snps",
    "Guanlab_infliximab_snps",
    "all_snps_community",
    "all_guanlab_snps",
    "lucia_snps",
]
META_PATH = [
    "metadata/drug_models/",
    "metadata/SJU_plink/",
    "metadata/guanlab",
    "metadata/guanlab",
    "metadata/guanlab",
    "metadata/guanlab",
    "metadata/guanlab",
    "metadata/guanlab",
]
RESULTS_PATH = Path("results/drug_models/exp_4_mar2")
LOCATION_MAP = {
    "BRAGGS": "UK",
    "DREAM": "Netherlands",
    "TEAR": "US",
    "Immunex": "US",
    "EIRA": "Sweden",
    "ABCoN": "US",
    "BRASS": "US",
    "react": "France",
    "new": "Netherlands",
}


def main(paths=META_PATH, snp_dict=SNP_BY_DRUG, results_path=RESULTS_PATH):

    # assert results path exists
    results_path.mkdir(parents=True, exist_ok=True)

    lda_run(
        Path(paths[7]),
        snp_dict[7],
        results_path,
        title="LDA on Lucia SNPs",
        hue="Response.EULAR",
        style=None,
    )

    lda_run(
        Path(paths[5]),
        snp_dict[5],
        results_path,
        title="LDA on All SNPs",
        hue="Response.EULAR",
        style=None,
    )

    lda_run(
        Path(paths[6]),
        snp_dict[6],
        results_path,
        title="LDA on Guanlab SNPs",
        hue="Response.EULAR",
        style=None,
    )
    lda_run(
        Path(paths[0]),
        snp_dict[0],
        results_path,
        title="LDA on Drug SNPs",
        hue="Response.EULAR",
        style=None,
    )

    pca_run(
        Path(paths[0]),
        snp_dict[0],
        results_path,
        title="PCA on Drug Model SNPS",
        hue="Cohort",
        style="Drug",
    )

    pca_run(
        Path(paths[1]),
        snp_dict[1],
        results_path,
        title="PCA on PLINK SNPS",
        hue="Cohort",
        style="Drug",
    )

    pca_run(
        Path(paths[2]),
        snp_dict[2],
        results_path,
        title="PCA on Guanlab adalimumab SNPS",
        hue="Cohort",
        style="Drug",
    )

    pca_run(
        Path(paths[3]),
        snp_dict[3],
        results_path,
        title="PCA on Guanlab etanercept SNPS",
        hue="Cohort",
        style="Drug",
    )

    pca_run(
        Path(paths[4]),
        snp_dict[4],
        results_path,
        title="PCA on Guanlab infliximab SNPS",
        hue="Cohort",
        style="Drug",
    )


def pca_run(path, snp_dict, results_path, title, hue, style):
    pca = PCA()
    data = create_drug_df(path, snp_dict).iloc[:2032]
    _xfeatures = data[list(filter(lambda x: x.startswith("rs"), data.columns))]
    _xfeatures = _xfeatures.fillna(0)
    pca.fit(_xfeatures.values)

    latent = pca.transform(_xfeatures)
    sns.set()
    fig = sns.scatterplot(
        x=latent[:, 0],
        y=latent[:, 1],
        hue=data[hue],
        palette="hls",
        style=data[style],
    )
    fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.title(title)
    ratios = pca.explained_variance_ratio_
    plt.xlabel(f"PCA 1 ({100*ratios[0]:.1f} %)")
    plt.ylabel(f"PCA 2 ({100*ratios[1]:.1f} %)")
    plt.tight_layout()
    fig = fig.get_figure().savefig(
        f"{results_path}/{title.lower().replace(' ','_')}.png", dpi=200
    )
    plt.clf()


def lda_run(path, snp_dict, results_path, title, hue, style):
    lda = LinearDiscriminantAnalysis()
    data = create_drug_df(path, snp_dict)
    _xfeatures = data[list(filter(lambda x: x.startswith("rs"), data.columns))]
    _xfeatures = _xfeatures.fillna(0)
    lda.fit(_xfeatures.values[:2032], data["Response.EULAR"].iloc[:2032])

    latent = lda.transform(_xfeatures.values[2032:])
    sns.set()
    fig = sns.scatterplot(
        x=latent[:, 0],
        y=latent[:, 1],
        hue=data[hue].loc[2032:],
        palette="hls",
        style=None,
    )
    fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    plt.title(title)
    ratios = lda.explained_variance_ratio_
    plt.xlabel(f"Component 1 ({100*ratios[0]:.1f} %)")
    plt.ylabel(f"Component 2 ({100*ratios[1]:.1f} %)")
    plt.tight_layout()
    plt.xlim([-5, 5])
    fig = fig.get_figure().savefig(
        f"{results_path}/{title.lower().replace(' ','_')}.png", dpi=200
    )
    plt.clf()
    coef = np.mean(np.abs(lda.coef_), axis=0)
    coefs = pd.DataFrame({"coef": coef, "title": _xfeatures.columns.tolist()})
    coefs = coefs.sort_values(by=["coef"])[-25:]
    plt.figure(figsize=(20, 5))
    fig = sns.barplot(data=coefs, x="title", y="coef", palette="hls")
    fig = fig.get_figure().savefig(
        f"{results_path}/{title.lower().replace(' ','_')}_coeff.png", dpi=200
    )
    plt.clf()


def create_drug_df(path, snp_dict):
    """returns a dictionary with three keys, one for each drug. each of
    these drugs then have two subkeys, one for training and one for testing.
    """

    clinical_data = create_clinical_data()
    releveant_snps = _read_snps(path, snp_dict)

    relevant_dosage = get_dosage_data(releveant_snps)
    drug_features = pd.concat([clinical_data, relevant_dosage], axis=1)

    return drug_features


def _read_snps(path, snp_file) -> dict:
    """read in the text files from guanlab that describe which snps belong"""
    snppath = path / snp_file
    with open(snppath, "r") as txtfile:
        lines = txtfile.read()
        snps = list(filter(lambda x: x, lines.split("\t")))
        snps = [i for x in snps for i in x.split("\n")]
        snps = list(filter(lambda x: x, snps))
    return snps


if __name__ == "__main__":
    main()
