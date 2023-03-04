import os
import pickle
from pathlib import Path

import seaborn as sns
from sklearn.decomposition import In
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from RAantiTNF.datareader import create_clinical_data, create_dosage_df

LDA_NAME = Path("models/lda_snp_selector.pkl")
PLOT_NAME = Path("models/visualization/lda_top2_latent.png")
USE_CACHE = True


def main():
    create_fit_lda()


def visualize_lda_results():
    response = create_clinical_data()["Response.NonResp"].values
    for i, dosage in enumerate(create_dosage_df()):
        filename = f"{LDA_NAME.parent/LDA_NAME.stem}{i+1}{LDA_NAME.suffix}"
        plotname = f"{PLOT_NAME.parent/PLOT_NAME.stem}{i+1}{PLOT_NAME.suffix}"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                lda = pickle.load(f)
            sample_titles = list(
                filter(lambda x: x.startswith("subject"), dosage.columns)
            )
            snp_dosages = dosage[sample_titles].values.T
            latent = lda.transform(snp_dosages)
            fig = sns.scatterplot(x=latent[:, 0], y=latent[:, 1], hue=response)
            fig.get_figure().savefig(plotname)


def create_fit_lda():
    response = create_clinical_data()["Response.deltaDAS"].values.round()
    for i, dosage in enumerate(create_dosage_df(pickle=True)):
        filename = f"{LDA_NAME.parent/LDA_NAME.stem}{i+1}{LDA_NAME.suffix}"
        plotname = f"{PLOT_NAME.parent/PLOT_NAME.stem}{i+1}{PLOT_NAME.suffix}"
        # if os.path.exists(filename):
        #    continue
        snps = dosage["marker_id"]
        sample_titles = list(filter(lambda x: x.startswith("subject"), dosage.columns))
        snp_dosages = dosage[sample_titles].values.T
        print(snp_dosages.shape)
        print(f"starting LDA {i+1}")
        lda = LinearDiscriminantAnalysis()
        lda.fit(snp_dosages, response)
        latent = lda.transform(snp_dosages)
        fig = sns.scatterplot(x=latent[:, 0], y=latent[:, 1], hue=response)
        fig.get_figure().savefig(plotname)

        print("saving LDA model")
        with open(filename, "wb") as f:
            pickle.dump(lda, f)


if __name__ == "__main__":
    main()
