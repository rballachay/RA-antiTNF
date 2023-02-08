import sqlite3
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

# number of chromosomes, aka number of dosage files to read
N_CHROMS = 22
N_ENTRIES = 2706

DOSAGE_PATH = "data/DREAM_RA_Responders_DosageData"
CLINICAL_PATH = "data/DREAM_RA_Responders_PhenoCov_Full.txt"


def get_dosage_data(
    snp_list,
    database="dbs/dosage_data.db",
) -> pd.DataFrame:
    """get dosage data when snp names are given

    returns dataframe with snp name as column name,
    index as each subject, and value as dosage
    """
    con = sqlite3.connect(database)
    cur = con.cursor()

    snp_data = cur.execute(
        f"SELECT * FROM SNPlookup WHERE marker_id in ({', '.join(['?']*len(snp_list))})",
        snp_list,
    )
    snp_df = pd.DataFrame(snp_data)
    snp_df = snp_df.iloc[:, [1, 6]]
    snp_df = snp_df.set_index(1)
    snp_dosages = snp_df[6].str.split(", ", expand=True).astype(float)
    return snp_dosages.T


def create_dosage_df(
    n_entries=N_ENTRIES,
    n_chroms=N_CHROMS,
    dosage_prefix=Path(DOSAGE_PATH),
    filename=Path("Training_chr.dos"),
    cols=[
        "chromosome",
        "marker_id",
        "location",
        "allele_1",
        "allele_2",
        "frequency",
    ],
) -> dd.DataFrame:
    """read in all dosage data + create dask dataframe (for memory reasons this
    was chosen instead of pandas) + return. is to be used in the case that snps
    are not already known
    """
    # chromosome file names
    chr_filenames = list(
        map(
            lambda x: dosage_prefix / f"{filename.stem}{x+1}{filename.suffix}",
            range(n_chroms),
        )
    )
    subjects = [f"subject_{n+1}" for n in range(n_entries)]
    names = cols + subjects
    dtypes = dict(zip(subjects, ["float64"] * len(subjects)))

    df = dd.read_csv(chr_filenames[0], sep=" ", header=None, names=names, dtype=dtypes)
    for file in chr_filenames[1:]:
        df2 = dd.read_csv(file, sep=" ", header=None, names=names, dtype=dtypes)
        df = dd.concat([df, df2])

    return df


def create_clinical_data(
    data_path=Path(CLINICAL_PATH),
) -> pd.DataFrame:
    return pd.read_csv(data_path, sep=" ")


if __name__ == "__main__":
    # create_dosage_data()
    # create_clinical_data()
    get_dosage_data(["rs10265155", "rs1990099", "rs10833455", "rs10833456"])
