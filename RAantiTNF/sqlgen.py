import sqlite3
from pathlib import Path

import pandas as pd

# number of chromosomes, aka number of dosage files to read
N_CHROMS = 22
N_ENTRIES = 2706


def create_dosage_data(
    n_chroms=N_CHROMS,
    database="dbs/dosage_data.db",
    dosage_prefix=Path("data/DREAM_RA_Responders_DosageData"),
    filename=Path("Training_chr.dos"),
    start_col=6,
):
    # chromosome file names
    chr_filenames = map(
        lambda x: dosage_prefix / f"{filename.stem}{x+1}{filename.suffix}",
        range(n_chroms),
    )

    con = sqlite3.connect(database)
    cur = con.cursor()

    # create lookup table for each of the snps
    cur.execute(
        """CREATE TABLE IF NOT EXISTS SNPlookup (
                                        chromosome integer NOT NULL,
                                        marker_id text PRIMARY KEY,
                                        location integer,
                                        allele_1 text NOT NULL,
                                        allele_2 text NOT NULL,
                                        frequency real NOT NULL,
                                        dosages text NOT NULL
                                    );"""
    )
    con.commit()

    for file in chr_filenames:
        SNPlookup = []
        with open(file, "r") as input_file:
            for lines in input_file:
                lines = lines.strip("\n").split(" ")
                _update = lines[:start_col] + [
                    ", ".join(list(map(str, lines[start_col:])))
                ]
                SNPlookup.append(_update)

        cur.executemany(
            f"""INSERT INTO SNPlookup 
                                (chromosome, marker_id, location, allele_1, allele_2, frequency, dosages) 
                                VALUES (?, ?, ?, ?, ?, ?, ?);""",
            SNPlookup,
        )
    con.commit()


def convert_dos_to_pickle(
    n_chroms=N_CHROMS,
    n_entries=N_ENTRIES,
    dosage_prefix=Path("data/DREAM_RA_Responders_DosageData"),
    filename=Path("Training_chr.dos"),
    pickle_path=Path("data/DREAM_pickles"),
    cols=[
        "chromosome",
        "marker_id",
        "location",
        "allele_1",
        "allele_2",
        "frequency",
    ],
):
    # chromosome file names
    chr_filenames = map(
        lambda x: dosage_prefix / f"{filename.stem}{x+1}{filename.suffix}",
        range(n_chroms),
    )

    subjects = [f"subject_{n+1}" for n in range(n_entries)]
    names = cols + subjects
    dtypes = dict(zip(subjects, ["float64"] * len(subjects)))

    for file in list(chr_filenames)[5:]:
        print(f"starting {file}")
        df = pd.read_csv(file, sep=" ", header=None, names=names, dtype=dtypes)
        df.to_pickle(str(pickle_path / f"{file.stem}.pkl"))


if __name__ == "__main__":
    # create_dosage_data()
    convert_dos_to_pickle()
