import dask.dataframe as dd

df = dd.read_csv(
    "/Users/RileyBallachay/Documents/McGill/RA-antiTNF/data/DREAM_RA_Responders_DosageData/Training_chr1.dos",
    sep=" ",
    header=None,
)

print(df.head())
