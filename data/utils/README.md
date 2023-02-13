# Data Utils 

## Merging data

In order to merge the raw genotype data from different experiments in the RA challenge, download PLINK2 and run the following command:

```bash
cd data/RA_challenge_qced_genotypes && bash ../utils/mergeSNPS.sh
```

## Create phenotype data

```bash
python3 -c 'import pandas as pd; data=pd.read_csv("data/DREAM_RA_Responders_PhenoCov_Full.txt", sep=" "); data.insert(0, "FID", 0); data=data.rename(columns={"ID":"IID"}); data.to_csv("data/utils/DREAM_pheno_FID.txt", index=False, sep="\t")'
```

## Run PLINK

```bash
plink --bfile data/Merged_Genotype/mergedDREAM --pheno data/utils/DREAM_pheno_FID.txt  --linear hide-covar --pheno-name Response.deltaDAS
```