# Data Utils 

## METHOD 1 

This method attempts to run GWAS on the un-imputed raw genotype files 

### Merging data

In order to merge the raw genotype data from different experiments in the RA challenge, download PLINK2 and run the following command:

```bash
cd data/RA_challenge_qced_genotypes && bash ../utils/mergeSNPS.sh
```

### Create phenotype data

```bash
python3 -c 'import pandas as pd; data=pd.read_csv("data/DREAM_RA_Responders_PhenoCov_Full.txt", sep=" "); data.insert(0, "FID", 0); data=data.rename(columns={"ID":"IID"}); data.to_csv("data/utils/DREAM_pheno_FID.txt", index=False, sep="\t")'
```

### Run PLINK

```bash
plink --bfile data/Merged_Genotype/mergedDREAM --pheno data/utils/DREAM_pheno_FID.txt  --linear hide-covar --pheno-name Response.deltaDAS
```

## METHOD 2

This method uses the imputed genotype dosages (.gen) instead of the raw genotype files.

### Create merged .fam file
```bash
cat data/DREAM_RA_Responders_GenProbData/* > data/Merged_Genotype/Training_all.gen
```

### Create .bed + .bim + .fam
```bash
plink --gen data/Merged_Genotype/Training_all.gen --sample data/TrainingData_PhenoCov_Release.sample
```

### Run Logistic Regression GWAS
```bash
plink --bfile data/Merged_Genotype/MergedGen --logistic
```

### Get Duplicate SNPs
```bash
cut -f 2 plink.assoc.logistic | sort | uniq -d > duplicateSNP.dups
```

### Run clumping
```bash
plink --bfile data/Merged_Genotype/MergedGen --clump plink.assoc.logistic  --exclude duplicateSNP.dups
```

### Get best SNPs
```bash
python -c 'import pandas as pd; import numpy as np; df=pd.read_csv("plink.clumped", delimiter=r"\s+"); np.savetxt("metadata/SJU_plink/plink_LD_SNPs", df.SNP, fmt='%s')'
```