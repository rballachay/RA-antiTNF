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

### Re-write phenotype in .fam file 
The final column of the .fam file corresponds to the phenotype. To avoid any data leakage in the selection of SNPS, we are going to keep the same SNPs held out, as we are treating the 600 cases as our test set in initial trials. We do, however, want to change the objective from linear to logistic, aka change the final column in the .fam file from case/control to a regression variable.
```bash
mv data/Merged_Genotype/MergedGen.fam data/Merged_Genotype/MergedGen_logistic.fam  && \
python -c "import pandas as pd; df=pd.read_csv('data/Merged_Genotype/MergedGen_logistic.fam',sep='\t',header=None); mapper=pd.read_csv('data/TrainingData_PhenoCov_Release.txt',sep=' '); mapper=dict(zip(mapper['ID'],mapper['Response.deltaDAS'])); newpheno=df[0].replace(mapper).fillna(-9); df[5]=newpheno; df.to_csv('data/Merged_Genotype/MergedGen.fam',index=False,header=False,sep='\t')"
```

### Create co-variance file 
In order to train linear model with covariance, we need to create a file which contains the covariance variable, FID and IID. This is done as follows below. Note that we are treating the one-hot encoded drugs as individual covariates for now.
```bash
python -c "import pandas as pd; df=pd.read_csv('data/Merged_Genotype/MergedGen_logistic.fam',sep='\t',header=None); mapper=pd.read_csv('data/TrainingData_PhenoCov_Release.txt',sep=' '); mapper=dict(zip(mapper['ID'],mapper['baselineDAS'])); df=df.loc[:,:2] ;baseline=df[0].replace(mapper).fillna(-9); df[2]=baseline; df.to_csv('data/utils/covars.dat',index=False,header=False,sep='\t')"
```

### Run Linear Regression GWAS

For the case with Drug as a covariate, we will run as follows 
```bash
plink --bfile data/Merged_Genotype/MergedGen --linear --out das --covar data/utils/covars.dat --hide-covar
```
### Get Duplicate SNPs
```bash 
cut -f 2 das.assoc.linear | sort | uniq -d > das.dups
```

### Run clumping
```bash
plink --bfile data/Merged_Genotype/MergedGen --clump das.assoc.linear  --exclude das.dups --out das
```

### Get best SNPs
```bash
python -c 'import pandas as pd; import numpy as np; df=pd.read_csv("das.clumped", delimiter=r"\s+"); np.savetxt("metadata/SJU_plink/plink_LD_SNPs", df.SNP, fmt="%s")'
```

### Other
If running with co-variates, but the co-variates are not desired, you can remove them as follows:
```bash

```