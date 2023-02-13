shopt -s expand_aliases
alias plink=~/Downloads/plink_mac_20230116/plink 

allSNPS=('EIRA_304'                 
'TEAR_B2_filtered'       
'dream3'
'BRAGGSS'              
'ERA'                     
'dream1'                  
'newsample_recode_mind05'
'BRASS_TNF'                
'ReAct'                     
'dream2')

mergedNames=('merged1'
'merged2'
'merged3'
'merged4'
'merged5'
'merged6'
'merged7'
'merged8'
'merged9'
'merged10'
)

lastSNP='ABCoN'

for i in ${!allSNPS[@]}; do
    plink --bfile ${lastSNP} --bmerge ${allSNPS[$i]} --make-bed --out ${mergedNames[$i]} --exclude ../utils/exclude_SNPs.txt || echo "failed first merge"

    # if the flip file exists, flip the SNPS
    if [ -f ${mergedNames[$i]}-merge.missnp ]; then
        plink --bfile ${lastSNP} --flip ${mergedNames[$i]}-merge.missnp --make-bed --out ${mergedNames[$i]}-temp --exclude ../utils/exclude_SNPs.txt || echo "failed flip"
        plink --bfile ${mergedNames[$i]}-temp --bmerge ${allSNPS[$i]} --make-bed --out ${mergedNames[$i]} --exclude ../utils/exclude_SNPs.txt || echo "failed merged flip"
    fi

    lastSNP=${mergedNames[$i]}
done