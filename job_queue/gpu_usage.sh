# https://www.hpc.dtu.dk/?page_id=4976
nodestat -G gpua100
#BSUB -R "select[avx2]" 
# How to monitor my GPU jobs?
bnvtop JOBID
# transfer host: 
# transfer.gbar.dtu.dk

# get quota: 
getquota_zhome.sh
getquota_work3.sh
