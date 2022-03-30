#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -par 28 -pe impi 28
##$ -mods l_hard h_vmem 7G #Max is 384G for pcc-skl 40 cores/node?
##$ -mods l_hard mem_req 7G
#$ -jc pcc-normal  # pcc-skl  #pcc-normal #pcc-skl.72h #pcc-normal.72h-----------------------------------------------------------------------

. /fefs/opt/x86_64/Gaussian/envset.sh
ulimit -s unlimited

source $HOME/.bashrc
source activate chemtsv2

python misc/qsub_parallel_job.py $1

