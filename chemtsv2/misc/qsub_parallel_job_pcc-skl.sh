#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -par 40 -pe impi 40
##$ -mods l_hard h_vmem 8G #Max is 384G for pcc-skl 40 cores/node?
##$ -mods l_hard mem_req 8G
#$ -jc pcc-skl  # pcc-skl  #pcc-normal #pcc-skl.72h #pcc-normal.72h-----------------------------------------------------------------------

. /fefs/opt/x86_64/Gaussian/envset.sh
ulimit -s unlimited

source $HOME/.bashrc
source activate chemtsv2

python misc/qsub_parallel_job.py $1

