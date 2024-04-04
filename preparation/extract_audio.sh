#!/bin/bash
#SBATCH --job-name=extract_audio # Kurzname des Jobs
#SBATCH --nodes=1 # Anzahl benötigter Knoten
#SBATCH --ntasks=1 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --partition=p1 # Verwendete Partition (z.B. p0, p1, p2 oder all)
#SBATCH --time=24:00:00 # Gesamtlimit für Laufzeit des Jobs (Format: HH:MM:SS)
#SBATCH --cpus-per-task=2 # Rechenkerne pro Task
#SBATCH --mem=6G # Gesamter Hauptspeicher pro Knoten
#SBATCH --gres=gpu:0 # Gesamtzahl GPUs pro Knoten
echo "=================================================================="
echo "Starting Batch Job at $(date)"
echo "Job submitted to partition ${SLURM_JOB_PARTITION} on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "Requested ${SLURM_CPUS_ON_NODE} CPUs on compute node $(hostname)"
echo "Working directory: $(pwd)"
echo "=================================================================="
###################### Optional for Pythonnutzer*innen #######################
# Die folgenden Umgebungsvariablen stellen sicher, dass
# Modellgewichte von Huggingface und PIP Packages nicht unter
# /home/$USER/.cache landen.
CACHE_DIR=/net/ml1/mnt/md0/scratch/staff/$USER/.cache
export PIP_CACHE_DIR=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
mkdir -p CACHE_DIR
########################################################
############### Starte eigenen Job hier ################
source /nfs/scratch/staff/simicch/02_speechrec/11a_AV_Fusion_clean/venv/bin/activate


# Configuration
lrs3="/nfs/data/LRS3_Dataset"
prep_dir="/nfs/scratch/staff/simicch/00_data/02_LRS3/prep_test_repo"
file_list="file.list.pretrain"
out_dir="/nfs/scratch/staff/simicch/00_data/02_LRS3/prep_test_repo/audio"
rank=7
nshard=8


echo "start"
echo "extract_audio.py"
echo ""
srun python extract_audio.py \
	--lrs3 ${lrs3}  \
	--prep_dir ${prep_dir}  \
	--file_list ${file_list}  \
	--out_dir ${out_dir}  \
	--rank ${rank}  \
	--nshard ${nshard} 


