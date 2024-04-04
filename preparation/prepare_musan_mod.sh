#!/bin/bash
#SBATCH --job-name=prepare_musan_mod # Kurzname des Jobs
#SBATCH --nodes=1 # Anzahl benötigter Knoten
#SBATCH --ntasks=1 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --partition=p1 # Verwendete Partition (z.B. p0, p1, p2 oder all)
#SBATCH --time=24:00:00 # Gesamtlimit für Laufzeit des Jobs (Format: HH:MM:SS)
#SBATCH --cpus-per-task=2 # Rechenkerne pro Task
#SBATCH --mem=6G # Gesamter Hauptspeicher pro Knoten
#SBATCH --gres=gpu:0 # Gesamtzahl GPUs pro Knoten
#SBATCH --qos=gpubasic # Quality-of-Service
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
CACHE_DIR=/nfs/scratch/staff/$USER/.cache
export PIP_CACHE_DIR=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
mkdir -p CACHE_DIR
########################################################
############### Starte eigenen Job hier ################
source /nfs/scratch/staff/simicch/02_speechrec/11a_AV_Fusion_clean/venv/bin/activate

# Example configuration
musan="/nfs/data/musan"
prep_dir="/nfs/scratch/staff/simicch/00_data/03_musan/prep_repo2"


echo "start"
echo "prepare_musan_mod.py"
echo ""
srun python prepare_musan_mod.py \
	--musan ${musan} \
	--prep_dir ${prep_dir}


