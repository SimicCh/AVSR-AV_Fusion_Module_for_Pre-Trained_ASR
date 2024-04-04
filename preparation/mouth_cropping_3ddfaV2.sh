#!/bin/bash
#SBATCH --job-name=mouth_cropping_3ddfaV2 # Kurzname des Jobs
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
CACHE_DIR=/nfs/scratch/staff/$USER/.cache
export PIP_CACHE_DIR=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
mkdir -p CACHE_DIR
########################################################
############### Starte eigenen Job hier ################
source /nfs/scratch/staff/simicch/02_speechrec/11a_AV_Fusion_clean/venv/bin/activate

# Example configuration
prep_dir="/nfs/scratch/staff/simicch/00_data/02_LRS3/prep_test_repo"
landmark_dir="/nfs/scratch/staff/simicch/00_data/02_LRS3/prep_test_repo/landmarks_3ddfaV2"
video_dir="/nfs/data/LRS3_Dataset"
out_dir="/nfs/scratch/staff/simicch/00_data/02_LRS3/prep_test_repo/cropped_videos_3ddfaV2"
file_list="file.list.pretrain"
meanface_path="./models/20words_mean_face.npy"
rank=9
nshard=10
start_idx=48
stop_idx=68
window_margin=12
crop_height=96
crop_width=96






echo "start"
echo "mouth_cropping_3ddfaV2.py"
echo ""
srun python mouth_cropping_3ddfaV2.py \
	--prep_dir ${prep_dir}  \
	--landmark_dir ${landmark_dir}  \
	--video_dir ${video_dir}  \
	--out_dir ${out_dir}  \
	--file_list ${file_list}  \
	--meanface_path ${meanface_path}  \
	--rank ${rank}  \
	--nshard ${nshard}  \
	--start_idx ${start_idx}  \
	--stop_idx ${stop_idx}  \
	--window_margin ${window_margin}  \
	--crop_height ${crop_height}  \
	--crop_width ${crop_width}


