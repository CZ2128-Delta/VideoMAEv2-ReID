#!/bin/bash
#SBATCH --account=pas2136
#SBATCH --job-name=VideoMAEv2_ft
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liu.12122@osu.edu
#SBATCH --output=slurm_output/VideoMAEv2_ft_appearance.out.%j

set -x

DATA_PATH="/fs/scratch/PAS3184/Fangxun/_VideoMAEv2_dataset"
DATA_ROOT="/fs/scratch/PAS3184/draft/appearance"
MODEL_PATH="./model_zoo/vit_b_k710_dl_from_giant.pth"
OUTPUT_DIR="./output"

# module load miniconda3/24.1.2-py310
# conda activate videomae

cd /users/PAS2985/cz2128/ReID/VideoMAEv2
python run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set custom \
    --nb_classes 64 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --input_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_workers 1 \
    --opt adamw \
    --lr 2e-4 \
    --warmup_epochs 0 \
    --epochs 100 \
    --reid_embed_dim 512 \
    --reid_neck_feat after \
    --id_loss_weight 1.0 \
    --triplet_loss_weight 1.0 \
    --triplet_margin 0.3 \
    --use_triplet