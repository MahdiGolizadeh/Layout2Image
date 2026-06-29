
export MODEL_DIR="./MODEL_BASE/stable-diffusion-v1-5"
export OUTPUT_DIR="./models_out"

CUDA_VISIBLE_DEVICES=0 \
/home/jovyan/miniconda3_hico/bin/accelerate launch --config_file ./utils/accelerate_config.yaml train_hico.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --train_data_yaml="utils/dataset/latent_LayoutDiffusion_large_grit.yaml" \
    --dataset_name=coco \
    --mixed_precision="fp16" \
    --resolution=512 \
    --learning_rate=1e-5 \

