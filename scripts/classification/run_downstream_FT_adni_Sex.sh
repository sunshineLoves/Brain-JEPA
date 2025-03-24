python downstream_eval.py \
    --downstream_task fine_tune \
    --task classification \
    --batch_size 8 \
    --nb_classes 2 \
    --num_seed 5 \
    --load_epoch 300 \
    --epochs 50 \
    --blr 0.00002 \
    --min_lr 0.00000005 \
    --smoothing 0.0 \
    --config configs/downstream/fine_tune.yaml \
    --output_root './output_dir' \
    --model_name vit_base \
    --data_make_fn adni_dx \
    --load_path pretrained \
    --use_normalization \
    --crop_size 450,160 \
    --patch_size 16 \
    --pred_depth 12 \
    --pred_emb_dim 384 \
    --attn_mode flash_attn \
    --add_w mapping \
    --downsample \
    --device cuda \
    --processed_dir data/processed/adni/preproc-fmriprep_label-Sex_desc-full_bold.npz
