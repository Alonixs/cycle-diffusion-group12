export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0
export RUN_NAME=experiment1
export SEED=42

for IMG_TYPE in org low high
do
   export OUTPUT_DIR="${RUN_NAME}_${SEED}_${IMG_TYPE}";

   python -m torch.distributed.launch --nproc_per_node 1 --use-env --master_port 1446 main.py --seed $SEED --cfg experiments/$RUN_NAME.cfg \
        --run_name $RUN_NAME$SEED --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 50 \
        --metric_for_best_model CLIPEnergy --greater_is_better false --save_strategy steps --save_steps 50 --save_total_limit 1 --load_best_model_at_end \
        --gradient_accumulation_steps 4 --num_train_epochs 0 --adafactor false --learning_rate 1e-3 --do_eval --output_dir output/$OUTPUT_DIR \
        --overwrite_output_dir --img_type $IMG_TYPE --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --eval_accumulation_steps 4 \
        --ddp_find_unused_parameters true --verbose true --disable_wandb True --save_images True
done
