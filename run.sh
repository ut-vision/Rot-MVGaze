


# cd /home/jqin/wk/Rot-MVGaze

# python main.py \
#     --mode test \
#     -out ./logs \
#     # --model_cfg_path 




cd /home/jqin/wk/Rot-MVGaze

# python main.py \
#     --exp_name 'xgaze_novel' \
#     --mode train \
#     -out ./logs \
#     --save_epoch 1

python main.py \
    --exp_name 'xgaze2mpiinv_novel' \
    --mode train \
    -out ./logs \
    --save_epoch 1 \
    --print_freq 10
    # --model_cfg_path 