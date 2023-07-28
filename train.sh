export CUDA_VISIBLE_DEVICES="0,1"
vocab_path="vocab.txt"
ckpt_path="experiments/conformer-rnnt-wenet"
resume_checkpoint="last.ckpt"
data_config_path="exp/data_config.json"

mkdir -p $ckpt_path
cp $data_config_path $ckpt_path/

python src/main.py --max_epochs 1000 \
                   --random_seed 3407 \
                   --data_config_path $data_config_path \
                   --vocab_path $vocab_path \
                   --train_batch_size 4 \
                   --eval_batch_size 4 \
                   --input_dim 80 \
                   --kernel_size 31 \
                   --encoder_dim 256 \
                   --dropout 0.1 \
                   --expansion_factor 8 \
                   --num_heads 4 \
                   --encoder_layer_nums 12 \
                   --decoder_layer_nums 3 \
                   --checkpoint_path $ckpt_path \
                   --num_devices 1 \
                   --train \
                   --predictor_embed_size 256 \
                   --predictor_hidden_size 256 \
                   --predictor_output_size 256 \
                   --predictor_embed_dropout 0.1 \
                   --predictor_num_layers 2 \
                   --join_dim 512 \
                   --lr 1e-5 \
                   --warmup 25000 \
                   --accumulate_batches 4 \
                   --resume_from $ckpt_path/$resume_checkpoint \
                   --use_relative
#                   --resume