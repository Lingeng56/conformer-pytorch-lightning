export CUDA_VISIBLE_DEVICES="1"
data_dir="data"
vocab_path="vocab.txt"
ckpt_path="experiments/conformer-small-lr1e-5-greedy-v1"

mkdir -p ckpt_path

python src/main.py --max_epochs 10 \
                   --train_data_dir $data_dir \
                   --test_data_dir $data_dir \
                   --train_url "train-clean-100" \
                   --test_url "test-clean" \
                   --vocab_path $vocab_path \
                   --train_batch_size 4 \
                   --eval_batch_size 4 \
                   --input_dim 40 \
                   --kernel_size 31 \
                   --encoder_dim 144 \
                   --dropout 0.3 \
                   --expansion_factor 2 \
                   --num_heads 4 \
                   --encoder_layer_nums 16 \
                   --decoder_layer_nums 1 \
                   --decoder_dim 320 \
                   --max_len 512 \
                   --lr 1e-5 \
                   --checkpoint_path $ckpt_path \
                   --num_devices 1 \
                   --train