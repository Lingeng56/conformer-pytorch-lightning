export CUDA_VISIBLE_DEVICES="1"
data_dir="data"
vocab_path="vocab.txt"
ckpt_path="experiments/conformer-small-lr5e-4-greedy-v1"

mkdir -p ckpt_path

python src/main.py --max_epochs 1000 \
                   --train_data_dir $data_dir \
                   --test_data_dir $data_dir \
                   --train_url "train-clean-100" \
                   --test_url "test-clean" \
                   --vocab_path $vocab_path \
                   --train_batch_size 8 \
                   --eval_batch_size 8 \
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
                   --lr 5e-4 \
                   --checkpoint_path $ckpt_path \
                   --num_devices 1 \
                   --decode_method 'greedy' \
                   --beam_size 2 \
                   --train