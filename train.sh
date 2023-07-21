export CUDA_VISIBLE_DEVICES="0"
data_dir="data"
vocab_path="vocab.txt"
ckpt_path="experiments/conformer-attention-lstm"
bpe_path="bpe_model.model"
train_part="train-clean-100"
test_part="test-clean"
dev_part="dev-clean"
resume_checkpoint="last.ckpt"

mkdir -p $ckpt_path

python src/main.py --max_epochs 1000 \
                   --train_data_list_path $data_dir/$train_part/data.list \
                   --test_data_list_path $data_dir/$test_part/data.list \
                   --dev_data_list_path $data_dir/$dev_part/data.list \
                   --vocab_path $vocab_path \
                   --bpe_path $bpe_path \
                   --train_batch_size 16 \
                   --eval_batch_size 16 \
                   --input_dim 80 \
                   --kernel_size 31 \
                   --encoder_dim 144 \
                   --dropout 0.1 \
                   --expansion_factor 2 \
                   --num_heads 4 \
                   --encoder_layer_nums 16 \
                   --decoder_layer_nums 1 \
                   --decoder_dim 320 \
                   --max_len 512 \
                   --checkpoint_path $ckpt_path \
                   --num_devices 1 \
                   --decode_method 'greedy' \
                   --beam_size 2 \
                   --train \
                   --resume \
                   --resume_from $ckpt_path/$resume_checkpoint