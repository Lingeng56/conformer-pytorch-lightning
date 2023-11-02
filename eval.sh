export CUDA_VISIBLE_DEVICES="1"
vocab_path="vocab.txt"
ckpt_path="experiments/conformer-rnnt-ctc-960"
resume_checkpoint="last.ckpt"
data_config_path="exp/data_config.json"
cmvn_path="data/train-960/global_cmvn"
wenet_ckpt_path="wenet.pt"

mkdir -p $ckpt_path
cp $data_config_path $ckpt_path/

python src/main.py --max_epochs 1000 \
                   --random_seed 777 \
                   --data_config_path $data_config_path \
                   --vocab_path $vocab_path \
                   --input_dim 80 \
                   --kernel_size 15 \
                   --encoder_dim 256 \
                   --dropout 0.1 \
                   --attention_dropout 0.1 \
                   --pos_enc_dropout 0.1 \
                   --hidden_dim 2048 \
                   --max_len 5000 \
                   --self_attention_dropout 0.1 \
                   --src_attention_dropout 0.1 \
                   --ctc_weight 0.2 \
                   --attention_weight 0.15 \
                   --transducer_weight 0.8 \
                   --lsm_weight 0.1 \
                   --reverse_weight 0.3 \
                   --delay_penalty 0.0 \
                   --lm_only_scale 0.25 \
                   --am_only_scale 0.0 \
                   --num_heads 4 \
                   --grad_clip 4 \
                   --accum_grad 2 \
                   --encoder_num_layers 12 \
                   --decoder_num_layers 3 \
                   --checkpoint_path $ckpt_path \
                   --num_devices 1 \
                   --predictor_embed_size 256 \
                   --predictor_hidden_size 256 \
                   --predictor_dim 256 \
                   --predictor_embed_dropout 0.1 \
                   --predictor_num_layers 2 \
                   --join_dim 512 \
                   --lr 0.001 \
                   --warmup 25000 \
                   --cmvn_path $cmvn_path \
                   --resume_from $ckpt_path/$resume_checkpoint \
                   --use_relative \
                   --eval \
                   --resume \
#                   --streaming_eval \
#                   --wenet_ckpt_path $wenet_ckpt_path \
#                   --train \
#                   --train
#                   --resume
