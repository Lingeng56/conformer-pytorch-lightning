data_dir="/home/wuliu/datasets/LibriSpeech"
save_dir="./data"
vocab_path="vocab.txt"
bpe_part="train-960"
bpe_model_prefix="bpe_model"
bpe_type="bpe"
vocab_size=5000

mkdir -p $data_dir


python utils/collect_librispeech.py --data_dir $data_dir \
                                    --save_dir $save_dir \
                                    --parts dev-other test-other



#spm_train --input=$save_dir/$bpe_part/transcripts.txt \
#          --model_prefix=$bpe_model_prefix \
#          --vocab_size=$vocab_size \
#          --model_type=$bpe_type
#
#python utils/convert_vocab.py $vocab_path


#spm_export_vocab --model=$bpe_model_prefix.model > $vocab_path
