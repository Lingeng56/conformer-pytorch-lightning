# conformer-pytorch-lightning
Implement conformer-ctc-rnnt with pytorch lightning

# Requirements
- pytorch
- torchaudio
- torchmetrics
- pytorch-lightning
- wandb

# Data Preprocessing

## Collect Data List
```shell
python utils/collect_librispeech.py --data_dir [path of librispeech] --save_dir ./data --parts dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500
```

## Build train-960
```shell
cd data
mkdir -p train-960
cd train-960
cat ../train-clean-100/wav_paths.txt >> wav_paths.txt
cat ../train-clean-360/wav_paths.txt >> wav_paths.txt
cat ../train-other-500/wav_paths.txt >> wav_paths.txt
cat ../train-clean-100/transcripts.txt >> transcripts.txt
cat ../train-clean-360/transcripts.txt >> transcripts.txt
cat ../train-other-500/transcripts.txt >> transcripts.txt
cat ../train-clean-100/data.list >> data.list
cat ../train-clean-360/data.list >> data.list
cat ../train-other-500/data.list >> data.list
wc -l wav_paths.txt
# 281241 wav_paths.txt
```

## Build Vocabulary
```shell

```


# Training

```shell
bash train.sh
```

# Evaluation
```shell
bash eval.sh
```

# Inference

**TO BE CONTINUE...**


# Experiments Result
dev-clean 2.5%
dev-other 9.2%
test-clean 3.7%
test-other 9.7%
