# conformer-pytorch-lightning
Implement conformer-ctc-rnnt with pytorch lightning

# Requirements
- pytorch
- torchaudio
- torchmetrics
- pytorch-lightning
- wandb

# Pipeline
1. Data Preprocessing
2. Training
3. Evaluation
4. Inference


# Data Preprocessing
1. Directories Structure
2. Data Format
3. Build Vocabulary
4. Compute CMVN

## Directories Structure
```shell
--data
      --train-960
                --wav_paths.txt
                --transcripts.txt
                --data.list
      --dev-clean
                --wav_paths.txt
                --transcripts.txt
                --data.list
      --test-clean
                --wav_paths.txt
                --transcripts.txt
                --data.list
      ...
```



## Data Format
wav_paths.txt
```text
wav_path_0
wav_path_1
...
wav_path_N
```

transcripts.txt
```text
transcript_0
transcript_2
...
transcript_N
```

data.list
```text
# Each line is a json object
JSON({'key': key_0, 'wav_path': wav_path_0, 'transcript': transcript_0})
JSON({'key': key_1, 'wav_path': wav_path_1, 'transcript': transcript_1})
...
JSON({'key': key_N, 'wav_path': wav_path_N, 'transcript': transcript_N})
```



## Build Vocabulary
### For LibriSpeech
Train bpe model
```shell
spm_train --input=whole_text.txt \
          --vocab_size=5000 \
          --model_type=unigram \
          --model_prefix=bpemodel \
          --input_sentence_size=10000000
```
Encode all texts
```shell
spm_encode --model=bpemodel.model \
           --output_format=piece < whole_text.txt > tmp_vocab.txt

```
Add `<blank>` and `<unk>` to vocab.txt
```shell
echo "<blank> 0" >> vocab.txt
echo "<unk> 1" >> vocab.txt
```
Build vocabulary
```shell
cat tmp_vocab.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> vocab.txt
```
Add `<sos/eos>` to vocab.txt
```shell
num_token=$(cat vocab.txt | wc -l)
echo "<sos/eos> $num_token" >> vocab.txt
```
## Compute CMVN

```shell
python utils/compute_cmvn_stats.py --num_workers 16 \
    --feature_dim 80 \
    --resample_rate 16000 \
    --in_scp data/train-960/wav_paths.txt \
    --out_cmvn data/train-960/global_cmvn
```


# Training

```shell
bash train.sh
```

# Evaluation
```shell 
bash eval.sh
```


# Experiments Result
| EVAL SET   | WER  |
|------------|------|
| dev-clean  | 2.5% |
| dev-other  | 9.2% |
| test-clean | 3.7% |
| test-other | 9.7% |