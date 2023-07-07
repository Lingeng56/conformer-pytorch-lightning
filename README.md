# conformer-pytorch-lightning
Implement conformer with pytorch lightning

# Requirements
- pytorch
- torchaudio
- torchmetrics
- pytorch-lightning

# Dataset
``` python
from torchaudio.datasets import LIBRISPEECH


 train_dataset = LIBRISPEECH(root='data', url='train-clean-100', download=True)
 test_dataset = LIBRISPEECH(root='data', url='test-clean', download=True)
```

# Training

```shell
bash train.sh
```

# Evaluation

In order to save time, I directly used the test dataset when doing validation after each epoch, 
so I just looked at the training log directly to get evaluation results.


# Inference

**TO BE CONTINUE...**


# Experiments Result

**TO BE CONTINUE...**


# Problems

1. Num of model's parameters don't match with paper
2. Model can't converge well...


# TODO

1. Finding bugs allows the model to train properly for convergence
2. Finish inference script
3. Reproduce the results in the paper, at least close
4. Complete the ablation experiment in the paper
