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

