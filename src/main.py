import argparse
import pytorch_lightning as pl
from model import ASRModel
from dataset import CustomDataset
from torch.utils.data import DataLoader
from utils import collate_fn
from tokenizer import Tokenizer


def train(args):
    tokenizer = Tokenizer()
    train_dataset = CustomDataset(data_dir=args.train_data_dir, url=args.train_url, tokenizer=tokenizer)
    test_dataset = CustomDataset(data_dir=args.test_data_dir, url=args.test_url, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    model = ASRModel(input_dim=args.input_dim,
                     kernel_size=args.kernel_size,
                     encoder_dim=args.encoder_dim,
                     dropout=args.dropout,
                     expansion_factor=args.expansion_factor,
                     num_heads=args.num_heads,
                     encoder_layer_nums=args.encoder_layer_nums,
                     decoder_dim=args.decoder_dim,
                     vocab_size=tokenizer.vocab_size,
                     tokenizer=tokenizer,
                     max_len=args.max_len,
                     lr=args.lr,
                     use_relative=args.use_relative)

    trainer = pl.Trainer(

    )
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader,
                ckpt_path=args.ckpt_path)



def main(args):
    pass
