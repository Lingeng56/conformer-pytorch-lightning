import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model import ASRModel
from src.dataset import CustomDataset
from torch.utils.data import DataLoader
from src.utils import collate_fn
from src.tokenizer import Tokenizer


def train(args):
    tokenizer = Tokenizer(args.vocab_path, args.max_len)
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

    checkpoint_callback = ModelCheckpoint(
        monitor='val_wer',
        dirpath=args.checkpoint_path,
        filename='{epoch}-{val_loss:.2f}-{val_wer:.2f}',
        save_last=True,
        save_top_k=5,
        mode='min',
    )

    trainer = pl.Trainer(
        devices=args.num_devices,
        accelerator='gpu',
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
        max_epochs=args.max_epochs,
        precision=32,
        enable_progress_bar=True
    )
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader)



def evaluate(args):
    pass


def main(args):
    if args.train:
        train(args)

    if args.eval:
        evaluate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Conformer-Pytorch-Lightning',
        description='Pytorch-Lightning Implementation of Conformer')
    parser.add_argument('--max_epochs', type=int, required=True, default=10)
    parser.add_argument('--train_data_dir', type=str, required=True)
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True, default=32)
    parser.add_argument('--eval_batch_size', type=int, required=True, default=32)
    parser.add_argument('--input_dim', type=int, required=True, default=256)
    parser.add_argument('--kernel_size', type=int, required=True, default=32)
    parser.add_argument('--encoder_dim', type=int, required=True, default=256)
    parser.add_argument('--dropout', type=float, required=True, default=0.3)
    parser.add_argument('--expansion_factor', type=float, required=True, default=0.2)
    parser.add_argument('--num_heads', type=int, required=True, default=4)
    parser.add_argument('--encoder_layer_num', type=int, required=True, default=4)
    parser.add_argument('--decoder_dim', type=int, required=True, default=32)
    parser.add_argument('--max_len', type=int, required=True, default=512)
    parser.add_argument('--lr', type=float, required=True, default=1e-5)
    parser.add_argument('--use_relative', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--num_devices', type=int, required=True, default=2)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    all_args = parser.parse_args()
    main(all_args)
