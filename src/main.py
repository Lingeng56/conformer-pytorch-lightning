import argparse
import pytorch_lightning as pl
import json
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from model import ASRModel
from dataset import CustomDataset
from torch.utils.data import DataLoader
from utils import load_vocabs
from processor import collate_fn


def train(args):
    pl.seed_everything(args.random_seed)

    data_config = json.load(open(args.data_config_path, 'r'))
    data_config['vocabs'], vocab_size = load_vocabs(args.vocab_path)
    cv_config = data_config.copy()
    cv_config['sort'] = False
    cv_config['shuffle'] = False
    cv_config['speed_perturb'] = False
    cv_config['spec_aug'] = False
    cv_config['batch'] = False
    train_dataset = CustomDataset(data_config, mode='train')
    test_dataset = CustomDataset(cv_config, mode='dev')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=None)
    # num_workers=args.num_workers,
    # pin_memory=args.pin_memory,
    # prefetch_factor=args.prefetch)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=cv_config['batch_size'],
                                 # num_workers=args.num_workers,
                                 # pin_memory=args.pin_memory,
                                 # prefetch_factor=args.prefetch,
                                 collate_fn=collate_fn)

    model = ASRModel(input_dim=args.input_dim,
                     kernel_size=args.kernel_size,
                     encoder_dim=args.encoder_dim,
                     dropout=args.dropout,
                     expansion_factor=args.expansion_factor,
                     num_heads=args.num_heads,
                     encoder_layer_nums=args.encoder_layer_nums,
                     decoder_layer_nums=args.decoder_layer_nums,
                     vocab_size=vocab_size,
                     target_max_len=data_config['token_max_length'],
                     use_relative=args.use_relative,
                     join_dim=args.join_dim,
                     predictor_embed_size=args.predictor_embed_size,
                     predictor_hidden_size=args.predictor_hidden_size,
                     predictor_output_size=args.predictor_output_size,
                     predictor_embed_dropout=args.predictor_embed_dropout,
                     predictor_num_layers=args.predictor_num_layers,
                     lr=args.lr,
                     warmup=args.warmup
                     )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_total_loss',
        dirpath=args.checkpoint_path,
        filename='{epoch}-{val_total_loss:.2f}-{val_rnnt_loss:.2f}-{val_ctc_loss:.2f}-{val_ce_loss:.2f}',
        save_last=True,
        save_top_k=5,
        mode='min',
    )
    trainer = pl.Trainer(
        devices=args.num_devices,
        accelerator='gpu',
        callbacks=[checkpoint_callback,
                   RichModelSummary(),
                   RichProgressBar()],
        check_val_every_n_epoch=None,
        val_check_interval=2000,
        max_epochs=args.max_epochs,
        enable_progress_bar=True,
        gradient_clip_val=10,
        num_sanity_val_steps=2
    )

    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=test_dataloader,
                ckpt_path=args.resume_from if args.resume else None)


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
    parser.add_argument('--random_seed', type=int, required=True, default=3407)
    parser.add_argument('--train_batch_size', type=int, required=True, default=16)
    parser.add_argument('--eval_batch_size', type=int, required=True, default=16)
    parser.add_argument('--data_config_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--input_dim', type=int, required=True, default=256)
    parser.add_argument('--kernel_size', type=int, required=True, default=32)
    parser.add_argument('--encoder_dim', type=int, required=True, default=256)
    parser.add_argument('--dropout', type=float, required=True, default=0.3)
    parser.add_argument('--expansion_factor', type=int, required=True, default=2)
    parser.add_argument('--num_heads', type=int, required=True, default=4)
    parser.add_argument('--encoder_layer_nums', type=int, required=True, default=4)
    parser.add_argument('--decoder_layer_nums', type=int, required=True, default=4)
    parser.add_argument('--predictor_embed_size', type=int, required=True, default=32)
    parser.add_argument('--predictor_hidden_size', type=int, required=True, default=32)
    parser.add_argument('--predictor_output_size', type=int, required=True, default=32)
    parser.add_argument('--predictor_embed_dropout', type=float, required=True, default=0.1)
    parser.add_argument('--predictor_num_layers', type=int, required=True, default=32)
    parser.add_argument('--join_dim', type=int, required=True, default=32)
    parser.add_argument('--use_relative', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--num_devices', type=int, required=True, default=2)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--prefetch', type=int, default=100)
    parser.add_argument('--accumulate_batches', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=25000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_from', type=str)
    all_args = parser.parse_args()
    main(all_args)
