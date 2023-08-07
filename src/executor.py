import json
import pytorch_lightning as pl
from encoder import ConformerEncoder
from decoder import CTCDecoder, BiTransformerDecoder
from joint import TransducerJoint
from model import Transducer
from predictor import Predictor
from utils import load_vocabs
from torch.utils.data import DataLoader
from dataset import CustomDataset
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar


class Executor:

    def __init__(self, args):
        pl.seed_everything(args.random_seed)
        self.__build_model(args)
        self.__build_dataloader(args)
        self.__build_trainer(args)
        self.args = args

    def __build_model(self, args):
        data_config = json.load(open(args.data_config_path, 'r'))
        vocabs, vocab_size = load_vocabs(args.vocab_path)
        data_config['vocabs'] = vocabs

        conformer_encoder = ConformerEncoder(
            input_dim=args.input_dim,
            kernel_size=args.kernel_size,
            encoder_dim=args.encoder_dim,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            pos_enc_dropout=args.pos_enc_dropout,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            encoder_num_layers=args.encoder_num_layers,
            max_len=args.max_len,
            use_relative=args.use_relative
        )
        ctc_decoder = CTCDecoder(
            vocab_size=vocab_size,
            encoder_dim=args.encoder_dim,
            dropout=args.dropout
        )

        attn_decoder = BiTransformerDecoder(
            vocab_size=vocab_size,
            decoder_dim=args.encoder_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_layers=args.decoder_num_layers,
            r_num_layers=args.decoder_num_layers,
            dropout=args.dropout,
            pos_enc_dropout=args.pos_enc_dropout,
            self_attention_dropout=args.self_attention_dropout,
            src_attention_dropout=args.src_attention_dropout
        )

        predictor = Predictor(vocab_size=vocab_size,
                              embed_size=args.predictor_embed_size,
                              output_size=args.predictor_dim,
                              hidden_size=args.predictor_hidden_size,
                              embed_dropout=args.predictor_embed_dropout,
                              num_layers=args.predictor_num_layers,
                              )

        joint = TransducerJoint(vocab_size=vocab_size,
                                enc_output_size=args.encoder_dim,
                                pred_output_size=args.predictor_dim,
                                join_dim=args.join_dim)

        self.model = Transducer(
            encoder=conformer_encoder,
            predictor=predictor,
            joint=joint,
            attention_decoder=attn_decoder,
            ctc=ctc_decoder,
            vocab_size=vocab_size,
            blank=vocabs['<blank>'],
            sos=vocabs['<bos>'],
            eos=vocabs['<eos>'],
            ignore_id=vocabs['<blank>'],
            ctc_weight=args.ctc_weight,
            reverse_weight=args.reverse_weight,
            lsm_weight=args.lsm_weight,
            transducer_weight=args.transducer_weight,
            attention_weight=args.attention_weight,
            delay_penalty=args.delay_penalty,
            warmup_steps=args.warmup_steps,
            lm_only_scale=args.lm_only_scale,
            am_only_scale=args.am_only_scale,
            lr=args.lr
        )

    def __build_dataloader(self, args):
        data_config = json.load(open(args.data_config_path, 'r'))
        data_config['vocabs'], vocab_size = load_vocabs(args.vocab_path)
        cv_config = data_config.copy()
        cv_config['sort'] = False
        cv_config['shuffle'] = False
        cv_config['speed_perturb'] = False
        cv_config['spec_aug'] = False
        train_dataset = CustomDataset(data_config, mode='train')
        test_dataset = CustomDataset(cv_config, mode='dev')
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=None,
                                           # num_workers=args.num_workers,
                                           # pin_memory=args.pin_memory,
                                           # prefetch_factor=args.prefetch,
                                           )
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=None,
                                          # num_workers=args.num_workers,
                                          # pin_memory=args.pin_memory,
                                          # prefetch_factor=args.prefetch,
                                          )

    def __build_trainer(self, args):
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=args.checkpoint_path,
            filename='{epoch}-{val_loss:.2f}',
            save_last=True,
            save_top_k=5,
            mode='min',
        )

        self.trainer = pl.Trainer(
            devices=args.num_devices,
            accelerator='gpu',
            callbacks=[checkpoint_callback,
                       RichModelSummary(),
                       RichProgressBar()],
            check_val_every_n_epoch=None,
            val_check_interval=2000,
            max_epochs=args.max_epochs,
            enable_progress_bar=True,
            num_sanity_val_steps=2,
            gradient_clip_val=args.grad_clip,
            accumulate_grad_batches=args.accum_grad,
            precision=16,
            strategy="deepspeed_stage_1"
        )

    def train(self):
        self.trainer.fit(model=self.model,
                         train_dataloaders=self.train_dataloader,
                         val_dataloaders=self.test_dataloader,
                         ckpt_path=self.args.resume_from if self.args.resume else None)

    def eval(self):
        pass
