import torch
import torch.nn as nn
import pytorch_lightning as pl
from encoder import ConformerEncoder
from torchaudio.models import Conformer
from decoder import LSTMDecoder
from torchmetrics import WordErrorRate
from scheduler import TransformerLR
from pytorch_lightning.callbacks import LearningRateMonitor


class ASRModel(pl.LightningModule):

    def __init__(self,
                 input_dim,
                 kernel_size,
                 encoder_dim,
                 dropout,
                 expansion_factor,
                 num_heads,
                 encoder_layer_nums,
                 decoder_layer_nums,
                 decoder_dim,
                 vocab_size,
                 tokenizer,
                 max_len,
                 decode_method,
                 beam_size=-1,
                 use_relative=False):
        super(ASRModel, self).__init__()
        # self.encoder = ConformerEncoder(input_dim,
        #                                 kernel_size,
        #                                 encoder_dim,
        #                                 dropout,
        #                                 expansion_factor,
        #                                 num_heads,
        #                                 encoder_layer_nums,
        #                                 max_len,
        #                                 use_relative
        #                                 )
        self.encoder = Conformer(input_dim=input_dim,
                                 num_heads=num_heads,
                                 ffn_dim=encoder_dim,
                                 depthwise_conv_kernel_size=kernel_size,
                                 dropout=dropout,
                                 num_layers=encoder_layer_nums)
            
        self.encoder_dim = encoder_dim
        self.lr = 0.05 / (self.encoder_dim ** 0.5)
        self.decoder = LSTMDecoder(input_dim, decoder_dim, decoder_layer_nums, vocab_size, tokenizer)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=False)
        self.train_metric = WordErrorRate()
        self.valid_metric = WordErrorRate()
        self.decode_method = decode_method
        self.beam_size = beam_size
        self.valid_step_preds = []
        self.valid_step_targets = []

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, sentences = batch['inputs'], batch['input_lengths'], batch[
            'targets'], batch['target_lengths'], batch['sentences']
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        probs = self.decoder(outputs)
        preds = self.decoder.decode(outputs, output_lengths, self.decode_method, self.beam_size)
        loss = self.criterion(probs.permute(1, 0, 2), targets, output_lengths, target_lengths)
        self.train_metric.update(preds, sentences)
        wer = self.train_metric.compute()
        self.log('training_loss', loss, prog_bar=True, on_step=True)
        self.log('training_wer', wer, prog_bar=True, on_step=True)
        self.log('current_lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, sentences = batch['inputs'], batch['input_lengths'], batch[
            'targets'], \
                                                                    batch['target_lengths'], batch['sentences']

        outputs, output_lengths = self.encoder(inputs, input_lengths)
        probs = self.decoder(outputs)
        preds = self.decoder.decode(outputs, output_lengths, self.decode_method, self.beam_size)
        loss = self.criterion(probs.permute(1, 0, 2), targets, output_lengths, target_lengths)
        self.valid_metric.update(preds, sentences)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_wer', self.valid_metric, prog_bar=True, on_step=True, on_epoch=True)
        

        self.valid_step_preds.extend(preds)
        self.valid_step_targets.extend(targets.tolist())

    def on_validation_epoch_end(self):
        with open('tmp-predictions.txt', 'w') as f:
            for pred, target in zip(self.valid_step_preds, self.valid_step_targets):
                f.write('%s\t%s\n' % (pred, target))

        self.valid_step_preds = []
        self.valid_step_targets = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, input_lengths = batch['inputs'], batch['input_lengths']
        outputs, output_lengths = self.encoder(inputs, input_lengths, self.decode_method, self.beam_size)
        preds = self.decoder.decode(outputs, output_lengths)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
        noam_sched = TransformerLR(optimizer, self.encoder_dim, 10000, 5)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": noam_sched,
                "interval": "step"
            }
        }
        return config

    
