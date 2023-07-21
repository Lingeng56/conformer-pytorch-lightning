import torch
import torch.nn as nn
import pytorch_lightning as pl
from encoder import ConformerEncoder
from decoder import LSTMAttentionDecoder
from torchmetrics import WordErrorRate
from scheduler import TransformerLR


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
                 vocab_size,
                 tokenizer,
                 max_len,
                 decode_method,
                 beam_size=-1,
                 use_relative=False):
        super(ASRModel, self).__init__()
        self.encoder_dim = encoder_dim
        self.lr = 0.05 / (self.encoder_dim ** 0.5)
        self.encoder = ConformerEncoder(input_dim,
                                        kernel_size,
                                        encoder_dim,
                                        dropout,
                                        expansion_factor,
                                        num_heads,
                                        encoder_layer_nums,
                                        max_len,
                                        use_relative
                                        )
        self.attn_decoder = LSTMAttentionDecoder(encoder_dim, decoder_layer_nums, num_heads, dropout, vocab_size,
                                                 tokenizer, max_len)
        self.ctc_decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1)
        )
        self.encoder_criterion = nn.CTCLoss(blank=0, zero_infinity=False)
        self.decoder_criterion = nn.CrossEntropyLoss(ignore_index=0)
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
        ctc_probs = self.ctc_decoder(outputs)
        attn_probs = self.attn_decoder(outputs)
        preds = self.attn_decoder.decode(outputs, output_lengths, self.decode_method, self.beam_size)
        ctc_loss = self.encoder_criterion(ctc_probs.permute(1, 0, 2), targets, output_lengths, target_lengths)
        ce_loss = self.decoder_criterion(attn_probs.view(-1, attn_probs.shape[-1]), targets.view(-1))
        loss = 0.5 * ctc_loss + 0.5 * ce_loss
        self.train_metric.update(preds, sentences)
        wer = self.train_metric.compute()
        self.log('train_total_loss', loss, prog_bar=True, on_step=True)
        self.log('train_ctc_loss', ctc_loss, prog_bar=True, on_step=True)
        self.log('train_ce_loss', ce_loss, prog_bar=True, on_step=True)
        self.log('training_wer', wer, prog_bar=True, on_step=True)
        self.log('current_lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=True)

        if self.global_step % 100 == 0 and self.global_rank == 0:
            print('Preds: %s' % preds[0])
            print('Truth: %s' % sentences[0])
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, sentences = batch['inputs'], batch['input_lengths'], batch[
            'targets'], \
                                                                    batch['target_lengths'], batch['sentences']

        outputs, output_lengths = self.encoder(inputs, input_lengths)
        ctc_probs = self.ctc_decoder(outputs)
        attn_probs = self.attn_decoder(outputs)
        preds = self.attn_decoder.decode(outputs, output_lengths, self.decode_method, self.beam_size)
        ctc_loss = self.encoder_criterion(ctc_probs.permute(1, 0, 2), targets, output_lengths, target_lengths)
        ce_loss = self.decoder_criterion(attn_probs.view(-1, attn_probs.shape[-1]), targets.view(-1))
        loss = 0.5 * ctc_loss + 0.5 * ce_loss
        self.valid_metric.update(preds, sentences)
        self.log('val_total_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_ctc_loss', ctc_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_ce_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_wer', self.valid_metric, prog_bar=True, on_step=True, on_epoch=True)

        self.valid_step_preds.extend(preds)

        self.valid_step_targets.extend(sentences)


    def on_validation_epoch_end(self):
        with open('tmp-predictions.txt', 'w') as f:
            for pred, target in zip(self.valid_step_preds, self.valid_step_targets):
                f.write('%s\t%s\n' % (pred, target))

        self.valid_step_preds = []
        self.valid_step_targets = []

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     inputs, input_lengths = batch['inputs'], batch['input_lengths']
    #     outputs, output_lengths = self.encoder(inputs, input_lengths, self.decode_method, self.beam_size)
    #     preds = self.decoder.decode(outputs, output_lengths)
    #     return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
        noam_sched = TransformerLR(optimizer, self.encoder_dim, 10000, 5)
        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": noam_sched,
                "interval": "step"
            }
        }
        return config
