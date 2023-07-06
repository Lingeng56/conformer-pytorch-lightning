import torch
import torch.nn as nn
import pytorch_lightning as pl
from encoder import ConformerEncoder
from decoder import LSTMDecoder
from torchmetrics import WordErrorRate



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
                 lr,
                 decode_method,
                 beam_size=-1,
                 use_relative=False):
        super(ASRModel, self).__init__()
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
        self.decoder = LSTMDecoder(encoder_dim, decoder_dim, decoder_layer_nums, vocab_size, tokenizer)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=False)
        self.metric = WordErrorRate()
        self.lr = lr
        self.decode_method = decode_method
        self.beam_size = beam_size




    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch['inputs'], batch['input_lengths'], batch['targets'], batch['target_lengths']
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        probs = self.decoder(outputs)
        loss = self.criterion(probs.permute(1, 0, 2), targets, output_lengths, target_lengths)
        self.log('training_loss', loss, prog_bar=True, on_step=True)
        return loss


    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, sentences = batch['inputs'], batch['input_lengths'], batch['targets'], \
                                                         batch['target_lengths'], batch['sentences']

        outputs, output_lengths = self.encoder(inputs, input_lengths)
        probs = self.decoder(outputs)
        preds = self.decoder.decode(outputs, output_lengths, self.decode_method, self.beam_size)
        loss = self.criterion(probs.permute(1, 0, 2), targets, output_lengths, target_lengths)
        self.metric.update(preds, sentences)

        self.log('val_loss', loss, prog_bar=True, on_step=True)
        self.log('val_wer', self.metric, prog_bar=True, on_step=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, input_lengths = batch['inputs'], batch['input_lengths']
        outputs, output_lengths = self.encoder(inputs, input_lengths, self.decode_method, self.beam_size)
        preds = self.decoder.decode(outputs, output_lengths)
        return preds


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer





