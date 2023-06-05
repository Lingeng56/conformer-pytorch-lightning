import torch
import torch.nn as nn
import pytorch_lightning as pl
from layers.encoder import ConformerEncoder
from layers.decoder import LSTMDecoder
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
                 decoder_dim,
                 vocab_size,
                 tokenizer,
                 max_len,
                 lr,
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
                                        use_relative)
        self.decoder = LSTMDecoder(encoder_dim, decoder_dim, vocab_size, tokenizer)
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=False)
        self.metric = WordErrorRate()
        self.lr = lr




    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch['inputs'], batch['input_lengths'], batch['targets'], batch['target_lengths']
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        probs = self.decoder(outputs)
        loss = self.criterion(probs, targets, output_lengths, target_lengths)
        self.log('training_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, sentences = batch['inputs'], batch['input_lengths'], batch['targets'], \
                                                         batch['target_lengths'], batch['sentences']
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        probs = self.decoder(outputs)
        preds = self.decoder.decode(probs)
        loss = self.criterion(probs, targets, output_lengths, target_lengths)
        self.metric.update(preds, sentences)

        self.log('val_loss', loss)
        self.log('word_error_rate', self.metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer
            # "lr_scheduler": {
            #     "scheduler": ReduceLROnPlateau(optimizer, ...),
            #     "monitor": "metric_to_track",
            #     "frequency": "indicates how often the metric is updated"
            #     # If "monitor" references validation metrics, then "frequency" should be set to a
            #     # multiple of "trainer.check_val_every_n_epoch".
            # },
        }





