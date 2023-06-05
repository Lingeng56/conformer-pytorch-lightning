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
                 decoder_dim,
                 vocab_size,
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
        self.decoder = LSTMDecoder(encoder_dim, decoder_dim, vocab_size)
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

        self.log('training_loss', loss)
        self.log('word_error_rate', self.metric)




    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        pass

    def lr_scheduler_step(self, scheduler, metric) -> None:
        pass





