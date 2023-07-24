import torch
import torch.nn as nn
import pytorch_lightning as pl
from encoder import ConformerEncoder
from decoder import LSTMAttentionDecoder
from torchmetrics import WordErrorRate
from scheduler import TransformerLR
from joint import TransducerJoint
from predictor import Predictor
from k2 import rnnt_loss
from utils import build_joint_text


class ASRModel(pl.LightningModule):

    def __init__(self,
                 input_dim,
                 kernel_size,
                 encoder_dim,
                 join_dim,
                 dropout,
                 expansion_factor,
                 num_heads,
                 encoder_layer_nums,
                 decoder_layer_nums,
                 vocab_size,
                 max_len,
                 predictor_embed_size,
                 predictor_hidden_size,
                 predictor_output_size,
                 predictor_embed_dropout,
                 predictor_num_layers,
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
                                                 max_len)
        self.ctc_decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, vocab_size),
            nn.LogSoftmax(dim=-1)
        )
        self.predictor = Predictor(
            vocab_size=vocab_size,
            embed_size=predictor_embed_size,
            output_size=predictor_output_size,
            embed_dropout=predictor_embed_dropout,
            num_layers=predictor_num_layers,
            hidden_size=predictor_hidden_size,
            dropout=dropout,
        )
        self.joint_network = TransducerJoint(
            vocab_size=vocab_size,
            enc_output_size=encoder_dim,
            pred_output_size=predictor_output_size,
            join_dim=join_dim
        )
        self.encoder_criterion = nn.CTCLoss(blank=0, zero_infinity=False, reduction='sum')
        self.decoder_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    def training_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, sentences = batch['inputs'], batch['input_lengths'], batch[
            'targets'], batch['target_lengths'], batch['sentences']
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        ctc_probs = self.ctc_decoder(outputs)
        attn_probs = self.attn_decoder(outputs)
        joint_text = build_joint_text(targets, 0)
        predictor_out = self.predictor(joint_text)
        joint_out = self.joint_network(outputs, predictor_out)
        rnnt_loss_ = rnnt_loss(logits=joint_out,
                               symbols=targets,
                               termination_symbol=0,
                               reduction="mean")
        ctc_loss = self.encoder_criterion(ctc_probs.permute(1, 0, 2), targets, output_lengths, target_lengths)
        ce_loss = self.decoder_criterion(attn_probs.view(-1, attn_probs.shape[-1]), targets.view(-1))
        loss = 0.1 * ctc_loss + 0.15 * ce_loss + 0.75 * rnnt_loss_
        self.log('train_total_loss', loss, prog_bar=True, on_step=True)
        self.log('train_ctc_loss', ctc_loss, prog_bar=True, on_step=True)
        self.log('train_ce_loss', ce_loss, prog_bar=True, on_step=True)
        self.log('train_rnnt_loss', rnnt_loss_, prog_bar=True, on_step=True)
        self.log('current_lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths, sentences = batch['inputs'], batch['input_lengths'], batch[
            'targets'], \
                                                                    batch['target_lengths'], batch['sentences']

        outputs, output_lengths = self.encoder(inputs, input_lengths)
        ctc_probs = self.ctc_decoder(outputs)
        attn_probs = self.attn_decoder(outputs)
        joint_text = build_joint_text(targets, 0)
        predictor_out = self.predictor(joint_text)
        joint_out = self.joint_network(outputs, predictor_out)
        rnnt_loss_ = rnnt_loss(logits=joint_out,
                               symbols=targets,
                               termination_symbol=0,
                               reduction="mean")
        ctc_loss = self.encoder_criterion(ctc_probs.permute(1, 0, 2), targets, output_lengths, target_lengths)
        ce_loss = self.decoder_criterion(attn_probs.view(-1, attn_probs.shape[-1]), targets.view(-1))
        loss = 0.1 * ctc_loss + 0.15 * ce_loss + 0.75 * rnnt_loss_
        self.log('val_total_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_ctc_loss', ctc_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_ce_loss', ce_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_rnnt_loss', rnnt_loss_, prog_bar=True, on_step=True, on_epoch=True)



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
