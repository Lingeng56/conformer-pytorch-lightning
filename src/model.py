import torch
import torch.nn as nn
import pytorch_lightning as pl
import k2
from utils import add_blank, add_sos_eos, reverse_sequence
from scheduler import WarmupLR
from label_smoothing_loss import LabelSmoothingLoss


class Transducer(pl.LightningModule):

    def __init__(self,
                 encoder,
                 predictor,
                 joint,
                 attention_decoder,
                 ctc,
                 vocab_size=5002,
                 blank=0,
                 sos=2,
                 eos=3,
                 ignore_id=0,
                 ctc_weight=0.0,
                 reverse_weight=0.0,
                 lsm_weight=0.0,
                 transducer_weight=1.0,
                 attention_weight=0.0,
                 delay_penalty=0.0,
                 warmup_steps=25000,
                 lm_only_scale=0.25,
                 am_only_scale=0.0,
                 lr=0.01
                 ):
        super(Transducer, self).__init__()
        # Define Model
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint
        self.attention_decoder = attention_decoder
        self.ctc_decoder = ctc

        # Define Attributions
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.lsm_weight = lsm_weight
        self.transducer_weight = transducer_weight
        self.attention_weight = attention_weight
        self.delay_penalty = delay_penalty
        self.warmup_steps = warmup_steps
        self.lm_only_scale = lm_only_scale
        self.am_only_scale = am_only_scale
        self.vocab_size = vocab_size
        self.blank = blank
        self.sos = sos
        self.eos = eos
        self.lr = lr

        # For K2 RnntLoss
        self.simple_am_proj = nn.Linear(self.encoder.encoder_dim, vocab_size)
        self.simple_lm_proj = nn.Linear(self.predictor.embed_size, vocab_size)

        # For AttentionLoss
        self.criterion_attn = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight
            )

    def training_step(self, batch, batch_idx):
        sorted_keys, padded_feats, feats_length, padded_labels, label_lengths, transcripts = batch
        encoder_out, encoder_mask = self.encoder(padded_feats, feats_length)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss_rnnt = self.rnnt_loss(encoder_out,
                                   encoder_mask,
                                   padded_labels,
                                   label_lengths)

        loss_attn = self.attn_loss(encoder_out,
                                   encoder_mask,
                                   padded_labels,
                                   label_lengths)

        loss_ctc = self.ctc_loss(encoder_out,
                                 encoder_out_lens,
                                 padded_labels,
                                 label_lengths)

        loss = self.ctc_weight * loss_ctc + self.attention_weight * loss_attn + self.transducer_weight * loss_rnnt

        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=encoder_out.size(0))
        self.log('train_ctc_loss', loss_ctc, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=encoder_out.size(0))
        self.log('train_attn_loss', loss_attn, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=encoder_out.size(0))
        self.log('train_rnnt_loss', loss_rnnt, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=encoder_out.size(0))
        self.log('train_batch_size', encoder_out.size(0), prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=encoder_out.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        sorted_keys, padded_feats, feats_length, padded_labels, label_lengths, transcripts = batch
        encoder_out, encoder_mask = self.encoder(padded_feats, feats_length)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss_rnnt = self.rnnt_loss(encoder_out,
                                   encoder_mask,
                                   padded_labels,
                                   label_lengths)

        loss_attn = self.attn_loss(encoder_out,
                                   encoder_mask,
                                   padded_labels,
                                   label_lengths)

        loss_ctc = self.ctc_loss(encoder_out,
                                 encoder_out_lens,
                                 padded_labels,
                                 label_lengths)

        loss = self.ctc_weight * loss_ctc + self.attention_weight * loss_attn + self.transducer_weight * loss_rnnt

        self.log('valid_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=encoder_out.size(0))
        self.log('valid_ctc_loss', loss_ctc, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=encoder_out.size(0))
        self.log('valid_attn_loss', loss_attn, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=encoder_out.size(0))
        self.log('valid_rnnt_loss', loss_rnnt, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True, batch_size=encoder_out.size(0))

    def rnnt_loss(self,
                  encoder_out,
                  encoder_mask,
                  padded_labels,
                  label_lengths):
        padded_labels_pad = add_blank(padded_labels, self.blank)
        predictor_out = self.predictor(padded_labels_pad)
        steps = self.global_step
        if steps > 2 * self.warmup_steps:
            self.delay_penalty = 0.0
        boundary = torch.zeros((encoder_out.size(0), 4),
                               dtype=torch.int64,
                               device=encoder_out.device)
        boundary[:, 3] = encoder_mask.squeeze(1).sum(1)
        boundary[:, 2] = label_lengths
        rnnt_text = torch.where(padded_labels == self.ignore_id, 0, padded_labels)
        lm = self.simple_lm_proj(predictor_out)
        am = self.simple_am_proj(encoder_out)
        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=rnnt_text,
                termination_symbol=self.blank,
                lm_only_scale=self.lm_only_scale,
                am_only_scale=self.am_only_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
                delay_penalty=self.delay_penalty,
            )
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=5,
        )
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joint.enc_ffn(encoder_out),
            lm=self.joint.pred_ffn(predictor_out),
            ranges=ranges,
        )
        logits = self.joint(
            am_pruned,
            lm_pruned,
            pre_project=False,
        )
        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=rnnt_text,
                ranges=ranges,
                termination_symbol=self.blank,
                boundary=boundary,
                reduction="sum",
                delay_penalty=self.delay_penalty,
            )
        simple_loss_scale = 0.5
        if steps < self.warmup_steps:
            simple_loss_scale = (
                    1.0 - (steps / self.warmup_steps) * (1.0 - simple_loss_scale))
        pruned_loss_scale = 1.0
        if steps < self.warmup_steps:
            pruned_loss_scale = 0.1 + 0.9 * (steps / self.warmup_steps)
        loss = (simple_loss_scale * simple_loss
                + pruned_loss_scale * pruned_loss)
        loss = loss / encoder_out.size(0)

        return loss

    def attn_loss(self,
                  encoder_out,
                  encoder_mask,
                  padded_labels,
                  label_lengths):
        input_targets, output_targets = add_sos_eos(padded_labels, self.sos, self.eos, self.ignore_id)
        input_lengths = label_lengths + 1
        r_padded_labels = reverse_sequence(padded_labels, label_lengths, float(self.ignore_id))
        r_input_targets, r_output_targets = add_sos_eos(r_padded_labels, self.sos, self.eos, self.ignore_id)
        decoder_out, r_decoder_out, _ = self.attention_decoder(encoder_out,
                                                               encoder_mask,
                                                               input_targets,
                                                               input_lengths,
                                                               r_input_targets,
                                                               self.reverse_weight)
        loss_attn = self.criterion_attn(decoder_out, output_targets)
        r_loss_attn = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_attn = self.criterion_attn(r_decoder_out, r_output_targets)

        loss_attn = loss_attn * (1 - self.reverse_weight) + self.reverse_weight * r_loss_attn
        loss_attn = loss_attn.sum()
        return loss_attn

    def ctc_loss(self,
                 encoder_out,
                 encoder_out_lens,
                 padded_labels,
                 label_lengths):
        decoder_loss = self.ctc_decoder(encoder_out,
                                        encoder_out_lens,
                                        padded_labels,
                                        label_lengths).sum()
        return decoder_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = WarmupLR(optimizer, warmup_steps=self.warmup_steps)
        return [optimizer], [scheduler]
