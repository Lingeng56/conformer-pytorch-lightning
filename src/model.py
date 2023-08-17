import torch
import torch.nn as nn
import torchaudio
from utils import add_blank, add_sos_eos, reverse_sequence
from label_smoothing_loss import LabelSmoothingLoss


class Transducer(nn.Module):

    def __init__(self,
                 encoder,
                 predictor,
                 joint,
                 attention_decoder,
                 ctc,
                 vocab_size=5002,
                 blank=0,
                 sos=5001,
                 eos=5001,
                 ignore_id=-1,
                 ctc_weight=0.0,
                 reverse_weight=0.0,
                 lsm_weight=0.0,
                 transducer_weight=1.0,
                 attention_weight=0.0,
                 delay_penalty=0.0,
                 warmup_steps=25000,
                 lm_only_scale=0.25,
                 am_only_scale=0.0,
                 wenet_ckpt_path=None
                 ):
        super(Transducer, self).__init__()
        # Define Model
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint
        self.decoder = attention_decoder
        self.ctc = ctc

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

        # For AttentionLoss
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight
        )

        if wenet_ckpt_path is not None:
            print('Load Wenet Checkpoint : %s' % wenet_ckpt_path)
            checkpoint = torch.load(wenet_ckpt_path)
            self.load_state_dict(checkpoint)

    def forward(self, batch):
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

        return {'loss': loss,
                'loss_attn': loss_attn,
                'loss_ctc': loss_ctc,
                'loss_rnnt': loss_rnnt,
                'encoder_out': encoder_out,
                'encoder_out_lens': encoder_out_lens}

    def rnnt_loss(self,
                  encoder_out,
                  encoder_mask,
                  padded_labels,
                  label_lengths):
        padded_labels_pad = add_blank(padded_labels, self.blank, self.ignore_id)
        predictor_out = self.predictor(padded_labels_pad)
        joint_out = self.joint(encoder_out, predictor_out)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        rnnt_text = torch.where(padded_labels == self.ignore_id, self.blank, padded_labels).to(torch.int32)
        rnnt_text_lengths = label_lengths.to(torch.int32)
        encoder_out_lens = encoder_out_lens.to(torch.int32)
        loss = torchaudio.functional.rnnt_loss(joint_out,
                                               rnnt_text,
                                               encoder_out_lens,
                                               rnnt_text_lengths,
                                               blank=self.blank,
                                               reduction="mean")
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
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out,
                                                     encoder_mask,
                                                     input_targets,
                                                     input_lengths,
                                                     r_input_targets,
                                                     self.reverse_weight)
        batch_size, seq_len, _ = decoder_out.size()
        loss_attn = self.criterion_att(decoder_out,
                                       output_targets)
        r_loss_attn = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_attn = self.criterion_att(r_decoder_out,
                                             r_output_targets)

        loss_attn = loss_attn * (1 - self.reverse_weight) + self.reverse_weight * r_loss_attn
        loss_attn = loss_attn.sum()
        return loss_attn

    def ctc_loss(self,
                 encoder_out,
                 encoder_out_lens,
                 padded_labels,
                 label_lengths):
        decoder_loss = self.ctc(encoder_out,
                                encoder_out_lens,
                                padded_labels,
                                label_lengths).sum()
        return decoder_loss

    @torch.no_grad()
    def greedy_search(self,
                      speech,
                      speech_lengths,
                      n_steps=64):
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum()
        hyps = self.basic_greedy_search(encoder_out, encoder_out_lens, n_steps=n_steps)
        return hyps


    @torch.no_grad()
    def basic_greedy_search(
            self: torch.nn.Module,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            n_steps: int = 64,
    ):
        # fake padding
        padding = torch.zeros(1, 1).to(encoder_out.device)
        # sos
        pred_input_step = torch.tensor([self.blank]).reshape(1, 1).to(encoder_out.device)
        cache = self.predictor.init_state(pred_input_step)
        new_cache = []
        t = 0
        hyps = []
        prev_out_nblk = True
        pred_out_step = None
        per_frame_max_noblk = n_steps
        per_frame_noblk = 0
        while t < encoder_out_lens:
            encoder_out_step = encoder_out[:, t:t + 1, :]  # [1, 1, E]
            if prev_out_nblk:
                step_outs = self.predictor.forward_step(pred_input_step, padding,
                                                        cache)  # [1, 1, P]
                pred_out_step, new_cache = step_outs[0], step_outs[1]

            joint_out_step = self.joint(encoder_out_step,
                                        pred_out_step)  # [1,1,v]
            joint_out_probs = joint_out_step.log_softmax(dim=-1)

            joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []
            if joint_out_max != self.blank:
                hyps.append(joint_out_max.item())
                prev_out_nblk = True
                per_frame_noblk = per_frame_noblk + 1
                pred_input_step = joint_out_max.reshape(1, 1)
                # state_m, state_c =  clstate_out_m, state_out_c
                cache = new_cache

            if joint_out_max == self.blank or per_frame_noblk >= per_frame_max_noblk:
                if joint_out_max == self.blank:
                    prev_out_nblk = False
                # TODO(Mddct): make t in chunk for streamming
                # or t should't be too lang to predict none blank
                t = t + 1
                per_frame_noblk = 0

        return hyps
