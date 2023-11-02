import torch
import torch.nn as nn
import torchaudio
from utils import add_blank


class Transducer(nn.Module):

    def __init__(self,
                 encoder,
                 predictor,
                 joint,
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
        # self.decoder = attention_decoder
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

        # For stream asr
        self.attn_cache = torch.zeros((0, 0, 0, 0))
        self.cnn_cache = torch.zeros((0, 0, 0, 0))
        self.offset = 0
        self.tmp_hyps = []
        self.pred_input_step = None
        self.pred_cache = None



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

        loss_ctc = self.ctc_loss(encoder_out,
                                 encoder_out_lens,
                                 padded_labels,
                                 label_lengths)

        # loss = self.ctc_weight * loss_ctc + self.attention_weight * loss_attn + self.transducer_weight * loss_rnnt
        loss = self.ctc_weight * loss_ctc + self.transducer_weight * loss_rnnt

        return {'loss': loss,
                # 'loss_attn': loss_attn,
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
    def greedy_search_streaming_eval(self,
                                     inputs,
                                     decoding_chunk_size,
                                     num_decoding_left_chunks=-1,
                                     n_steps=64
                                     ):
        subsampling_rate = 4
        context = 7
        stride = subsampling_rate * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling_rate + context
        num_frames = inputs.size(1)
        attn_cache = torch.zeros((0, 0, 0, 0), device=inputs.device)
        cnn_cache = torch.zeros((0, 0, 0, 0), device=inputs.device)
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        hyps = []
        pred_cache = None
        pred_input_step = None
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_inputs = inputs[:, cur:end, :]
            chunk_outputs, attn_cache, cnn_cache = self.encoder.forward_chunk(inputs=chunk_inputs,
                                                                              offset=offset,
                                                                              required_cache_size=required_cache_size,
                                                                              attn_cache=attn_cache,
                                                                              cnn_cache=cnn_cache
                                                                              )
            chunk_out_lens = torch.tensor([chunk_outputs.size(1)])
            chunk_hyps, (pred_input_step, pred_cache) = self.basic_greedy_search(
                encoder_out=chunk_outputs,
                encoder_out_lens=chunk_out_lens,
                n_steps=n_steps,
                cache=None,
                pred_input_step=None
            )
            offset += chunk_outputs.size(1)
            hyps += chunk_hyps

        return hyps

    def init_state(self):
        print('Model Reset')
        self.attn_cache = torch.zeros((0, 0, 0, 0))
        self.cnn_cache = torch.zeros((0, 0, 0, 0))
        self.offset = 0
        self.tmp_hyps = []
        self.pred_input_step = None
        self.pred_cache = None


    @torch.no_grad()
    def greedy_search_streaming_app(self,
                                    chunk_inputs,
                                    n_steps=64
                                    ):
        chunk_outputs, self.attn_cache, self.cnn_cache = self.encoder.forward_chunk(inputs=chunk_inputs,
                                                                                    offset=self.offset,
                                                                                    required_cache_size=-1,
                                                                                    attn_cache=self.attn_cache,
                                                                                    cnn_cache=self.cnn_cache
                                                                                    )
        chunk_out_lens = torch.tensor([chunk_outputs.size(1)])
        chunk_hyps, (self.pred_input_step, self.pred_cache) = self.basic_greedy_search(
            encoder_out=chunk_outputs,
            encoder_out_lens=chunk_out_lens,
            n_steps=n_steps,
            cache=self.pred_cache,
            pred_input_step=self.pred_input_step
        )
        self.offset += chunk_outputs.size(1)
        self.tmp_hyps += chunk_hyps

        return self.tmp_hyps

    @torch.no_grad()
    def greedy_search(self,
                      speech,
                      speech_lengths,
                      n_steps=64):
        encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
            speech,
            speech_lengths
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum()
        hyps, _ = self.basic_greedy_search(encoder_out, encoder_out_lens, n_steps=n_steps)
        return hyps

    @torch.no_grad()
    def basic_greedy_search(
            self: torch.nn.Module,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            n_steps: int = 64,
            cache: torch.Tensor = None,
            pred_input_step: torch.Tensor = None
    ):
        # fake padding
        padding = torch.zeros(1, 1).to(encoder_out.device)
        # sos
        if pred_input_step is None:
            pred_input_step = torch.tensor([self.blank]).reshape(1, 1).to(encoder_out.device)
        else:
            pred_input_step = pred_input_step.to(encoder_out.device)

        if cache is None:
            cache = self.predictor.init_state(pred_input_step)
        else:
            cache = (cache[0].to(encoder_out.device), cache[1].to(encoder_out.device))

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
                t = t + 1
                per_frame_noblk = 0

        return hyps, (pred_input_step, cache)
