import sys

sys.path.append('/home/wuliu/workspace/conformer-pytorch-lightning/src')

import json
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch
import numpy as np
import io
from encoder import ConformerEncoder
from decoder import CTCDecoder, BiTransformerDecoder
from joint import TransducerJoint
from model import Transducer
from module import TransducerModule
from predictor import RNNPredictor
from utils import load_vocabs
from cmvn import GlobalCMVN
from deploy_common import Common

DEVICE = Common.device


def build_model():
    args = Common()
    data_config = json.load(open(args.data_config_path, 'r'))
    vocabs, vocab_size = load_vocabs(args.vocab_path)
    data_config['vocabs'] = vocabs

    char_dict = {idx: w for w, idx in vocabs.items()}

    cmvn = GlobalCMVN(
        cmvn_path=args.cmvn_path
    )

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
        use_relative=args.use_relative,
        cmvn=cmvn
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

    predictor = RNNPredictor(vocab_size=vocab_size,
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

    model = Transducer(
        encoder=conformer_encoder,
        predictor=predictor,
        joint=joint,
        attention_decoder=attn_decoder,
        ctc=ctc_decoder,
        vocab_size=vocab_size,
        blank=vocabs['<blank>'],
        sos=vocabs['<sos/eos>'],
        eos=vocabs['<sos/eos>'],
        ignore_id=-1,
        ctc_weight=args.ctc_weight,
        reverse_weight=args.reverse_weight,
        lsm_weight=args.lsm_weight,
        transducer_weight=args.transducer_weight,
        attention_weight=args.attention_weight,
        delay_penalty=args.delay_penalty,
        warmup_steps=args.warmup_steps,
        lm_only_scale=args.lm_only_scale,
        am_only_scale=args.am_only_scale,
        wenet_ckpt_path=args.wenet_ckpt_path,
        device=args.device
    )

    module = TransducerModule(
        model,
        lr=args.lr,
        ckpt_path=args.checkpoint_path,
        char_dict=char_dict,
        bpe_model=data_config['bpe_model'],
        warmup_steps=25000)

    checkpoint = torch.load(args.resume_path)
    module.load_state_dict(checkpoint['state_dict'])

    module.to(DEVICE)
    return module


def preprocess_stream(audio_bytes):
    waveform, sample_rate = torchaudio.load(audio_bytes)
    waveform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=16000
    )(waveform)
    waveform = waveform * (1 << 15)
    feat = kaldi.fbank(waveform,
                       num_mel_bins=80,
                       frame_length=25,
                       frame_shift=10,
                       dither=0.1,
                       energy_floor=0.0,
                       sample_frequency=16000)
    feat_length = torch.tensor(feat.size(0))

    feat = feat.to(DEVICE).unsqueeze(0)
    feat_length = feat_length.to(DEVICE).unsqueeze(0)

    return feat, feat_length


def preprocess(audio):
    waveform, sample_rate = torchaudio.io._compat.load_audio_fileobj(audio)
    waveform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=16000
    )(waveform)
    waveform = waveform * (1 << 15)
    # Only keep key, feat, label
    feat = kaldi.fbank(waveform,
                       num_mel_bins=80,
                       frame_length=25,
                       frame_shift=10,
                       dither=0.1,
                       energy_floor=0.0,
                       sample_frequency=16000)
    feat_length = torch.tensor(feat.size(0))

    feat = feat.to(DEVICE).unsqueeze(0)
    feat_length = feat_length.to(DEVICE).unsqueeze(0)

    return feat, feat_length
