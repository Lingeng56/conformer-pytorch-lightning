from dataclasses import dataclass




@dataclass
class Common:
    device = "cuda:1"
    data_config_path = "exp/data_config.json"
    vocab_path = "vocab.txt"
    cmvn_path = "data/train-960/global_cmvn"
    input_dim = 80
    kernel_size = 15
    encoder_dim = 256
    dropout = 0.1
    attention_dropout = 0.1
    pos_enc_dropout = 0.1
    hidden_dim = 2048
    max_len = 5000
    self_attention_dropout = 0.1
    src_attention_dropout = 0.1
    ctc_weight = 0.2
    attention_weight = 0.15
    transducer_weight = 0.8
    lsm_weight = 0.1
    reverse_weight = 0.3
    delay_penalty = 0.0
    lm_only_scale = 0.25
    am_only_scale = 0.0
    num_heads = 4
    encoder_num_layers = 12
    decoder_num_layers = 3
    use_relative = True
    predictor_embed_size = 256
    predictor_hidden_size = 256
    predictor_dim = 256
    predictor_embed_dropout = 0.1
    predictor_num_layers = 2
    join_dim = 512
    warmup_steps = 25000
    lr = 0.001
    wenet_ckpt_path = None
    resume_path = "experiments/conformer-rnnt-ctc-960/best.ckpt"
    checkpoint_path = "experiments/conformer-rnnt-ctc-960"
