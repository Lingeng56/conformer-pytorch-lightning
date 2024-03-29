import argparse
from executor import Executor

# Config Arguments
parser = argparse.ArgumentParser(
    prog='Conformer-Pytorch-Lightning',
    description='Pytorch-Lightning Implementation of Conformer')
parser.add_argument('--max_epochs', type=int, required=True, default=10)
parser.add_argument('--random_seed', type=int, required=True, default=3407)
parser.add_argument('--data_config_path', type=str, required=True)
parser.add_argument('--vocab_path', type=str, required=True)
parser.add_argument('--input_dim', type=int, required=True, default=256)
parser.add_argument('--kernel_size', type=int, required=True, default=32)
parser.add_argument('--encoder_dim', type=int, required=True, default=256)
parser.add_argument('--dropout', type=float, required=True, default=0.3)
parser.add_argument('--attention_dropout', type=float, required=True, default=0.2)
parser.add_argument('--pos_enc_dropout', type=float, required=True, default=0.2)
parser.add_argument('--hidden_dim', type=int, required=True, default=256)
parser.add_argument('--max_len', type=int, required=True, default=5000)
parser.add_argument('--num_heads', type=int, required=True, default=4)
parser.add_argument('--encoder_num_layers', type=int, required=True, default=4)
parser.add_argument('--decoder_num_layers', type=int, required=True, default=4)
parser.add_argument('--self_attention_dropout', type=float, required=True, default=0.1)
parser.add_argument('--src_attention_dropout', type=float, required=True, default=0.1)
parser.add_argument('--predictor_embed_size', type=int, required=True, default=32)
parser.add_argument('--predictor_hidden_size', type=int, required=True, default=32)
parser.add_argument('--predictor_dim', type=int, required=True, default=32)
parser.add_argument('--predictor_embed_dropout', type=float, required=True, default=0.1)
parser.add_argument('--predictor_num_layers', type=int, required=True, default=32)
parser.add_argument('--join_dim', type=int, required=True, default=32)
parser.add_argument('--ctc_weight', type=float, required=True, default=0.1)
parser.add_argument('--attention_weight', type=float, required=True, default=0.15)
parser.add_argument('--transducer_weight', type=float, required=True, default=0.75)
parser.add_argument('--lsm_weight', type=float, required=True, default=0.1)
parser.add_argument('--reverse_weight', type=float, required=True, default=0.3)
parser.add_argument('--delay_penalty', type=float, required=True, default=0.0)
parser.add_argument('--lm_only_scale', type=float, required=True, default=0.1)
parser.add_argument('--am_only_scale', type=float, required=True, default=0.1)
parser.add_argument('--use_relative', action='store_true')
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--num_devices', type=int, required=True, default=2)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--pin_memory', type=bool, default=True)
parser.add_argument('--prefetch', type=int, default=100)
parser.add_argument('--warmup_steps', type=int, default=25000)
parser.add_argument('--grad_clip', type=int, default=4)
parser.add_argument('--accum_grad', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--streaming_eval', action='store_true')
parser.add_argument('--decoding_chunk_size', type=int, default=32)
parser.add_argument('--num_decoding_left_chunks', type=int, default=-1)
parser.add_argument('--use_dynamic_chunk_size', action='store_true')
parser.add_argument('--use_dynamic_left_chunk', action='store_true')
parser.add_argument('--static_chunk_size', type=int, default=-1)
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_from', type=str)
parser.add_argument('--cmvn_path', type=str)
parser.add_argument('--wenet_ckpt_path', type=str, default=None)
args = parser.parse_args()

# Execute Task
executor = Executor(args)
if args.train:
    executor.train()

if args.eval:
    executor.eval()
