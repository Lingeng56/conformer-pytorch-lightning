import torch.nn as nn


class TransducerJoint(nn.Module):

    def __init__(self,
                 vocab_size,
                 enc_output_size,
                 pred_output_size,
                 join_dim
                 ):
        super(TransducerJoint, self).__init__()

        self.activation = nn.Tanh()
        self.enc_ffn = nn.Linear(enc_output_size, join_dim)
        self.pred_ffn = nn.Linear(pred_output_size, join_dim)
        self.ffn_out = nn.Linear(join_dim, vocab_size)


    def forward(self,
                enc_out,
                pred_out,
                pre_project=True):

        if pre_project:
            enc_out = self.enc_ffn(enc_out)
            pred_out = self.pred_ffn(pred_out)

        if enc_out.ndim != 4:
            enc_out = enc_out.unsqueeze(2)

        if pred_out.ndim != 4:
            pred_out = pred_out.unsqueeze(1)

        out = enc_out + pred_out
        out = self.activation(out)
        out = self.ffn_out(out)
        return out
