import torch.nn as nn
import torch


class LSTMDecoder(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, decoder_layer_nums, vocab_size, tokenizer):
        super(LSTMDecoder, self).__init__()
        self.tokenizer = tokenizer
        self.lstm = nn.LSTM(input_size=encoder_dim, hidden_size=decoder_dim, num_layers=decoder_layer_nums, batch_first=True)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_outputs, hidden=None):
        outputs, _ = self.lstm(encoder_outputs, hidden)
        outputs = self.fc(outputs)
        probs = self.softmax(outputs)
        return probs

    def decode(self, encoder_outputs, method='greedy', beam_size=-1):
        assert method in ['greedy', 'beam']
        if method == 'greedy':
            sentences = encoder_outputs.argmax(dim=-1).tolist()
            for idx, sentence in enumerate(sentences):
                sentence = ' '.join([self.tokenizer.idx2word[token_id] for token_id in sentence])
                sentences[idx] = sentence
        else:
            sentences = encoder_outputs.argmax(dim=-1).tolist()
            for idx, sentence in enumerate(sentences):
                sentence = [self.tokenizer.idx2word[token_id] for token_id in sentence]
                sentences[idx] = sentence
        return sentences

