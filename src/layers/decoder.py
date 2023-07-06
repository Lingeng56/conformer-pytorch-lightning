import torch.nn as nn
import torch


class LSTMDecoder(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, vocab_size, tokenizer):
        super(LSTMDecoder, self).__init__()
        self.tokenizer = tokenizer
        self.lstm = nn.LSTM(input_size=encoder_dim, hidden_size=decoder_dim, num_layers=1)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = self.fc(outputs)
        probs = self.softmax(outputs)
        return probs

    def decode(self, encoder_outputs, method='greedy', beam_size=-1):
        assert method in ['greedy', 'beam']
        sentences = []
        if method == 'greedy':
            for encoder_output in encoder_outputs:
                sentence = self.greedy_search(encoder_output)
                sentences.append(sentence)
        else:
            for encoder_output in encoder_outputs:
                sentence = self.greedy_search(encoder_output)
                sentences.append(sentence)
        return sentences

    def greedy_search(self, encoder_output):
        decoder_input = torch.tensor([[self.tokenizer.bos]])

        # init hidden state
        decoder_hidden = encoder_output[:, -1, :]

        decoded_sequence = []

        for _ in range(self.tokenizer.max_len):
            decoder_output, decoder_hidden = self(decoder_input, decoder_hidden)

            top_value, top_index = decoder_output.topk(1)

            next_token = top_index.item()

            decoded_sequence.append(next_token)

            if next_token == self.tokenizer.eos:
                break

            decoder_input = torch.tensor([[next_token]])

        return decoded_sequence
