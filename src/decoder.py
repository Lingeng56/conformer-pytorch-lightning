import torch.nn as nn



class LSTMDecoder(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, vocab_size):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=encoder_dim, hidden_size=decoder_dim, num_layers=1)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, inputs):
        outputs, _ = self.lstm(inputs)
        outputs = self.fc(outputs)
        probs = self.softmax(outputs)
        return probs

    def decode(self, probs):
        sentence = None
        return sentence
