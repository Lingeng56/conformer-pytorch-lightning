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

    def decode(self, encoder_outputs, output_lengths, method='greedy', beam_size=5):
        assert method in ['greedy', 'beam']
        if method == 'greedy':
            sentences = self(encoder_outputs).argmax(dim=-1).tolist()
            for idx, sentence in enumerate(sentences):
                sentences[idx] = self.__clean_sentence(sentence, output_lengths[idx])
        else:
            sentences = self.beam_search(encoder_outputs, output_lengths, beam_size, 0)
            for idx, sentence in enumerate(sentences):
                sentences[idx] = self.__clean_sentence(sentence, output_lengths[idx])
        return sentences

    def __clean_sentence(self, sentence, output_length):
        sentence = sentence[:output_length]
        if self.tokenizer.word2idx['<EOS>'] in sentence:
            sentence = sentence[:sentence.index(self.tokenizer.word2idx['<EOS>'])]
        if self.tokenizer.word2idx['<BOS>'] in sentence:
            sentence = sentence[sentence.index(self.tokenizer.word2idx['<BOS>']) + 1:]
        sentence = ' '.join(sentence)
        return sentence


    def beam_search(self, encoder_outputs, output_lengths, beam_size, blank):
        probs = self(encoder_outputs)
        batch_size, seq_len, vocab_size = probs.size()
        beam = [((self.tokenizer.bos, ), 0.0)]
        sentences = []
        for idx, prob in enumerate(probs):
            for t in range(output_lengths[idx]):
                candidates = []
                for seq, score in beam:
                    if len(seq) > 0 and seq[-1] == blank:
                        candidate = (seq, score + torch.log(prob[t, blank]))
                        candidates.append(candidate)

                    for vocab_idx in range(vocab_size):
                        if len(seq) == 0 or vocab_idx != seq[-1]:
                            candidate = (seq + (vocab_idx, ), score + torch.log(prob[t, vocab_idx]))
                            candidates.append(candidate)

                candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
                beam = []

                for seq, score in candidates:
                    if seq not in [s for s, _ in beam]:
                        beam.append((seq, score))


                total_score = sum([torch.exp(score) for _, score in beam])
                beam = [(seq, score - torch.log(total_score)) for seq, score in beam]

            best_seq, best_score = max(beam, key=lambda x: x[1])
            sentences.append(best_seq)

        return sentences

