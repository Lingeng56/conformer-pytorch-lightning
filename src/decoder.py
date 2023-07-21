import torch.nn as nn
import torch
from utils import remove_duplicates_and_blank


class View(nn.Module):

    def __init__(self, shape, contiguous=True):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous


    def forward(self, inputs):
        return inputs.view(self.shape).contiguous() if self.contiguous else inputs.view(self.shape)


class LSTMAttentionDecoder(nn.Module):

    def __init__(self, hidden_state_dim, decoder_layer_nums, num_heads, dropout, vocab_size, tokenizer, max_len):
        super(LSTMAttentionDecoder, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.hidden_state_dim = hidden_state_dim
        self.decoder_layer_nums = decoder_layer_nums
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(input_size=self.hidden_state_dim, hidden_size=self.hidden_state_dim, num_layers=self.decoder_layer_nums, batch_first=True, dropout=dropout, bias=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_state_dim << 1, self.hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            nn.Linear(self.hidden_state_dim, self.vocab_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_state_dim)
        self.attention = nn.MultiheadAttention(self.hidden_state_dim, num_heads, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward_step(self, input_var, hidden_states, encoder_outputs):
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.dropout(embedded)

        if self.training:
            self.lstm.flatten_parameters()

        outputs, hidden_states = self.lstm(embedded, hidden_states)

        context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)

        outputs = torch.cat((outputs, context), dim=2)

        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1))
        step_outputs = self.softmax(step_outputs)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states



    def forward(self, encoder_outputs, targets=None):
        hidden_states, attn = None, None
        targets, batch_size, max_length = self.validate_args(encoder_outputs, targets)
        input_var = targets[:, 0].unsqueeze(1)
        logits = list()

        for di in range(max_length):
            step_outputs, hidden_states = self.forward_step(
                input_var=input_var,
                hidden_states=hidden_states,
                encoder_outputs=encoder_outputs
            )
            logits.append(step_outputs)
            input_var = logits[-1].topk(1)[1]

        logits = torch.stack(logits, dim=1)

        return logits

    def validate_args(self, encoder_outputs, targets=None):
        batch_size = encoder_outputs.size(0)

        if targets is None:
            targets = torch.LongTensor([self.tokenizer.bos_id] * batch_size).view(batch_size, 1).to(encoder_outputs.device)
            max_length = self.max_len

        else:
            max_length = targets.size(1)

        return targets, batch_size, max_length




    def decode(self, probs, output_lengths, method='greedy', beam_size=5):
        with torch.no_grad():
            assert method in ['greedy', 'beam']
            if method == 'greedy':
                sentences = torch.unique_consecutive(probs.argmax(dim=-1), dim=-1).tolist()
                for idx, sentence in enumerate(sentences):
                    sentences[idx] = self.__clean_sentence(sentence, output_lengths[idx])
            else:
                sentences = self.beam_search(probs, output_lengths, beam_size, 0)
                for idx, sentence in enumerate(sentences):
                    sentences[idx] = self.__clean_sentence(sentence, output_lengths[idx])
            return sentences


    def __clean_sentence(self, sentence, output_length):
        # sentence = sentence[:output_length]
        # if self.tokenizer.word2idx['<EOS>'] in sentence:
        #     sentence = sentence[:sentence.index(self.tokenizer.word2idx['<EOS>'])]
        # if self.tokenizer.word2idx['<BOS>'] in sentence:
        #     sentence = sentence[sentence.index(self.tokenizer.word2idx['<BOS>']) + 1:]
        sentence = remove_duplicates_and_blank(sentence)
        sentence = sentence[:sentence.index(self.tokenizer.eos_id) if self.tokenizer.eos_id in sentence else None]
        sentence = [self.tokenizer.idx2word[t] for t in sentence]
        sentence = self.tokenizer.bpe_model.decode(sentence)
        return sentence


    def beam_search(self, encoder_outputs, output_lengths, beam_size, blank):
        probs = self(encoder_outputs)
        batch_size, seq_len, vocab_size = probs.size()
        beam = [((), 0.0)]
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

