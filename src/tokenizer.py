import sentencepiece as sp


class Tokenizer:

    def __init__(self, bpe_path, vocab_path, max_len):
        self.bpe_model = sp.SentencePieceProcessor()
        self.bpe_model.Load(bpe_path)
        self.vocab_path = vocab_path
        self.max_len = max_len
        self.__load_vocab()

    def __load_vocab(self):
        self.idx2word = dict()
        with open(self.vocab_path) as f:
            for line in f:
                line = line.strip()
                word, idx = line.split()
                self.idx2word[int(idx)] = word
        self.word2idx = {word: idx for idx, word in self.idx2word.items()}
        self.vocab_size = len(self.idx2word)
        self.bos_id = self.word2idx['<bos>']
        self.eos_id = self.word2idx['<eos>']
        self.blank_id = self.word2idx['<blank>']
        self.unk_id = self.word2idx['<unk>']

    def __call__(self, sentence):
        if isinstance(sentence, str):
            return self.tokenize_single_sentence(sentence)
        elif isinstance(sentence, list):
            return self.tokenize_multi_sentences(sentence)

    def tokenize_single_sentence(self, sentence):
        result = []
        sentence = self.bpe_model.EncodeAsPieces(sentence)
        for word in sentence:
            input_id = self.word2idx.get(word, 1)
            result.append(input_id)
            if len(result) == self.max_len - 1:
                break

        if len(result) < self.max_len:
            result += [self.blank_id] * (self.max_len - len(result))
        return result

    def tokenize_multi_sentences(self, sentence_list):
        results = []
        for sentence in sentence_list:
            results.append(self.tokenize_single_sentence(sentence))
        return results
