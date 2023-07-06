class Tokenizer:

    def __init__(self, vocab_path, max_len):
        self.vocab_path = vocab_path
        self.max_len = max_len
        self.__load_vocab()


    def __load_vocab(self):
        self.idx2word = []
        with open(self.vocab_path) as f:
            for line in f:
                word = line.strip()
                self.idx2word.append(word)
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.vocab_size = len(self.idx2word)
        self.bos = self.word2idx['<BOS>']
        self.eos = self.word2idx['<EOS>']


    def __call__(self, sentence):
        if isinstance(sentence, str):
            return self.tokenize_single_sentence(sentence)
        elif isinstance(sentence, list):
            return self.tokenize_multi_sentences(sentence)


    def tokenize_single_sentence(self, sentence):
        result = [self.word2idx['<BOS>']]
        sentence = sentence.split(' ')
        for word in sentence:
            input_id = self.word2idx[word]
            result.append(input_id)
            if len(result) == self.max_len - 1:
                break
        result.append(self.word2idx['<EOS>'])
        result += ([-1] * self.max_len - len(result))
        return result



    def tokenize_multi_sentences(self, sentence_list):
        results = []
        for sentence in sentence_list:
            results.append(self.tokenize_single_sentence(sentence))
        return results
