class Tokenizer:

    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.__load_vocab()


    def __load_vocab(self):
        self.idx2word = []
        with open(self.vocab_path) as f:
            for line in f:
                word = line.strip()
                self.idx2word.append(word)
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}
        self.vocab_size = len(self.idx2word)


    def __call__(self, sentence):
        if isinstance(sentence, str):
            return self.tokenize_single_sentence(sentence)
        elif isinstance(sentence, list):
            return self.tokenize_multi_sentences(sentence)


    def tokenize_single_sentence(self, sentence):
        result = []
        sentence = sentence.split(' ')
        for word in sentence:
            input_id = self.word2idx[word]
            result.append(input_id)
        return result



    def tokenize_multi_sentences(self, sentence_list):
        results = []
        for sentence in sentence_list:
            results.append(self.tokenize_single_sentence(sentence))
        return results
