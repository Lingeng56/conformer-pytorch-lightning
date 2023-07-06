import argparse
import glob


def generate_vocab(data_dirs, vocab_path):
    files = []
    for dir in data_dirs:
        search_path = dir + '/**/*.txt'
        files += glob.glob(search_path, recursive=True)
    vocab = set()
    for filename in files:
        with open(filename) as f:
            for line in f:
                line = line.strip()
                line = line.split(' ')[1:]
                for word in line:
                    vocab.add(word)

    with open(vocab_path, 'w') as f:
        f.write('<BOS>\n<EOS>\n')
        for word in vocab:
            f.write(word + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Generate Vocabulary')
    parser.add_argument('--data_dirs', nargs='+', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    args = parser.parse_args()
    generate_vocab(args.data_dirs, args.vocab_path)

