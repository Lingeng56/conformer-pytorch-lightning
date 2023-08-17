import sys


vocab_path = sys.argv[1]
with open(vocab_path, 'w') as f_out:
    with open('bpe_model.vocab', 'r') as f_in:
        f_out.write('<blank> 0\n')
        f_out.write('<unk> 1\n')
        f_out.write('<bos/eos> 2\n')
        curr_id = 3
        for line in f_in:
            line = line.strip()
            symbol, _ = line.split()
            if symbol == '<unk>':
                continue
            f_out.write('%s %d\n' % (symbol, curr_id))
            curr_id += 1

