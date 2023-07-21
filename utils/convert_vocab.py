import sys


vocab_path = sys.argv[1]
with open(vocab_path, 'w') as f_out:
    with open('bpe_model.vocab', 'r') as f_in:
        f_out.write('<blank> 0\n')
        f_out.write('<unk> 1\n')
        f_out.write('<bos> 2\n')
        f_out.write('<eos> 3\n')
        curr_id = 4
        for line in f_in:
            line = line.strip()
            symbol, _ = line.split()
            if symbol == '<unk>':
                continue
            f_out.write('%s %d\n' % (symbol, curr_id))
            curr_id += 1

