"""
TODO
Collect all transcripts for BPE
Collect all wav_data_paths for DataList
Build DataList for Dataset Sampling
"""
import os
import glob
import argparse



def collect_transcripts(data_dir, save_dir):
    search_path = data_dir + '/**/*.txt'
    files = glob.glob(search_path, recursive=True)
    transcript_ids = []
    transcripts = []
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                transcript_id, transcript = line.split()[0], ' '.join(line.split()[1:])
                transcripts.append(transcript)
                transcript_ids.append(transcript_id)

    with open(os.path.join(save_dir, 'transcripts.txt'), 'w') as f:
        for transcript in transcripts:
            f.write('%s\n' % transcript)

    return dict(zip(transcript_ids, transcripts))



def collect_wav_paths(data_dir, save_dir):
    search_path = data_dir + '/**/*.flac'
    wav_paths = glob.glob(search_path, recursive=True)
    wav_path_ids = [os.path.basename(path).split('.')[0] for path in wav_paths]


    with open(os.path.join(save_dir, 'wav_paths.txt'), 'w') as f:
        for wav_path in wav_paths:
            f.write('%s\n' % wav_path)

    return dict(zip(wav_path_ids, wav_paths))


def build_data_list(id2transcript, id2wav_path, save_dir):
    with open(os.path.join(save_dir, 'data.list'), 'w') as f:
        for key in id2transcript.keys():
            transcript = id2transcript[key]
            wav_path = id2wav_path[key]
            f.write('{"key": "%s", "wav_path": "%s", "transcript": "%s"}\n' % (key, wav_path, transcript))




def main(args):
    for part in args.parts:
        if not os.path.exists(os.path.join(args.save_dir, part)):
            os.mkdir(os.path.join(args.save_dir, part))
        id2transcript =collect_transcripts(os.path.join(args.data_dir, part), os.path.join(args.save_dir, part))
        id2wav_path = collect_wav_paths(os.path.join(args.data_dir, part), os.path.join(args.save_dir, part))
        build_data_list(id2transcript, id2wav_path, os.path.join(args.save_dir, part))
        print('Collecting Part [%s] Successfully' % part)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Collect LIBRISPEECH Dataset"
    )
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--parts', type=str, nargs='+', required=True)
    args_ = parser.parse_args()
    main(args_)

