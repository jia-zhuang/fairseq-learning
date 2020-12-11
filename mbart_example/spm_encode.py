import sentencepiece as spm
import argparse


sp = spm.SentencePieceProcessor('models/mbart.cc25/sentence.bpe.model')


def encode(input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            fout.write(' '.join(sp.encode_as_pieces(line)) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    encode(args.input_file, args.output_file)
