import os
import torch

from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.tasks import FairseqTask, register_task


@register_task('simple_classification')
class SimpleClassificationTask(FairseqTask):

    @staticmethod
    def add_args(parser):

        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        input_vocab = Dictionary.load(os.path.join(args.data, 'dict.input.txt'))
        label_vocab = Dictionary.load(os.path.join(args.data, 'dict.label.txt'))
        print('| [input] dictionary: {} types'.format(len(input_vocab)))
        print('| [label] dictionary: {} types'.format(len(label_vocab)))

        return cls(args, input_vocab, label_vocab)
    
    def __init__(self, args, input_vocab, label_vocab):
        super().__init__(args)
        self.input_vocab = input_vocab
        self.label_vocab = label_vocab
    
    def load_dataset(self, split, **kwargs):
        prefix = os.path.join(self.args.data, '{}.input-label'.format(split))

        # Read input sentences
        sentences, lengths = [], []
        with open(prefix + '.input', encoding='utf-8') as file:
            for line in file:
                sentence = line.strip()

                tokens = self.input_vocab.encode_line(
                    sentence, add_if_not_exist=False
                )

                sentences.append(tokens)
                lengths.append(tokens.numel())
        
        # Read labels
        labels = []
        with open(prefix + '.label', encoding='utf-8') as file:
            for line in file:
                label = line.strip()
                labels.append(
                    torch.LongTensor([self.label_vocab.add_symbol(label)])
                )
        
        assert len(sentences) == len(labels)
        print('| {} {} {} examples'.format(self.args.data, split, len(sentences)))

        self.datasets[split] = LanguagePairDataset(
            src=sentences,
            src_sizes=lengths,
            src_dict=self.input_vocab,
            tgt=labels,
            tgt_sizes=torch.ones(len(labels)),
            tgt_dict=self.label_vocab,
            left_pad_source=False,
            input_feeding=False,
        )

    def max_positions(self):
        return (self.args.max_positions, 1)
    
    @property
    def source_dictionary(self):
        return self.input_vocab

    @property
    def target_dictionary(self):
        return self.label_vocab
