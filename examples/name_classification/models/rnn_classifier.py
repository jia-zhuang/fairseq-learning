import torch
import torch.nn as nn
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models import register_model_architecture

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        combined = torch.cat([input, hidden], dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


@register_model('rnn_classifier')
class FairseqRNNClassifier(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--hidden-dim', type=int, metavar='N',
            help='dim of the hidden state'
        )
    
    @classmethod
    def build_model(cls, args, task):

        rnn = RNN(
            input_size=len(task.source_dictionary),
            hidden_size=args.hidden_dim,
            output_size=len(task.target_dictionary)
        )

        return cls(
            rnn=rnn,
            input_vocab=task.source_dictionary,
        )
    
    def __init__(self, rnn, input_vocab):
        super().__init__()

        self.rnn = rnn
        self.input_vocab = input_vocab

        # ?
        self.register_buffer('one_hot_inputs', torch.eye(len(input_vocab)))
    
    def forward(self, src_tokens, src_lengths):
        bsz, max_src_len = src_tokens.size()

        hidden = self.rnn.initHidden()
        hidden = hidden.repeat(bsz, 1)
        hidden = hidden.to(src_tokens.device)

        for i in range(max_src_len):
            input = self.one_hot_inputs[src_tokens[:, i].long()]

            output, hidden = self.rnn(input, hidden)
        
        return output


@register_model_architecture('rnn_classifier', 'pytorch_tutorial_rnn')
def pytorch_tutorial_rnn(args):
    args.hidden_dim = getattr(args, 'hidden_dim', 128)
