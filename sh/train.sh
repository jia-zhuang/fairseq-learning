#!/usr/bin/env bash

mkdir -p checkpoints/simple_lstm
# CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
#   --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
#   --arch fconv_iwslt_de_en --save-dir checkpoints/fconv

fairseq-train data-bin/iwslt14.tokenized.de-en \
    --user-dir examples/simple_lstm \
    --arch tutorial_simple_lstm \
    --encoder-dropout 0.2 --decoder-dropout 0.2 \
    --optimizer adam --lr 0.005 --lr-shrink 0.5 \
    --max-tokens 12000 \
    --save-dir checkpoints/simple_lstm
