#!/usr/bin/env bash

MODEL_DIR=data-bin
fairseq-interactive \
  --path checkpoints/fconv/checkpoint_best.pt $MODEL_DIR \
  --beam 5 --source-lang de --target-lang en \
  --tokenizer moses 
