#!/usr/bin/env bash

# TEXT=data/iwslt14.tokenized.de-en
# fairseq-preprocess --source-lang de --target-lang en \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/iwslt14.tokenized.de-en

fairseq-preprocess \
  --trainpref names/train --validpref names/valid --testpref names/test \
  --source-lang input --target-lang label \
  --destdir names-bin --dataset-impl raw
