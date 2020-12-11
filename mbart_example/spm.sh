#!/usr/bin/env bash

python spm_encode.py --input_file datasets/tmp/test.source --output_file datasets/drug_qa_sep_surround_answer0.sp/test.source &
python spm_encode.py --input_file datasets/tmp/test.target --output_file datasets/drug_qa_sep_surround_answer0.sp/test.target &

python spm_encode.py --input_file datasets/tmp/train.source --output_file datasets/drug_qa_sep_surround_answer0.sp/train.source &
python spm_encode.py --input_file datasets/tmp/train.target --output_file datasets/drug_qa_sep_surround_answer0.sp/train.target &

python spm_encode.py --input_file datasets/tmp/val.source --output_file datasets/drug_qa_sep_surround_answer0.sp/val.source &
python spm_encode.py --input_file datasets/tmp/val.target --output_file datasets/drug_qa_sep_surround_answer0.sp/val.target &
