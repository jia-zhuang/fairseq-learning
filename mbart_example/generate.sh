#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

fairseq-generate databin/car_spo \
    --path models/car_spo_debug2/checkpoint12.pt \
    --task translation_from_pretrained_bart \
    --gen-subset valid \
    --source-lang zh_CN --target-lang en_XX \
    --batch-size 16 --langs $langs \
    > output12.txt
