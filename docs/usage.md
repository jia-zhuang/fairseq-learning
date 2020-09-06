# 使用

详细参考 [fairseq 官方文档](https://fairseq.readthedocs.io/en/latest/getting_started.html)

## 简答使用

#### 使用预训练模型

下载

```
curl -O https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2

tar -xvjf wmt14.v2.en-fr.fconv-py.tar.bz2

ls wmt14.en-fr.fconv-py

# 显示5个文件
bpecodes  dict.en.txt  dict.fr.txt  model.pt  README.md
```

使用 `fairseq-interactive` 可交互式翻译：

```
> MODEL_DIR=wmt14.en-fr.fconv-py
> fairseq-interactive \
    --path $MODEL_DIR/model.pt $MODEL_DIR \
    --beam 5 --source-lang en --target-lang fr \
    --tokenizer moses \
    --bpe subword_nmt --bpe-codes $MODEL_DIR/bpecodes
| loading model(s) from wmt14.en-fr.fconv-py/model.pt
| [en] dictionary: 44206 types
| [fr] dictionary: 44463 types
| Type the input sentence and press return:
Why is it rare to discover new marine mammal species?
S-0     Why is it rare to discover new marine mam@@ mal species ?
H-0     -0.0643349438905716     Pourquoi est-il rare de découvrir de nouvelles espèces de mammifères marins?
P-0     -0.0763 -0.1849 -0.0956 -0.0946 -0.0735 -0.1150 -0.1301 -0.0042 -0.0321 -0.0171 -0.0052 -0.0062 -0.0015
```

有几个值得注意的地方：

- `--path`：用来指定模型的路径
- 后面一个目录($MODEL_DIR) 用来寻找字典文件
- `@@ ` 是 sub-word 分词，类似 BERT 里的 word-piece

#### 数据预处理

```bash
TEXT=data/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en
```

#### 训练

```bash
mkdir -p checkpoints/fconv
CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
  --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --arch fconv_iwslt_de_en --save-dir checkpoints/fconv
```

#### 生成

```bash
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/fconv/checkpoint_best.pt \
    --batch-size 128 --beam 5
```
