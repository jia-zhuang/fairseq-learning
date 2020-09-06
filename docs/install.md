# 安装

fairseq 可通过 pip 安装，有两种安装方式。

### 常规安装

```bash
pip install fairseq
```

会将源码安装到 python 的第三方 packages 目录下(即 `/path/to/python/lib/python3.6/site-packages/`)，安装后该目录下会多出如下目录：

- fairseq：源文件主目录
- fairseq_cli：命令行文件目录
- fairseq-0.9.0-py3.6.egg-info：包信息目录

并且在 python 的命令目录下(即`/path/to/python/bin`)会多出一系列 fairseq 相关命令：

- fairseq-preprocess
- fairseq-train
- fairseq-generate
- fairseq-validate
- fairseq-score
- fairseq-eval-lm
- fairseq-interactive

### 源码可编辑模式安装

```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

fairseq 的源码就在当前目录下，只会在 python 第三方 packages 目录下(即 `/path/to/python/lib/python3.6/site-packages/`)生产一个名为 `fairseq.egg-link` 链接文件，文件内有一个路径指向 fairseq 的源码位置。另外也会在 python 的命令目录下生产 fairseq 相关的命令文件。这种安装方式，可以直接在 fairseq 的源码目录中做修改，并且能直接生效，方便做开发。