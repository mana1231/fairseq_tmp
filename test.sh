#!/bin/bash

mkdir -p data-bin
mkdir -p tmp
curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
curl https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
CUDA_VISIBLE_DEVICES=1 uv run fairseq-generate data-bin/wmt14.en-fr.newstest2014 \
    --path data-bin/wmt14.en-fr.fconv-py/model.pt \
    --beam 5 --batch-size 128 --remove-bpe | tee ./tmp/gen.out
# ...
# | Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
# | Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Compute BLEU score
grep ^H ./tmp/gen.out | cut -f3- >./tmp/gen.out.sys
grep ^T ./tmp/gen.out | cut -f2- >./tmp/gen.out.ref
uv run fairseq-score --sys ./tmp/gen.out.sys --ref ./tmp/gen.out.ref
# BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)