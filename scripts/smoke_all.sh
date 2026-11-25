#!/bin/bash
# Quick crash test of every algorithm (tiny budget). Run from ~/rlt.
for al in ${@:-efficientzero gumbel_muzero stochastic_muzero sampled_muzero sampled_efficientzero unizero}; do
  timeout 1200 python -W ignore -m rltrading.train_lz --algo $al --max_env_step 1200 --eval_freq 100000 \
     --ckpt_every 200 --tag _smoke --exp_root smoke > logs/smoke_$al.log 2>&1
  echo "$al rc=$? $(tail -1 logs/smoke_$al.log | cut -c1-200)"
done
