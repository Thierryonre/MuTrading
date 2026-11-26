#!/bin/bash
# Run train+backtest jobs locally, at most $MAXJ concurrently. One job per line in $1:
#   <log_name> <train_lz args...>
# Run from ~/rlt with the rlt env active.
MAXJ=${MAXJ:-3}
while read -r name args; do
  [ -z "$name" ] && continue
  while [ "$(pgrep -fc '[r]ltrading.train_lz')" -ge "$MAXJ" ]; do sleep 60; done
  (
    out=$(python -W ignore -m rltrading.train_lz $args --threads 2 2>&1 | tee logs/$name.log | grep -a '^DONE' | awk '{print $2}')
    if [ -n "$out" ]; then python -W ignore -m rltrading.backtest --exp "$out" > logs/bt_$name.log 2>&1; fi
    echo "finished $name $out"
  ) &
  sleep 30
done < "$1"
wait
echo QUEUE_DONE
