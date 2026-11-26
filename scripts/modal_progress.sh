#!/bin/bash
# Show latest training iteration per active rltrading Modal app.
for a in $(modal app list --json 2>/dev/null | python -c 'import sys,json; [print(x["app_id"]) for x in json.load(sys.stdin) if x["description"]=="mutrading" and x["stopped_at"] is None]'); do
  echo "== $a"
  timeout 25 modal app logs $a 2>/dev/null | grep -a -E 'Training Iteration|DONE|TEST|Traceback|Error' | tail -6 | cut -c1-140
done
