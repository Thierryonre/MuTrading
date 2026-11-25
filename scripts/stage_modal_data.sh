#!/bin/bash
# Copy pre-built feature arrays to the dir that modal_app.py ships into the training image.
mkdir -p ~/rlt/modal_data && rm -f ~/rlt/modal_data/* && cp "${RLT_DATA_DIR:-$HOME/rlt/data}"/cache/arrays_* ~/rlt/modal_data/ && ls -la ~/rlt/modal_data
