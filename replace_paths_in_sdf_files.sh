#!/bin/bash

OLD_PATH="/home/ubuntu/lbm_eval/"
NEW_PATH="/data/maxshen/lbm_eval/"

find /data/maxshen/lbm_eval/lbm_eval_models -name "*.sdf" -exec \
  sed -i "s|${OLD_PATH}|${NEW_PATH}|g" {} +
