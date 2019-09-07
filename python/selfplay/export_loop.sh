#!/bin/bash -eu

while true
do
    ./selfplay/export_model_for_selfplay.sh "$@"
    sleep 10
done
