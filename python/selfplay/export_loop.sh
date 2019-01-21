#!/bin/bash -eu

while true
do
    ./selfplay/export_model_for_selfplay.sh $1
    sleep 10
done
