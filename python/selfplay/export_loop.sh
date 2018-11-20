#!/bin/bash -eu

while true
do
    ./selfplay/export_modelv3_for_selfplay.sh $1
    sleep 120
done
