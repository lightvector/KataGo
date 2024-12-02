#!/bin/bash

while true; do
    python flask_wrapper_human_sl_server.py
    echo "The server crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
