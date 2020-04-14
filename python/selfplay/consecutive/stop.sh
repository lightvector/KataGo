#!/bin/bash -eu
#Temporarily halts the script after the current command finishes
kill -STOP `cat save_pid.txt`
