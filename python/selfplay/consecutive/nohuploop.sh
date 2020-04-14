#!/bin/bash -eu
BASEDIR="$HOME"/selfplay
SCRIPTDIR="../.."
KATAEXEC="../../../cpp/katago"
THREADS=12
TRAININGNAME="freshtwenty"
MODELKIND="b20c256"

nohup ./blockingloop.sh "$BASEDIR" "$SCRIPTDIR" "$KATAEXEC" "$THREADS" "$TRAININGNAME" "$MODELKIND" >> log_all.txt 2>&1 &
echo $! > save_pid.txt
#if you want to stop the loop without rebooting the machine run the following command
#kill `cat save_pid.txt`
