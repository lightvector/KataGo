#!/bin/sh
BASEDIR="$HOME"/code/backup
SCRIPTDIR="$HOME"/code/kata/python
KATAEXEC="./katago"
THREADS=12
TRAININGNAME="freshtwenty"
MODELKIND="b20c256"

nohup ./blockingloop.sh "$BASEDIR" "$SCRIPTDIR" "$KATAEXEC" "$THREADS" "$TRAININGNAME" "$MODELKIND" >> log_all.txt 2>&1 &
