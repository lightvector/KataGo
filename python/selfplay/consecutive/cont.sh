#!/bin/sh
#continues running the script where it left off, as long as the PID still exists
kill -CONT `cat save_pid.txt`
