# Consecutive training using one machine

You can train your own network using these scripts on one machine, by switching from selfplay to training instead of training in parallel.

## Instructions

Either copy the selfplay dir to another dir or run the script in place. Copy a relevant configuration file and rename it `keeper1.cfg` to `consecutive`. Copy a relevant selfplay configuration file and rename it `selfplay1.cfg` to the same dir. Edit the paths in `nohuploop.sh` to reflect the correct location of your selfplay data and katago executable.

## Suggested settings for learning

In `shuffle.sh` you should edit the min rows to 10000 because we're going to be working with less data to start. Expand window per row should be set to 1, taper window exponent to 1. In `train.py` you should decrease the samples per epoch to 200000 since we will generate fewer games to have a smaller learning loop. The samples per epoch should be similar to the amount of rows in your shuffle, since this script already runs learning on the same data three times at different learning rates. In your own copy of `selfplay1.cfg` you should set the amount of games to 3000 or so to get 200000 rows of data every shuffle, keeping cheap visits on. I suggest you reduce the visits in `keeper1.cfg` down to 100 or lower, because it cuts into your selfplay time significantly.

## Modifications for smaller board sizes.

In both config files you should change the settings. Set `bSizes` to the board size you wish to train, `bSizeRelProbs` should be set to `1`. `handicapProb` should be `0.0` on very small boards if you don't want it to learn lopsided situations.

You can modify `-pos-len` in `train.sh` to the biggest size you are going to train for.

## How to run

Just run the `./nohuploop.sh` script and you can close the terminal - it will run in the background. You can `tail -f log_all.txt` to check how it's going at any point in time and `cat gatekeeper.txt` to see the score of the latest test. The old selfplays will be stored in `selfplay_old` so if you want to train a larger model in the future you should train on these.

## Backing up logs

Moving the files with `mv` won't work since the script will still write to the same physical address on the disk. But you can `cp` the file to another location and empty the current one.

## How to stop

The script writes to `save_pid.txt` which has the PID of the script process running in the background. You can use the `kill` command to stop it. It will finish the last step it's doing, since this only ends the outer loop.
