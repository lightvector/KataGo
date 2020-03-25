# Consecutive training using one machine

You can train your own network using these scripts on one machine, by switching from selfplay to training instead of training in parallel.

## Instructions

Either copy the selfplay dir to another dir or run the script in place. Copy a relevant configuration file and rename it `keeper1.cfg` to `consecutive`. Copy a relevant selfplay configuration file and rename it `selfplay1.cfg` to the same dir. Edit the paths in `nohuploop.sh` to reflect the correct location of your selfplay data and katago executable.

## Suggested settings for learning

In `shuffle.sh` you should edit the min rows to 50000 because we're going to be working with less data to start. I suggest you uncomment the cycling learning rates in the `blockingloop.sh` script. In `train.py` you should decrease the samples per epoch to 200000 since we will generate fewer games to have a smaller learning loop. You should also reduce the maximum rows to 200000 in `shuffle.py`. The samples per epoch should be similar to the amount of rows in your shuffle, since this script already runs learning on the same data three times at different learning rates. I suggest you reduce the visits in `keeper1.cfg` down to 100 or lower, because it cuts into your selfplay time significantly.

You have to decide on the amount of times you will generate a network on each 200K rows. At first, you'll want multiple networks to be generated in this time, as the extra learning is beneficial at the start of the run, and each one will be stronger. So maybe set the max games in `selfplay1.cfg` to a number that gives 50K usable rows in the beginning, keep the shuffle rows at 200K. You'll generate ~4 networks on each 200K rows of data that way, but you can try different values. When you get more gatekeeping fails, you increase the number of games to generate fewer networks (generating them too often is not efficient). Once you hit one network per each 200K window, expand window per row should be set to 1, taper window exponent to 1. You can uncomment the `rsync` line in the script to learn on each set of games exactly once.

This is a smaller learning window than the KataGo main runs. I've tried every smaller one, but anything below 200K had overfitting problems. As the network gets even stronger you may need to train on a larger window size.

## Increasing the network size

The way the main KataGo run does it is the new network is trained on existing training data. I've had success using only 2 million rows to train a network from scratch. It took me 50+ nets to achieve the previous level. I used cycling learning rates, but I haven't actually compared the results of just pure training on existing data. The reason I use cycling learning rates is that existing networks are stronger right after a learning rate cut, so I always do one epoch at cut learning rates.

## Modifications for smaller board sizes

In both config files you should change the settings. Set `bSizes` to the board size you wish to train, `bSizeRelProbs` should be set to `1`. `handicapProb` should be `0.0` on very small boards if you don't want it to learn lopsided situations.

You can modify `-pos-len` in `train.sh` to the biggest size you are going to train for.

## How to run

Just run the `./nohuploop.sh` script and you can close the terminal - it will run in the background. You can `tail -f log_all.txt` to check how it's going at any point in time and `cat gatekeeper.txt` to see the score of the latest test. The old selfplays will be stored in `selfplay_old` so if you want to train a larger model in the future you should train on these.

## Backing up logs

Moving the files with `mv` won't work since the script will still write to the same physical address on the disk. But you can `cp` the file to another location and empty the current one.

## How to stop

The script writes to `save_pid.txt` which has the PID of the script process running in the background. You can use the `kill` command to stop it. It will finish the last step it's doing, since this only ends the outer loop. I have provided a `stop.sh` loop to temporarily halt it and `cont.sh` to resume from the last step it completed. That way you can use the machine for something else before continuing selfplay.
