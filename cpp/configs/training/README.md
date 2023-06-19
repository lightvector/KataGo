## Self-play training configs

**See `selfplay1.cfg` for inline comments describing some of the game initialization parameters which to avoid duplication are not replicated in many of the other configs.** Feel free to also diff these various configs with each other to see what the important differences are, which could be the parameters that you want to adjust or consider yourself.

#### Brief notes about what these configs are:

* `selfplay1.cfg`, `selfplay2.cfg`, `selfplay8a.cfg` - Example configs with reasonable parameters for very early training, for machines with 1, 2, 8 GPUs. Uses lower visits (600 full / 100 cheap), and trains more frequently on smaller boards than 19x19 to generate more data quickly. Historically was used for 6-block and early 10-block training in KataGo, except this config also has improved search parameters that didn't exist back then.
* `selfplay1_maxsize9.cfg` - The same as `selfplay1.cfg`, but different board size distribution including only boards of size 9 and smaller.
* `selfplay8b.cfg` - 8 GPU early training config after selfplay8a.cfg. More visits (1000 full / 200 cheap), more 19x19 boards, and minor other parameter adjustments. Historically was used for 10-block and 15-block training in KataGo, except this config also has improved search parameters that didn't exist back then.
* `selfplay8b20.cfg` - Same as `selfplay8b.cfg` but fewer threads and smaller batch size. Historically was used for 20-block nets that require less batching to be GPU-efficient, except this config also has improved search parameters that didn't exist back then.
* `selfplay8midrun.cfg` - 8 GPU training config that resembles a config historically used for much of KataGo's main public distributed training run, including 40-block training and 60-block training. More visits (1500 full / 250 cheap), more 19x19 boards, higher early game root policy temperature, different CPUCT to adjust policy convergence, and a variety of changes to game and komi initialization.
* `selfplay8mainb18.cfg` - 8 GPU training config that resembles a config used for KataGo's public distributed training run after switching to b18c384nbt architecture. More visits to compensate the lighter architecture (2000 full / 350 cheap), more heavy-tailed komi randomization, adjusted cpuct parameters to compenste for visits change.

* `gatekeeeper*.cfg` - These configs are loosely analogous to some of the above except used for the gatekeeper. Unlike selfplay configs, these aren't very important, since pretty much any reasonable config will distinguish between significantly better and worse nets in a training run. Also, runs work fine and often are more efficient without a gatekeeper at all, although a gatekeeper can be nice when starting out a run to help debugging and make sure that nets are actually improving.

