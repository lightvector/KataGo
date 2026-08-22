KataGo v1.17.2
https://github.com/lightvector/KataGo

For neural nets from the latest run, download from here:
https://katagotraining.org/
For nets from earlier runs, see:
https://katagoarchive.org/

For the human SL net (b18c384nbt-humanv0.bin.gz), it was released with:
https://github.com/lightvector/KataGo/releases/tag/v1.15.0

For differences between this version and older versions, see releases page at
https://github.com/lightvector/KataGo/releases/

On Linux, depending on versions of your libraries and some issues with the libzip library, there may sometimes be problems
getting the precompiled executables to work. However, on Linux, KataGo is usually relatively straighforward to compile from source.
See: https://github.com/lightvector/KataGo/blob/master/Compiling.md

-----------------------------------------------------
USAGE:
-----------------------------------------------------
KataGo is just an engine and does not have its own graphical interface. So generally you will want to use KataGo along with a GUI or analysis program.
(https://github.com/lightvector/KataGo#guis)

FIRST: Run a command like this to make sure KataGo is working, with the neural net file you downloaded. On OpenCL, it will also tune for your GPU.

./katago benchmark                                                   # if you have default_gtp.cfg and default_model.bin.gz
./katago benchmark -model <NEURALNET>.bin.gz                         # if you have default_gtp.cfg
./katago benchmark -model <NEURALNET>.bin.gz -config gtp_custom.cfg  # use this .bin.gz neural net and this .cfg file

It will tell you a good number of threads. Edit your .cfg file and set "numSearchThreads" to that many to get best performance.

OR: Run this command to have KataGo generate a custom gtp config for you based on answering some questions:

./katago genconfig -model <NEURALNET>.bin.gz -output gtp_custom.cfg

NEXT: A command like this will run KataGo's engine. This is the command to give to your [GUI or analysis program](#guis) so that it can run KataGo.

./katago gtp                                                   # if you have default_gtp.cfg and default_model.bin.gz
./katago gtp -model <NEURALNET>.bin.gz                         # if you have default_gtp.cfg
./katago gtp -model <NEURALNET>.bin.gz -config gtp_custom.cfg  # use this .bin.gz neural net and this .cfg file

You may need to specify different paths when entering KataGo's command for a GUI program, e.g.:

path/to/katago gtp -model path/to/<NEURALNET>.bin.gz
path/to/katago gtp -model path/to/<NEURALNET>.bin.gz -config path/to/gtp_custom.cfg

KataGo should be able to work with any GUI program that supports GTP, as well as any analysis program that supports Leela Zero's `lz-analyze` command, such as Lizzie (https://github.com/featurecat/lizzie) or Sabaki (https://sabaki.yichuanshen.de/).

-----------------------------------------------------
HUMAN-STYLE PLAY AND ANALYSIS:
-----------------------------------------------------

You can also have KataGo imitate human play if you download the human SL model b18c384nbt-humanv0.bin.gz from https://github.com/lightvector/KataGo/releases/tag/v1.15.0, and run a command like the following, providing both a normal model and the human SL model:

./katago.exe gtp -model <NEURALNET>.bin.gz -human-model b18c384nbt-humanv0.bin.gz -config gtp_human5k_example.cfg

The gtp_human5k_example.cfg configures KataGo to imitate 5-kyu-level players. You can change it to imitate other ranks too, as well as to do many more things, including making KataGo play in a human style but still at a strong level or analyze in interesting ways. Read the config file itself for documentation on some of these possibilities!

And see also this guide to using the human SL model, which is written from the perspective of the JSON-based analysis engine mentioned below, but is also applicable to gtp as well.
https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide

-----------------------------------------------------
OTHER THINGS YOU CAN DO:
-----------------------------------------------------

Run a JSON-based analysis engine (https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md) that can do efficient batched evaluations for a backend Go service:

./katago analysis -model <NEURALNET>.gz -config <ANALYSIS_CONFIG>.cfg

Run a high-performance match engine that will play a pool of bots against each other sharing the same GPU batches and CPUs with each other:

./katago match -config <MATCH_CONFIG>.cfg -log-file match.log -sgf-output-dir <DIR TO WRITE THE SGFS>

Force OpenCL tuner to re-tune:

./katago tuner -config <GTP_CONFIG>.cfg

Print version:

./katago version

-----------------------------------------------------
TUNING FOR PERFORMANCE:
-----------------------------------------------------
You will very likely want to tune some of the parameters in `default_gtp.cfg` for your system for good performance, including the number of threads, fp16 usage, NN cache size, pondering settings, and so on. You can also adjust things like KataGo's resign threshold or utility function. Most of the relevant parameters should be be reasonably well documented directly inline in that config.

There are other a few notes about usage and performance at : https://github.com/lightvector/KataGo

-----------------------------------------------------
TROUBLESHOOTING:
-----------------------------------------------------
Some common issues are described here:
https://github.com/lightvector/KataGo#common-causes-of-errors

Or, feel free to hop into the Computer Go discord chat, which has become a general chatroom for a variety of computer Go hobbyists and users, and which you can often find people willing to help.
https://discord.gg/fhDHgfk




