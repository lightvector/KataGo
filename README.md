# Go Neural Net Sandbox

This repo is currently a sandbox for personal experimentation in neural net training in Go. I've only put a little work into making this repo usable by others, since this is foremost a personal sandbox, but if you're interested, here's how to get started:

   * You must have [Python3](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/install/) installed.
   * Either download a pre-trained model (https://github.com/lightvector/GoNN/releases) or train one yourself (see below).
   * Do things with the trained model
      * Run a GTP engine using the model to make moves directly: `./play.py -model <MODELS_DIRECTORY>/model<EPOCH>`
      * Dump one of the raw weight matrices: `./visualize.py -model-file <MODELS_DIRECTORY>/model<EPOCH> -dump p2/w:0`
      * Directly run a model on a position from an sgf file: `./eval_sgf.py -model-file <MODELS_DIRECTORY>/model<EPOCH> -sgf SGFFILE -move TURNNUMBER`

See LICENSE for software license. License aside, informally, if do you successfully use any of the code or any wacky ideas about neural net structure explored in this repo in your own neural nets or to run any of your own experiments, I would to love hear about it and/or might also appreciate a casual acknowledgement where appropriate. Yay.

### Training a Model

If you'd like to train your own model and/or experiment with the architectures in this repo, here are the steps.
   * You must have HDF5 installed for C++ (https://support.hdfgroup.org/HDF5/release/obtainsrc.html).
   * Training consists of converting .sgf files into training rows written in HDF5 format, then reading that HDF5 file in Python to train using numpy and h5py to feed them to Tensorflow.
   * The utility that converts .sgf to .h5 is written in C++. Compile it using CMake and make in the cpp directory, which expects you to have h5c++
     available. (on windows you can also use the deprecated compile.sh):
      * `cd cpp`
      * `cmake .`
      * `make`
   * Run the compiled `write` executable with appropriate flags on a directory of SGF files to generate an h5 file of preprocessed training data.
      * Example: `./write -pool-size 50000 -train-shards 10 -val-game-prob 0.05 -gamesdir <DIRECTORY WITH SGF FILES> -output <TARGET>.h5` - writes an h5 file making 10 passes over the games with a running buffer of 50000 rows to randomize the order of outputted rows, reserving 5% of games as a validation set.
   * Back up in the root directory, run train.py using that h5 file to train the neural net.
      * Example: `./train.py -traindir <DIRECTORY TO OUTPUT MODELS AND LOGS> -gamesh5 <H5FILE>.h5`
      * By default, it considers 1 million samples to be one epoch. If you want to see initial results more quickly you can pass `-fast-factor N` to divide this and a few other things by N, and `-validation-prop P` to only use P proportion of the validation set, but you will probably want to manually modify train.py (see `TRAINING PARAMETERS` section) as well as model.py (to try a much smaller neural net first).
      * Depending on how much training data you have and how long you're willing to train for, you probably want to edit the `knots` in train.py that determine the learning rate schedule.
      * If you getting errors due to running out of GPU memory, you probably want to also edit train.py to shrink the `batch_size` and/or the `validation_batch_size`.

# Experimental Notes
You can see the implementations of the relevant neural net structures in "model.py", although I may adapt and change them as time goes on.

### History
   * May-June 2018 - No significant architectural changes, but [added player ranks](https://github.com/lightvector/GoNN#ranks-as-an-input-june-2018) as an input feature to the neural net and included a lot of amateur games in the training set. Filtered pro games with the resulting net to find instructive positions for players of different ranks, producing a neat collection of Go problems: [neuralnetgoproblems.com](https://neuralnetgoproblems.com). Cleaned up a few input features and tried a slightly larger net with more training.
   * Apr 2018 - Added a row to the [current results](https://github.com/lightvector/GoNN#current-results) reflecting the large improvement from embedding global pooled properties in the [middle of the neural net](https://github.com/lightvector/GoNN#update-mar-2018) rather than only the policy head, along with some minor adjustments to learning rates and other tweaks.
   * Mar 2018 - Much larger neural nets and updates to [current results](https://github.com/lightvector/GoNN#current-results). Global pooled properties [are good in the main trunk of the resnet as well](https://github.com/lightvector/GoNN#update-mar-2018)! Also, [increasing center-position learning rates](https://github.com/lightvector/GoNN#update-mar-2018-1) everywhere else in the net helps training speed a little. Promising experiments with [dilated convolutions](https://github.com/lightvector/GoNN#dilated-convolutions-mar-2018), and a note about [making neural nets not always need history](https://github.com/lightvector/GoNN#some-thoughts-about-history-as-an-input-mar-2018).
   * Feb 2018 - Tried [special ladder blocks in the policy](https://github.com/lightvector/GoNN#update-feb-2018), tried [ladders as a training target](https://github.com/lightvector/GoNN#using-ladders-as-an-extra-training-target-feb-2018), retested [global pooled properties](https://github.com/lightvector/GoNN#update-feb-2018-1). And ran new experiments with net architecture - [wide low-rank residual blocks](https://github.com/lightvector/GoNN#wide-low-rank-residual-blocks-feb-2018), [parametric ReLUs](https://github.com/lightvector/GoNN#parametric-relus-feb-2018), [chain pooling](https://github.com/lightvector/GoNN#chain-pooling-feb-2018), and some [observations on redundancy in models](https://github.com/lightvector/GoNN#redundant-parameters-and-learning-rates-feb-2018)
   * Dec 2017 - Initial results and experiments - [special ladder residual blocks](https://github.com/lightvector/GoNN#special-ladder-residual-blocks-dec-2017), [global pooled properties](https://github.com/lightvector/GoNN#global-pooled-properties-dec-2017)

## Current Results
As of the end of April 2018, the best neural nets I've been training from this sandbox have been quite good at matching or exceeding results I've seen published elsewhere, presumably due to the combination of the various enhancements discussed below. See this table for a summary of results (bolded) in comparison with other published results:

| Neural Net | Structure | Params | KGS Top1 | GoGoD Top1 | Training Steps | Vs GnuGo | Vs Pachi
|------|---|---|---|---|---|---|---|
| [Clark and Stokey (2015)](https://arxiv.org/abs/1412.3409)  | CNN 8xVarious | ~560,000 |  44.4% | 41.1% | 147M | 87% (172/200)
| [Maddison et al. (2015)](https://arxiv.org/abs/1412.6564) |  CNN 12x(64-192) | ~2,300,000 |  55.2% |  | 685M x 50 + 82M | 97% (/300?) | 11% (/220?)
| [AlphaGoFan192 (2016)](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) | CNN 13x192 | 3,880,489 | 55.4% | | 340M x 50 | | 85% (w/RL)
| [AlphaGoFan256 (2016)](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) | CNN 13x256 | 6,795,881 | 55.9% | |
| **This Sandbox (Mar 2018)** | **ResNet 5x192** | **3,597,792** | **56.8%** | **52.5%** | **205-209M** | **99.2% (496/500)** |
| [Darkforest (2016)](https://arxiv.org/abs/1511.06410) | CNN 12x384  | 12,329,756  | 57.1%  |   | 128M | 100% (300/300) | 72.6% (218/300)
| [Cazenave (2017)](http://www.lamsade.dauphine.fr/~cazenave/papers/resnet.pdf) | ResNet 10x256 | 12,098,304 | 58.2% | 54.1% | 380M |
| **This Sandbox (Mar 2018)** | **ResNet 12x192** | **8,250,720** | **58.2%** | **54.2%** | **325-333M** | **100% (500/500)** | **90.0% (45/50)**
| [Cazenave (2017)](http://www.lamsade.dauphine.fr/~cazenave/papers/resnet.pdf) | ResNet 14x256 | 16,816,896 |       | 54.6% | 355M |
| **This Sandbox (Apr 2018)** | **ResNet 12x192** | **8,057,424** | **58.6%** | **54.7%** | **201M-209M** |  |
| **This Sandbox (Jun 2018)** | **ResNet 12x224** | **10,950,416** | | **55.3%** | **301M** |  |
| [AlphaGoZero(2017)](https://deepmind.com/documents/119/agz_unformatted_nature.pdf) | ResNet 20x256 | 22,837,864 | 60.4% | | >1000M? |

As seen in the above table, thanks to the various enhancements I've been playing with, the neural nets in this sandbox compare quite favorably, matching or surpassing accuracy results for neural nets with larger numbers of parameters. They require fewer training steps to achieve these results and are probably computationally faster to evaluate per step (I currently don't use any techniques such as rotational equivariance or other weight representations that reduce the number of parameters in a way that doesn't also reduce the computation cost).

However, one thing to keep in mind is that not all of the results in this table are precisely comparable. In the above papers, different authors often made different choices about the date range, what range of player ranks to allow, what version of the dataset to use (GoGoD has biannual updates), and whether to include or exclude handicap games. Overall, I think most of these choices shouldn't affect the results too much, but I would guess could affect accuracy numbers by a reasonable fraction of a percent.

For the results above, I tried to closely match the choices made by Cazenave (2017):
   * GoGoD - I used the Summer 2016 edition including all 19x19 games from 1900 to 2014 as the training set, and used 2015 to 2016 as the test set.
   * KGS - I used the "at least 7 dan or both players at least 6 dan" data set from https://u-go.net/gamerecords/ from 2000 to 2014 as the training set, and 2015 as the testing set. I excluded handicap games and filtered only to moves by players 6d or stronger (in rare cases, a 5d or weaker player might play a 7d or stronger player in a non-handicap game).

I also tested the GoGoD-trained neural nets by direct play via GnuGo 3.8 (level 10) and Pachi 11.00 (100k playouts). I generated moves by converting the net's policy prediction into a move choice by sorting the predictions by probability then selecting from a biased distribution that would disproportionately favor the moves with more probability. The distribution depended on the move number of the game, early on favoring diversity and later becoming more deterministic (see play.py for the exact algorithm). The 12 block net won every single time in 500 against GnuGo, while the 5-block net won more than 99% of games. The 12-block net also proved much stronger than Pachi, winning about 90% of games.

#### Choice of Loss function
There is also one further interesting choice that some of the above papers differ on, which is whether to use cross entropy or L2 loss when training the neural net.

Although I haven't performed detailed experiments, in at least two of the training runs for my own neural nets to produce results for the above table, I found that using L2 loss increased the test set Top1 accuracy by about 0.2% while decreasing the Top4 accuracy by about 0.2% and the average log likelihood (i.e. the cross entropy loss itself) by about 0.005 nats. Directionally, this is in line with what one might expect from theoretical priors. L2 loss "cares" relatively more about the total probability mass of correct classifications in cases with a few choices, while cross-entropy "cares" more about avoiding putting too small of probability on surprising moves. So it's not surprising that L2 does slightly better at Top1 while cross entropy does slightly better in the tail. For a nice discussion of loss functions, see also: https://arxiv.org/pdf/1702.05659.pdf

For the above results I initially used cross-entropy loss, except that for the 12 block KGS and GoGoD neural nets partway through when I realized the L2/cross-entropy difference between the various papers, I switched to L2 loss for a better comparison with Cazenave's results which also use L2 loss. For each loss function, I briefly by hand tried a few different learning rate factors to see roughly what worked best, since a little bit of learning rate tuning is needed to deal with the fact that L2 and cross entropy are a small constant factor different in typical magnitude and gradient.

#### Architecture

The neural nets I trained consisted of a 5x5 convolution, followed by 5 to 12 residual blocks with a main trunk of 192 channels and with 192 channels in each residual block, followed by policy head consisting of a 3x3 convolution and global pooling features, and then a final 5x5 convolution and a softmax output. I used batch normalization in a "pre-activation" scheme between each convolution. The neural nets were also further augmented with many of the enhancements described below, the important of which were the global pooling and parametric relus, and secondarily the dilated convolutions and the position-dependent learning rates inspired by "redundant" parametrizations. See model.py for the precise architecture.

#### Other Stats

For interest, here are a few more stats beyond the ones most consistently shared and reported by other papers for the current neural nets:

| Structure | Dataset | Params | Top1 Accuracy | Top4 Accuracy | Cross-Entropy Loss (nats) | Training Steps |
|-----|---|---|---|---|---|---|
| ResNet 5 Blocks | GoGoD | 3,597,792 | 52.5% | 81.6% | 1.609 | 209M |
| ResNet 12 Blocks (Mar 2018) | GoGoD | 8,250,720 | 54.2% | 82.8% | 1.542 | 325M |
| ResNet 12 Blocks (Apr 2018) | GoGoD | 8,057,424 | 54.7% | 83.8% | 1.496 | 209M |
| ResNet 5 Blocks | KGS | 3,597,792 | 56.8% | 85.4% | 1.429 | 205M |
| ResNet 12 Blocks (Mar 2018) | KGS | 8,250,720 | 58.2% | 86.2% | 1.378 | 329M |
| ResNet 12 Blocks (Apr 2018) | KGS | 8,057,424 | 58.6% | 86.6% | 1.346 | 201M |

Unfortunately, many papers don't report the cross entropy loss, which is a shame since its values are very nicely interpretable. For example, the cross entropy of 1.378 nats for the 12 block KGS ResNet corresponds to a perplexity of exp(1.378) = 3.967, which means that for KGS on average the neural net has achieved the same entropy as if on every move it could pin down the next move uniformly to about 4 possible moves.

Also, to get an idea of how top-1 accuracy and nats and direct playing strength correlate, at least when trained on GoGoD and for these kinds of neural nets and for the particular way I used the policy net used to choose moves, here's a table showing the rough correspondence I observed as the neural nets in this sandbox gradually grew larger and improved. Elos were determined very roughly by a mix of self-play between various adjacent versions and testing vs GnuGo 3.8, totalling a few thousand games.

| GoGoD Accuracy (Top 1) | GoGoD Cross Entropy Loss (nats) | Relative Elo |
|-----|---|---|
| 34% | 2.73 | 0
| 37% | 2.50 | ~350
| 40% | 2.31 | ~700
| 45% | 1.95 | ~950
| 47% | 1.85 | ~1050
| 49% | 1.77 | ~1150
| 51% | 1.67 | ~1300
| 53% | 1.60 | ~1450
| 54% | 1.55 | ~1600

#### Next Ideas

As of March 2018, I'm happy with the above results and plan to move on to some experiments of other kinds, such as using neural nets to investigate systematic differences between players of different ranks. Or, maybe investigating if there is a reasonable self-play process for generating training games for a value net for territory scoring rather than area scoring. Given that white has a small but systematic advantage at 7.5 komi, and pro-game evidence from the era of 5.5 komi shows that black has a clear edge there, I'm curious who AlphaZero-strength bots would favor at 6.5 komi, if only they had the finer granularity of territory scoring where 6.5 komi would actually be meaningfully different than 7.5. Of course, I might revisit these results if I get ideas for more enhancements to try.


## Special Ladder Residual Blocks (Dec 2017)
Experimentally, I've found that neural nets can easily solve ladders, if trained directly to predict ladders (i.e. identify all laddered groups, rather than predict the next move)! Apparently 3 or 4 residual blocks is sufficient to solve ladders extending out up to 10ish spaces, near the theoretical max that such convolutions can reach. Near the theoretical max, they start to get a bit fuzzy, such as being only 70% sure of a working ladder, instead of 95+%, particularly if the ladder maker or ladder breaker stone is near the edge of the 6-wide diagonal path that affects the ladder.

However, specially-designed residual blocks appear to significantly help such a neural net detect solve ladders that extend well beyond the reach of its convolutions, as well as make it much more accurate in deciding when a stone nearly at the edge of the path that could affect the ladder actually does affect the ladder. This is definitely not a "zero" approach because it builds in Go-specific structure into the neural net, but nonetheless, the basic approach I tried was to take the 19x19 board and skew it via manual tensor reshaping:

    1234          1234000
    5678    ->    0567800
    9abc          009abc0
    defg          000defg

Now, columns in the skewed board correspond to diagonals on the original board. Then:

   * Compute a small number C of "value" and "weight" channels from the main resnet trunk via 3x3 convolutions.
   * Skew all the channels.
   * Compute a cumulative sum (tf.cumsum) along the skewed columns of both value*weight and weight, and divide to obtain a cumulative moving average.
   * Also repeat with reverse-cumulative sums and skewing the other way, to obtain all 4 diagonal directions.
   * Unskew all the results to make them square again.
   * Concatenate all the resulting 4C channels and multiply by a 4CxN matrix where N is the number of channels in the main trunk to transform the results back into "main trunk feature space".
   * Also apply your favorite activation function and batch norm at appropriate points throughout the above.
   * Add the results as residuals back to the main resnet trunk.

In the event that many of the weights are near zero, this will have the effect of propagating information potentially very long distances across the diagonals. In practice, I applied an exp-based transform to the weight channel to make it behave like an exponentially-weighted moving average, to obtain the effect that ladders care mostly about the first stone or stones they hit, and not the stones beyond them, as well as a bias to try to make it easier for the neural net to put low weight on empty spaces to encourage long-distance propagation.

Adding such a residual block to the neural net appears to greatly help long-distance ladder solving! When I trained a neural net with this to identify laddered groups, it appeared to have decently accurate ladder solving in test positions well beyond the theoretical range that its convolutions could reach alone, and I'm currently investigating whether adding this special block into a policy net helps the policy net's predictions about ladder-related tactics.

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/laddertarget.png" width="350" height="350"/></td></tr>
<tr><td><sub>Net correctly flags working ladders. No white stone is a ladder breaker.</sub></tr></td>
</table>
<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/laddertargetbroken.png" width="350" height="350"/></td></tr>
<tr><td><sub>Net correctly determines that ladders don't work.</sub></tr></td>
</table>

#### Update (Feb 2018):
Apparently, adding this block into the neural net does not cause it to be able to learn ladders in a supervised setting. From digging into the data a little, my suspicion is that in supervised settings, whether a ladder works or not is too strongly correlated with whether it gets formed in the first place, and similarly for escapes due to ladder breakers. So that unless it's plainly obvious whether the ladder works or not (the ladder-target experiments show this block makes it much easier, but it's still not trivial), the neural net fails to pick up on the signal. It's possible that in a reinforcement learning setting (e.g. Leela Zero), this would be different.

Strangely however, adding this block in *did* improve the loss, by about 0.015 nats at 5 million training steps persisting to still a bit more than 0.010 nats at 12 million training steps. I'm not sure exactly what the neural net is using this block for, but it's being used for something. Due to the bottlenecked nature of the block (I'm using only C = 6), it barely increases the number of parameters in the neural net, so this is a pretty surprising improvement in relative terms. So I kept this block in the net while moving on to later experiments, and I haven't gone back to testing further.

## Using Ladders as an Extra Training Target (Feb 2018)

In another effort to make the neural net understand ladders, I added a second head to the neural net and forced the neural net to simultaneously predict the next move and to identify all groups that were in or could be put in inescapable atari.

Mostly, this didn't work so well. With the size of neural net I was testing (~4-5 blocks, 192 channels) I was unable to get the neural net to produce predictions of long-distance ladders anywhere near as well as it did when it had the ladder target alone, unless I was willing to downweight the policy target in the loss function enough that it would no longer produce as-good predictions of the next move. With just a small weighting on the ladder target, the neural net learned to produce highly correct predictions for a variety of local inescapable ataris, such as edge-related captures, throw-in tactics and snapbacks, mostly everything except long-distance ladders. Presumably due to the additional regularization, this improved the loss for the policy very slightly, bouncing around 0.003-0.008 nats around 5 to 10 million training steps.

But simply adding the ladder feature directly as an input to the neural net dominated this, improving the loss very slightly further. Also, with the feature directly provided as an input to the neural net, the neural net was finally able to tease out enough signal to mostly handle ladders well.

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/extraladdertargetfail2.png" width="300" height="300"/></td><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/ladderinput2.png" width="300" height="300"/></td></tr>
<tr><td colspan="2"><sub>Extra ladder target training doesn't stop net from trying to run from working ladder (left), but directly providing it as an input feature works (right). </sub></td></tr>
</table>

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/ladderinput1.png" width="300" height="300"/></td><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/ladderinput3.png" width="300" height="300"/></td></tr>
<tr><td colspan="2"><sub>And on the move before, whether the ladder works clearly affects the policy.</sub></td></tr>
</table>

However, even with such a blunt signal, it still doesn't always handle ladders correctly! In some test positions, sometimes it fails to capture stones in a working ladder, or fails to run from a failed ladder, or continues to run from a working ladder, etc. Presumably pro players would not get into such situations in the first place, so there is a lack of data on these situations.

This suggests that a lot of these difficulties are due to the supervised-learning setting, rather than merely difficulty of learning to solve ladders. I'm quite confident that in a reinforcement-learning setting more like the "Zero" training, with ladders actually provided directly as an input feature, the neural net would rapidly learn to not make such mistakes. It's also possible that in such settings, with a special ladder block the neural net would also not need ladders as an input feature and that the block would end up being used for ladder solving instead of whatever mysterious use it's currently being put to.

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/ladderinput4.png" width="350" height="350"/></td></tr>
<tr><td><sub>Even when told via input feature that this ladder works, the NN still wants to try to escape.</sub></tr></td>
</table>


## Global Pooled Properties (Dec 2017)
Starting from a purely-convolutional policy net, I noticed a pretty significant change in move prediction accuracy despite only a small increase in the number of trainable parameters when I added the following structure to the policy head. The intent is to allow the neural net to compute some "global" properties of the board that may affect local play.
   * Separately from the main convolutions that feed into the rest of the policy head, on the side compute C channels of 3x3 convolutions (shape 19x19xC).
   * Max-pool, average-pool, and stdev-pool these C channels across the whole board (shape 1x1x(3C))
   * Multiply by (3C)xN matrix where N is the number of channels for the convolutions for the rest of the policy head (shape 1x1xN).
   * Broadcast the result up to 19x19xN and add it into the 19x19xN tensor resuting from the main convolutions for the policy head.

The idea is that the neural net can use these C global max-pooled or average-pooled channels to compute things like "is there currently a ko fight", and if so, upweight the subset of the N policy channels that correspond to playing ko threat moves, or compute "who has more territory", and upweight the subset of the N channels that match risky-move patterns or safe-move-patterns based on the result.

Experimentally, that's what it does! I tried C = 16, and when visualizing the activations 19x19xC in the neural net in various posititions just prior to the pooling, I found it had chosen the following 16 global features, which amazingly were mostly all humanly interpretable:
   * Game phase detectors (when pooled, these are all very useful for distinguishing opening/midgame/endgame)
       * 1 channel that activated when near a stone of either color.
       * 1 that activated within a wide radius of any stone. (an is-it-the-super-early-opening detector)
       * 1 that activated when not near a strong or settled group.
       * 1 that activated near an unfinished territoral border, and negative in any settled territory by either side.
   * Last-move (I still don't understand why these are important to compute to be pooled, but clearly the neural net thought they were.)
       * 5 distinct channels that activated near the last move or the last-last move, all in hard-to-understand but clearly different ways
   * Ko fight detector (presumably used to tell if it was time to play a ko threat anywhere else on the board)
       * 1 channel that highlighted strongly on any ko fight that was worth anything.
   * Urgency or weakness detectors (presumably used to measure the global temperature and help decide things like tenukis)
       * 4 different channels that detected various aspects of "is moving in this region urgent", such as being positive near weak groups, or in contact fights, etc.
   * Who is ahead? (presumably used to decide to play risky or safe)
       * 1 channel that activated in regions that the player controlled
       * 1 channel that activated in regions that the opponent controlled

So apparently global pooled properties help a lot. I bet this could also be done as a special residual block earlier in the neural net rather than putting it only in the policy head.

Here's a heatmap of the ko-fight detector channel, just prior to pooling. It activates bright green in this position after a ko was just taken. Everywhere else the heatmap is covered with checkerlike patterns, suggesting the kinds of shapes that the detector is sensitive to.

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/kothreatglobalproperty.png" width="350" height="350"/>

As a result of the pooling, the net predicts a distant ko threat:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/kothreatpolicy.png" width="350" height="350"/>

But when there is no ko, the net no longer wants to play ko threats:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/afterko.png" width="350" height="350"/>

#### Update (Feb 2018):

The original tests of global pooled properties was done when the neural net was an order of magnitude smaller than it is now (~250K params instead of ~3M params), so I did a quick test of removing this part of the policy head to sanity check if this was still useful. Removing it immediately worsened the loss by about 0.04 nats on the first several million training steps. Generally, differences early in training tend to diminish as the neural net converges further, but I would still guess at minimum 0.01 nats of harm at convergence, so this was a large enough drop that I didn't think it worth the GPU time to run it any longer.

So this is still adding plenty of value to the neural net. From a Go-playing perspective, it's also fairly obvious in practice that these channels are doing work. At least in ko fights, since the capture of an important ko plainly causes the neural net to suggest ko-threat moves all over the board that it would otherwise never suggest, including ones too far away to be easily reached by successive convolutions. I find it interesting that I haven't yet found any other published architectures include such a structure in the neural net.

Just for fun, here's some more pictures of channels just prior to the pooling. These are from the latest nets that use parametric ReLU, so the channels have negative values now as well (indicated by blues, purples, and magentas).

Despite only being trained to predict moves, this net has developed a rough territory detector! It also clearly understands the upper right white group is dead:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/territorydetector.png" width="350" height="350"/>

And this appears to be a detector for unfinished endgame borders:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/unfinishedborderdetector.png" width="350" height="350"/>

#### Update (Mar 2018):

I experimented with these just a little more and found that the "stdev" part of the pooling doesn't seem to help. Only the average and max components are useful. Eliminating the stdev component of the pooling resulted in a few percent of performance improvement with no noticeable loss in prediction quality. If anything, it even improved slightly, although well within noise (on the order of 0.002 nats or less throughout training).

More importantly, it seems that global pooling is valuable as an addition to the main residual trunk, rather than just the policy head! I tested this out by replacing a few ordinary and dilated-convolution residual blocks with residual blocks structured similarly to the policy head - computing an extra convolution in parallel whose channels were then globally pooled, transformed, and then summed back with the first convolution's output, before being fed onward to the second convolution of the residual block as usual. For a 12-block net, augmenting two of the residual blocks this way resulted in a major 0.02 nats of improvement persisting at about 60 million training steps, while slightly decreasing the number of parameters and having no cost to performance.

One story for why global pooling might be effective in the main trunk is that there the computed features (such as global quantity of ko threats or game phase) can have a much more integrated effect on the neural net's predictions by adjusting its procesing of different shapes and tactics (for example, favoring or disfavoring moves that could lead to future ko fights). Whereas if the global pooling is deferred only until the policy head, the only thing the neural net can do with the information is the relatively cruder operation of precomputing a few fixed sets of moves it would want to play and simply upweighting or downweighting those sets of moves based on the global features.

As an aside, all the stats in the "current results" tables earlier were produced without this improvement, since I didn't think to try including global pooling in the trunk before all the other final experiments and test games were underway. While the apparent gain would likely continue to diminish with further training time, I expect some of the results would be yet slightly better if this were incorporated and everything re-run. (edit: as of Apr 2018, did add a row now in the stats table showing this).

#### Update (Apr 2018):

Just to get a proper comparison, I re-ran a proper training run with global pooling in the trunk rather than only the policy head on the GoGoD data set, along with some improvements to the learning rate schedule and updated the [current results table](https://github.com/lightvector/GoNN#current-results) above. The difference is pretty large! Global pooling is by far the most successful architectural idea I"ve tried so far.


## Wide Low-Rank Residual Blocks (Feb 2018)

In one experiment, I found that adding a single residual block that performs a 1x9 and then a 9x1 convolution (as well as in parallel a 9x1 and then a 1x9 convolution, sharing weights) appears to decrease the loss slightly more than adding an additional ordinary block that does two 3x3 convolutions.

The idea is that such a block makes it easier for the neural net to propagate certain kinds of information faster across distance in the case where it doesn't need very detailed of nonlinear computations, much faster than 3x3 convolutions would allow. For example, one might imagine this being used to propagate information about large-scale influence from walls.

However, there's the drawback that either this causes an horizontal-vertical asymmetry if you don't do the convolutions in both orders of orientations, or else this block costs twice in performance as much as an ordinary residual block. I suspect it's possible to get a version that is more purely benefiting, but I haven't tried a lot of permutations on this idea yet, so this is still an area to be explored.

## Parametric ReLUs (Feb 2018)

I found a moderate improvement when switching to using parametric ReLUs (https://arxiv.org/pdf/1502.01852.pdf) instead of ordinary ReLUs. I also I found a very weird result about what parameters it wants to choose for them. I haven't heard of anyone else using parametric ReLUs for Go, so I'm curious if this result replicates in anyone else's neural nets.

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/parametricrelu.png" width="500" height="250"/></td></tr>
<tr><td><sub>Left: ordinary ReLU. Right: parametric ReLU, where "a" is a trainable parameter. Image from https://arxiv.org/pdf/1502.01852.pdf </sub></tr></td>
</table>

For a negligible increase in the number of parameters of the neural net, using parametric ReLUs improved the loss by about 0.025 nats over the first 5 million training steps, a fairly significant improvement given the simplicity of the change. This decayed to little to closer to 0.010 to 0.015 nats by 15 million training steps, but was still persistently and clearly better, well above the noise in the loss between successive runs.

As far as I can tell, this was not simply due to something like having a problem of dead ReLUs beforehand, or some other simple issue. Ever since batch normalization was added, much earlier, all stats about the gradients and values in the inner layers have indicated that very few of the ReLUs die during training. Rather, I think this change is some sort of genuine increase in the fitting ability of the neural net.

The idea that this is doing something useful for the net is supported by a strange observation: for the vast majority of the ReLUs, it seems like the neural net wants to choose a negative value for *a*! That means that the resulting activation function is non-monotone. In one of the most recent nets, depending on the layer, the mean value of *a* for all the ReLUs in a layer varies from around -0.15 to around -0.40, with standard deviation on the order of 0.10 to 0.15.

With one exception: the ReLUs involved in the global pooled properties layer persistently choose positive *a*, with a mean of positive 0.30. If any layer were to be different, I'd expect it to be this one, since these values are used in a very different way than any other layer, being globally pooled before being globally rebroadcast. Still, it's interesting that there's such a difference.

For the vast majority of the ReLUs though, as far as I can tell, the neural net actually does "want" the activation to be non-monotone. In a short test run where *a* was initialized to a positive value of 0.25 rather than 0, the neural net over the course of the first few million training steps forced all the *a*s to be negative mean again, except for the ones in the global pooled properties layer. In the meantime, it also had a larger training loss indicating that it was not fitting as well due to the positive *a*s.

I'd be very curious to hear whether this reproduces for anyone else. For now, I've been keeping the parametric ReLUs, since they do seem to be an improvement, although I'm quite mystified about why non-monotone functions are good here.

## Chain Pooling (Feb 2018)

Using the following functions:
   * https://www.tensorflow.org/api_docs/python/tf/unsorted_segment_max
   * https://www.tensorflow.org/api_docs/python/tf/gather

...it's possible to implement a layer that performs max-pooling across connected chains of stones (rather than pooling across 2x2 squares as you normally see in the image literature).

I tried adding a residual block that applied this layer, and got mixed results. On the one hand, when trying the resulting neural net in test cases, it improves all the situations you would expect it to improve.

For example, an ordinary neural net has no problem with making two eyes in this position when its lower eye is falsified:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/nochainpool1.png" width="350" height="350"/>

But, when you make it too far away, the neural net doesn't have enough convolutional layers to propagate the information, so it no longer suggests S19 to live:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/nochainpool2.png" width="350" height="350"/>

Adding a residual block with a chain pooling layer improves this, the neural net now suggests to live even when the move is arbitrarily far away, as long as the group is solidly connected:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/chainpool1.png" width="350" height="350"/>

And even does so partially when there's a gap in the chain, since the chain pooling is still able to propagate the information a good distance.

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/chainpool2.png" width="350" height="350"/>

I've also observed improvements in a variety of other cases, particularly in large capturing races or in endgame situations where a filled dame on a large chain requires a far-away defense. In all these cases, chain-pooling helped the neural net find the correct response.

However, when I added a single chain pooling residual block with a small number of channels (32 channels), the loss actually got worse by a significant 0.010 nats even all the way out to 64 million training steps. And in self-play, the overall playing strength was not obviously better, winning 292 games out of 600 against the old neural net. I don't know exactly what's going on that makes the neural net solve races and liberty situations better in test positions but not be noticeably better, perhaps either these positions are just rare and not so important, or perhaps the chain pooling is causing the rest of the neural net to not fit as well for some reason.

When I added two chain pooling blocks each with 64 channels, this time the loss did get better, but only by about 0.005 nats at 10 million training steps, which would presumably decay a little further as the neural net converged, which was still underwhelming.

One additional problem is that the chain pooling is fairly computationally expensive compared to plain convolutional layers. I haven't done detailed measurements, but it's probably at least several times more expensive, so even with an improvement it's not obvious that this is worth it over simply adding more ordinary residual blocks.

So all the results in test positions looked promising, but the actual stats on performance and improvement in the loss were a bit disappointing. I'll probably revisit variations on this idea later, particularly if I can find a cheaper way to compute something like this in Tensorflow.

## Dilated Convolutions (Mar 2018)

It turns out that there's a far better approach than chain pooling to help the neural net to propogate information across larger distances, which is the technique of dilated convolutions. As described in https://arxiv.org/pdf/1511.07122.pdf, dilated convolutions are like regular convolutions except that the patch of sampled values, in our case 3x3, is not a contiguous 3x3 square, but rather is composed of 9 values sampled at a wider spacing, according to a dilation factor > 1. The idea, obviously, is that the neural net can see farther with the same number of layers at the cost of some local discrimination ability.

I experimented with this by taking the then-latest neural net architecture and for each residual block after the first, altering 64 of the 192 channels of the first convolution in the block to be computed via dilated convolutions with dilation factors of 2 or 3 instead of regular convolutions (dilation factor = 1). The results were very cool; the neural net improved on all the same kinds of situations that chain pooling helped with, enabling the neural net to understand much longer-distance relationships and perceive the status of larger groups correctly:

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/nodilated1.png" width="300" height="300"/></td><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/dilated1.png" width="300" height="300"/></td></tr>
<tr><td colspan="2"><sub>Left: 5-block net fails to reply to throw-in by making eyes in the upper right.

Right: 5-block with dilated convolutions succeeds.</sub></td></tr>
</table>

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/nodilated2.png" width="300" height="300"/></td><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/dilated2.png" width="300" height="300"/></td></tr>
<tr><td colspan="2"><sub>Left: 5-block net fails to respond to eye-poke by making eye up top.

Right: 5-block with dilated convolutions succeeds.</sub></td></tr>
</table>

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/nodilated3.png" width="300" height="300"/></td><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/dilated3.png" width="300" height="300"/></td></tr>
<tr><td colspan="2"><sub>Left: 5-block net fails to reply to distant liberty fill in a capturing race.

Right: 5-block with dilated convolutions succeeds.</sub></td></tr>
</table>

Presumably, the neural net learned in its first few layers to compute features that indicate when there is connectivity with stones at distance 2 or 3, which then are used by the dilated convolutions to propagate information about liberties and eyes 2 or 3 spaces per convolution, rather than only 1 space at a time.

Like chain pooling, holding the number of parameters constant I did not find it to noticeably improve the cross-entropy loss. However when replacing 64/192 of the channels in this way, the loss also did not get worse either, unlike how it would definitely get worse if one were to simply delete those channels. Combined with the above pictures, it suggests the neural net is putting the dilations to good use to learn longer-distance relationships at the cost of being a little worse at some local shapes or tactics. And unlike chain pooling, dilated convolutions appear to be quite cheap, barely at all increasing the computational cost of the neural net.

So unlike chain pooling, this technique still seems quite promising and worth experimenting with further. For example, I haven't done any testing on whether this improves the overall quality of the neural net for use within MCTS. Many of the deep-learning generation of Go bots, before they reach AlphaGo-levels of dominance, appear to have their biggest weaknesses in the perception of large chains or large dragons and understanding their liberty or their life and death status. Recently also the literature appears to be filled with results like https://openreview.net/forum?id=HkpYwMZRb that suggest that the "effective depth" of neural nets does not scale as fast as their physical depth. This also might explain why it's slow for neural nets like those of AlphaGo Zero and Leela Zero to learn long-distance effects even when they have more than enough physical depth to eventually do so. These seem like precisely the kinds of problems that dilated convolutions could help with, and it's plausible that on the margin the tradeoff of making local shape understanding a bit worse in exchange would be good (as hopefully the search could correct for that).


## Redundant Parameters and Learning Rates (Feb 2018)

There is a class of kinds of twiddles you can do to the architecture of a neural net that are completely "redundant" and leave the neural net's representation ability exactly unchanged. In many contexts I've seen people completely dismiss these sorts of twiddles, but confusingly, I've noticed that they actually can have effects on the learning. And sometimes in published papers, I've noticed the authors use such architectures without any mention of the redundancy, leaving me to wonder how widespread the knowledge about these sorts of things is.

#### Redundant Convolution Weights

For example, in one paper on ResNets in Go, (http://www.lamsade.dauphine.fr/~cazenave/papers/resnet.pdf), for the initial convolution that begins the residual trunk, the author chose to use the sum of a 5x5 convolution and a 1x1 convolution. One might intuit this change to be potentially good, since the first layer will probably often not be computing overly sophisticated 5x5 patterns, and a significant amount of its job will be computing various important logical combinations of the input features to pass through (e.g. "stone has 1 liberty AND is next to the opponent's previous move") that will frequently involve the center weight, so the center weights will be important more often than the edge weights.

However, adding a 1x1 convolution to the existing 5x5 is completely redundant, because that's equivalent to doing only a 5x5 convolution with a different central weight.

    Adding the results of two separate convolutions with these kernels:
    a   b   c   d   e
    f   g   h   i   j
    k   l   m   n   o                z
    p   q   r   s   t
    u   v   w   x   y

    Is the same as doing a single convolution with this kernel:
    a   b   c   d   e
    f   g   h   i   j
    k   l (m+z) n   o
    p   q   r   s   t
    u   v   w   x   y

Nonetheless, it has an effect on the learning. With standard weight initialization, this causes the center weight in the equivalent single 5x5 kernel to be initialized much larger on average. For example I think Xavier initialization will initialize *z* to be sqrt(25) = 5 times larger on average than *m*. Moreover, during training, the learning rate of the center weight in the kernel is effectively doubled, because now *m* and *z* will both experience the gradient, and when they each take a step, their sum will take twice as large of a step.

So in my own neural nets I tried adding an additional parameter in the center of the existing 5x5 convolution, and in fact, it was an improvement! Only about 0.01 nats in early training, decaying to a slight improvement of less than 0.005 nats by 25 million training steps. But still an improvement to the overall speed of learning.

(The current neural nets in this sandbox use a 5x5 convolution rather than a 3x3 to begin the ResNet trunk since the small number of channels of the input layer makes it uniquely cheap to do a larger convolution here, unlike the rest of the neural net with 192 channels. It probably helps slightly in kicking things off, such as being able to detect the edge from 2 spaces away instead of only 1 space away by the start of the first residual block.)

I haven't gotten to it yet, but in the future I want to test more changes of this flavor. For example, given that it was a good idea to increase the weight initialization and learning rate on the center weight for the initial 5x5 convolution, could it be a good idea to do the same for all the 3x3 convolutions everywhere else in the neural net?

#### Update (Mar 2018):
Apparently, it is a good idea! Emphasizing the center weight initialization by 30% and increasing the learning rate by a factor of 1.5 on the center weight for the other convolutions in the neural net improved the rate of learning in one test giving about an 0.01 nat improvement at around 50 million training samples.

Also, to confirm the intuition behind this idea, I looked in a neural net trained without any special center emphasis at the average norm of the learned convolution weights by position within the convolution kernel. Here is a screenshot of the average norm of the weights for the first 4 residual blocks of a particular such net, displayed with coloring in a spreadsheet:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/convweights.png" width="450" height="350"/>

So on its own, the neural net chooses on average to put more weight in the center than on the edges or corners of the 3x3 convolutions. So it's not surprising that allowing the neural net to adjust the center weight relatively more rapidly than the other weights is an improvement for learning speed, even if, as I suspect, in ultimate convergence it might not matter.

#### Batch Norm Gammas

Another redundancy I've been experimenting with is the fact that due to historical accident, the current neural nets in this sandbox use scale=True in [batch norm layers](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization), even though the resulting gamma scale parameters are completely redundant with the convolution weights afterwards. This is because ReLUs (and parametric ReLUs) are scale-preserving, so a neural net having a batch norm layer with a given gamma is equivalent to one without gamma but where the next layer's convolution weights for that incoming channel are gamma-times-larger.

Yet, despite the redundancy, the presence of those redundant parameters does influence the learning. In some quick tests I was not able to get rid of them. Although I didn't try so extensively, in a handful of tries I was unable to find a combination of setting scale=False and increasing the learning rate to compensate that did not slow down the total efficiency of learning. So the latest neural nets for now continue to use scale=True.

##### Update (Apr 2018):

Since it's been a long time, I re-tested this again and was able to eliminate these. I'm not exactly sure what the difference was this time versus the previous attempt, or maybe the previous attempt found a difference only due to confounding noise between different initializations. It might be worth exploring this kind of detail further though. What, theoretically, should one expect the effect of having this particular kind of redundancy to have on the learning?

## Some Thoughts About History as an Input (Mar 2018)

Most or all of the open-source projects I've seen that aim to reproduce Alpha-Zero-like training seem to have the issue that the resulting neural net requires the history of the last several moves as input to the neural net. This somewhat limits the use of these bots for analysis, whether for whole-board tsumego or for asking of "what if" questions such as analyzing with/without an extra local stone, or for seeing in kyu-level games what the policy net would suggest without the presumption that the opponent's move was a good move (as would be in nearly all later games composing the training data).

I just wanted to record here that there is a very simple solution which I've used successfully so far to enable the neural net to also play well without history and that hardly costs anything. You simply take a small random percentage of positions in training, say 5%-10%, and don't provide the history information on those positions.
   * If you're using binary indicators of the locations of past moves and captures, this corresponds to zeroing out those planes.
   * If you're using the AlphaGoZero representation where you give whole past board states rather than giving binary indicators of recent moves, it's probably better to make the past board states to be unchanged copies of the current board state rather than zeroing them, since having them be identical is the closest way to say that no local moves happened there. (As opposed to zeroing them, which would be like telling the net that every stone was suddenly and recently placed).

Now, since you have training examples without history, the neural net adapts to produce reasonable predictions without history. But since nearly all the training examples still do have history, the neural net still mostly optimizes for having history and should lose little if any strength there.

Also, I wonder if there might be room for further experimentation along these lines. In general, normal methods of training the policy part of a net will cause it to learn that moves provided in its history represent strong play, because all examples in its training data have that property. However in many branches of an MCTS search, or when used as an analysis tool, this is definitely not the case. This may cause the policy to be too obedient in response to bad moves ("a strong opponent wouldn't have played here unless it was sente") and make the search less effective at refuting them (by tenukiing).

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/lastmove.png" width="300" height="300"/></td><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/nolastmove.png" width="300" height="300"/></td></tr>
<tr><td colspan="2"><sub>Left: With history, net wants to unnecessarily respond to black's slow gote move in the lower left.

Right: With move history planes zeroed out, net suggests moves in the more urgent areas at the top.
</table>


## Ranks as an Input (June 2018):

I added the rank of the player of the moves being predicted as an input to the neural net. A total of 64 different (rank * online server) combinations are provided to the neural net as a one-hot-encoded vector, which passes through an 8-dimensional embedding layer before being combined with the rest of the board features. As a result, the neural net can learn to predict how players of different strengths play differently!

As an example, in this position, it thinks that a typical 19 kyu (OGS) player would most likely connect against black's atari:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/19k.png" width="380" height="380"/>

But a 1 dan player would more likely prefer to sacrifice those stones and complete the lower left shape:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/1d.png" width="380" height="380"/>

And the net thinks a 9 dan player would be likely to do the same, but might also likely tenuki to the upper left, which is also reasonably urgent:

<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/9d.png" width="380" height="380"/>

Using this neural net and a bunch of heuristic filtering criteria, I found a way to generate surprisingly reasonable whole-board open-space fighting and shape problems, which I posted here as a training tool for Go players: [neuralnetgoproblems.com](https://neuralnetgoproblems.com).
