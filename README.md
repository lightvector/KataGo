## Go Neural Net Sandbox

This repo is currently a sandbox for personal experimentation in neural net training in Go. I haven't made any particular attempt to make the training pipeline usable by others, but if you're interested, the rough summary is:

   * You must have HDF5 installed for C++ (https://support.hdfgroup.org/HDF5/release/obtainsrc.html), as well as Tensorflow installed for Python 3.
   * Compile using "compile.sh" in writedata, which expects you to have h5c++ available. (yes, no makefiles or build system, very hacky).
   * Run the resulting "write.exe" on a directory of SGF files to generate an h5 file of preprocessed training data.
   * Run train.py using that h5 file to train the neural net.

See LICENSE for software license. License aside, informally, if do you successfully use any of the code or any wacky ideas about neural net structure explored in this repo in your own neural nets or to run any of your own experiments, I would to love hear about it and/or might also appreciate a casual acknowledgement where appropriate. Yay.

## Experimental Notes
You can see the implementations of the relevant neural net structures in "model.py", although I may adapt and change them as time goes on.

### Current results
Currently, the best neural nets I've been training from this sandbox have about 3.28 million parameters, arranged as a resnet with about 6 blocks with a main trunk of 192 channels, and a policy head, trained for about 64 million samples. On GoGoD games from 1995 to mid 2016, which is about 50000 games with about 10 million training samples, the validation cross-entropy loss achieved with this many parameters is about 1.66 nats, corresponding to a perplexity of exp(1.66) = 5.26. So the neural net has achieved about the same entropy on average as if on every move it could pin down the next move to uniformly within 5 or 6 possible moves. The top-1 prediction accuracy is about 51%.

I haven't tried running on the KGS dataset yet that most papers out there tend to test on. From the few papers that test on both GoGoD and KGS, it seems very roughly that KGS is a bit easier to predict, with accuracies about 4% higher than for the pro games than for KGS amateur games, so probably these results correspond to roughly a 55% top-1 accuracy on KGS, which I'll get around to testing eventually.

I also tried testing a recent neural net against GnuGo (not actually the latest and best as I did this test weeks ago, but fairly recent), choosing moves by sorting and then drawing the move based using a beta distribution to draw from [0,1] to draw the move, and out of 250 games, it lost only 3. So things are developing pretty well.

Also, for a comparison with other neural nets, see table below for summary of results.

| Neural Net | Structure | Params | KGS Top1 | GoGoD Top1 | Training Steps | Vs GnuGo |
|------|---|---|---|---|---|---|
| [Clark and Stokey (2015)](https://arxiv.org/abs/1412.3409)  | CNN 8 layers | ~560000 |  44.4% | 41.1% | 147M | 87%
| [Maddison et al. (2015)](https://arxiv.org/abs/1412.6564) |  CNN 12 layers | ~2300000 |  55.2% |  | 685M x 50 + 82M | 97%
| [AlphaGoFanHui-192 (2016)](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) | CNN 13 layers | 3880489 | 55.4% | | 340M x 50
| [AlphaGoFanHui-256 (2016)](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) | CNN 13 layers | 6795881 | 55.9%
| [Darkforest (2016)](https://arxiv.org/abs/1511.06410) | CNN 12 layers  | 12329756  | 57.1%  |   | 128M | 99.7%
| [Cazenave (2017)](http://www.lamsade.dauphine.fr/~cazenave/papers/resnet.pdf) | ResNet 10 blocks | 12098304 | 55.5% | 50.7% | 70M
| [Cazenave (2017)](http://www.lamsade.dauphine.fr/~cazenave/papers/resnet.pdf) | ResNet 10 blocks | 12098304 | 58.2% | 54.6% | 350M
| [AlphaGoZero-20Blocks(2017)](https://deepmind.com/documents/119/agz_unformatted_nature.pdf) | ResNet 20 blocks | 22837864 | 60.4% | | >1000M?
| Current Sandbox | ResNet 4 Blocks + 2 Special | 3285048 | | 51% | 64M | 98-99%

Based on this table and also observations during training, I think it's almost certainly the case that the prediction quality could be increased further simply by making the neural nets bigger and training much longer. At 64M steps the loss has clearly not quite stopped decreasing, but I usually stop the training by then anyways. Also the above table suggests that making the neural net bigger is always just better at these scales. Indeed I'm observing essentially no overfitting, so we're still well within the regime where continuing to make the model capacity larger will allow a better fit. But in the interests of actually getting to run a variety of experiments in a reasonable time on a limited budget (just two single-GPU machines on Amazon EC2), I've so far deliberately refrained from making the neural net much bigger or spending weeks optimizing any particular neural net.

Also, to get an idea of how top-1 accuracy and nats and direct playing strength correlate, at least on GoGoD and at least for these kinds of neural nets, here's a table showing the rough correspondence I observed as the neural nets in this sandbox gradually grew larger and improved. Elos were determined roughly by a mix of self-play and testing vs GnuGo.

| GoGoD Accuracy (Top 1) | GoGoD Cross Entropy Loss (nats) | Relative Elo |
|-----|---|---|
| 34% | 2.73 | 0
| 37% | 2.50 | ~350
| 40% | 2.31 | ~700
| 45% | 1.95 | ~950
| 47% | 1.85 | ~1050
| 49% | 1.77 | ~1150
| 51% | 1.67 | ~1300


### Special Ladder Residual Blocks
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
<caption align="bottom"><small>Net correctly flags working ladders. No white stone is a ladder breaker.</small></caption>
</table>
<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/laddertargetbroken.png" width="350" height="350"/></td></tr>
<caption align="bottom"><small>Net correctly determines that ladders don't work.</small></caption>
</table>

#### Update (201802):
Apparently, adding this block into the neural net does not cause it to be able to learn ladders in a supervised setting. From digging into the data a little, my suspicion is that in supervised settings, whether a ladder works or not is too strongly correlated with whether it gets formed in the first place, and similarly for escapes due to ladder breakers. So that unless it's plainly obvious whether the ladder works or not (the ladder-target experiments show this block makes it much easier, but it's still not trivial), the neural net fails to pick up on the signal. It's possible that in a reinforcement learning setting (e.g. Leela Zero), this would be different.

Strangely however, adding this block in *did* improve the loss, by about 0.015 nats at 10 epochs persisting to still a bit more than 0.010 nats at 25 epochs. I'm not sure exactly what the neural net is using this block for, but it's being used for something. Due to the bottlenecked nature of the block (I'm using only C = 6), it barely increases the number of parameters in the neural net, so this is a pretty surprising improvement in relative terms. So I kept this block in the net while moving on to later experiments, and I haven't gone back to testing further.

### Using Ladders as an Extra Training Target

In another effort to make the neural net understand ladders, I added a second head to the neural net and forced the neural net to simultaneously predict the next move and to identify all groups that were in or could be put in inescapable atari.

Mostly, this didn't work so well. With the size of neural net I was testing (~4-5 blocks, 192 channels) I was unable to get the neural net to produce predictions of long-distance ladders anywhere near as well as it did when it had the ladder target alone, unless I was willing to downweight the policy target in the loss function enough that it would no longer produce as-good predictions of the next move. With just a small weighting on the ladder target, the neural net learned to produce highly correct predictions for a variety of local inescapable ataris, such as edge-related captures, throw-in tactics and snapbacks, mostly everything except long-distance ladders. Presumably due to the additional regularization, this improved the loss for the policy very slightly, bouncing around 0.003-0.008 nats around 10 to 20 epochs.

But simply adding the ladder feature directly as an input to the neural net dominated this, improving the loss very slightly further. Also, with the feature directly provided as an input to the neural net, the neural net was finally able to tease out enough signal to mostly handle ladders well. 

<div class="image">
<img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/extraladdertargetfail2.png" width="300" height="300"/><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/ladderinput2.png" width="300" height="300"/>
<p><small>Extra ladder target training doesn't stop net from trying to run from working ladder (left), but directly providing it as an input feature works (right). </small></p>
</div>

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/ladderinput1.png" width="300" height="300"/></td><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/ladderinput3.png" width="300" height="300"/></td></tr>
<caption align="bottom"><small>And on the move before, whether the ladder works clearly affects the policy.</small></caption>
</table>

However, even with such a blunt signal, it still doesn't always handle ladders correctly! In some test positions, sometimes it fails to capture stones in a working ladder, or fails to run from a failed ladder, or continues to run from a working ladder, etc. Presumably pro players would not get into such situations in the first place, so there is a lack of data on these situations. 

This suggests that a lot of these difficulties are due to the supervised-learning setting, rather than merely difficulty of learning to solve ladders. I'm quite confident that in a reinforcement-learning setting more like the "Zero" training, with ladders actually provided directly as an input feature, the neural net would rapidly learn to not make such mistakes. It's also possible that in such settings, with a special ladder block the neural net would also not need ladders as an input feature and that the block would end up being used for ladder solving instead of whatever mysterious use it's currently being put to. 

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/ladderinput4.png" width="350" height="350"/></td></tr>
<caption align="bottom"><small>Even when told via input feature that this ladder works, the NN still wants to try to escape.</small></caption>
</table>


### Global Pooled Properties
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

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/kothreatglobalproperty.png" width="350" height="350"/></td></tr>
<caption align="bottom"><small>Heatmap of ko-fight-detector just prior to global pooling, activating bright green after a ko was taken. Elsewhere, the heatmap is covered with checkerlike patterns. </small></caption>
</table>

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/kothreatpolicy.png" width="350" height="350"/></td></tr>
<caption align="bottom"><small>The net uses this to suggests a distant plausible ko threat. </small></caption>
</table>

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/afterko.png" width="350" height="350"/></td></tr>
<caption align="bottom"><small>When there is no ko, the net no longer wants to play ko threats. </small></caption>
</table>

#### Update (201802):

The original tests of global pooled properties was done when the neural net was an order of magnitude smaller than it is now (~250K params instead of ~3M params), so I did a quick test of removing this part of the policy head to sanity check if this was still useful. Removing it immediately worsened the loss by about 0.04 nats on the first several epochs. Generally, differences on the first few epochs tend to diminish as the neural net converges further, but I would still guess at minimum 0.01 nats of harm at convergence, so this was a large enough drop that I didn't think it worth the GPU time to run it any longer.

So this is still adding plenty of value to the neural net. From a Go-playing perspective, it's also fairly obvious in practice that these channels are doing work. At least in ko fights, since the capture of an important ko plainly causes the neural net to suggest ko-threat moves all over the board that it would otherwise never suggest, including ones too far away to be easily reached by successive convolutions. I find it interesting that I haven't yet found any other published architectures include such a structure in the neural net.

Just for fun, here's some more pictures of channels just prior to the pooling. These are from latest nets that use parameteric ReLU, so the channels have negative values now as well (indicated by blues, purples, and magentas).
<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/territorydetector.png" width="350" height="350"/></td></tr>
<caption align="bottom"><small>Despite only being trained to predict moves, the net has developed a rough territory detector. It also clearly understands the upper right white group is dead. </small></caption>
</table>
<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/unfinishedborderdetector.png" width="350" height="350"/></td></tr>
<caption align="bottom"><small>This appears to be a detector for unfinished endgame borders. </small></caption>
</table>

### Wide Low-Rank Residual Blocks:

In one experiment, I found that adding a single residual block that performs a 1x9 and then a 9x1 convolution (as well as in parallel a 9x1 and then a 1x9 convolution, sharing weights) appears to decrease the loss slightly more than adding an additional ordinary block that does two 3x3 convolutions.

The idea is that such a block makes it easier for the neural net to propagate certain kinds of information faster across distance, in the case where such information doesn't need to involve as-detail of nonlinear computations, faster than 3x3 convolutions would allow. For example, one might imagine this being used to propagate information about large-scale influence from walls.

However, there's the drawback that either this causes an horizontal-vertical asymmetry, if you don't do the convolutions in both orders of orientations, or else this block costs twice in performance as much as an ordinary residual block. I suspect it's possible to get a version that is more purely benefiting, but I haven't tried a lot of permutations on this idea yet, so this is still an area to be explored.

### Parametric ReLUs

I found a moderate improvement when switching to using parametric ReLUs (https://arxiv.org/pdf/1502.01852.pdf) instead of ordinary ReLUs. I also I found a very weird result about what alpha parameters it wants to choose for them. I haven't heard of anyone else using parametric ReLUs for Go, I'm curious if this result replicates in anyone else's neural nets.

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/GoNN/master/images/readme/parametricrelu.png" width="350" height="350"/></td></tr>
<caption align="bottom"><small>Left: ordinary ReLU. Right: parametric ReLU, where "a" is a trainable parameter. Image from https://arxiv.org/pdf/1502.01852.pdf </small></caption>
</table>

For a negligible increase in the number of parameters of the neural net, using parametric ReLUs improved the loss by about 0.025 nats over the first 10 epochs a fairly significant improvement given the simplicity of the change. This decayed to little to closer to 0.010 to 0.015 nats by 30 epochs, but was still persistently and clearly better, well above the noise in the loss between successive runs.

As far as I can tell, this was not simply due to something like having a problem of dead ReLUs beforehand. Ever since batch normalization was added, much earlier, all stats about the gradients and values in the inner layers have indicated that very few of the ReLUs die during training. I think this change is some sort of genuine increase in the fitting ability of the neural net. 

The idea that this is doing something useful for the net is supported by a strange observation: for the vast majority of the ReLUs, it seems like the neural net wants to choose a negative value for *a*! That means that the resulting activation function is non-monotone. In one of the most recent nets, depending on the layer, the mean value of *a* for all the ReLUs in a layer varies from around -0.15 to around -0.40, with standard deviation on the order of 0.10 to 0.15.

With one exception: the ReLUs involved in the global pooled properties layer persistently choose positive *a*, with a mean of positive 0.30. If any layer were to be different, I'd expect it to be this one, since these values are used in a very different way than any other layer, being globally pooled before being globally rebroadcast. Still, it's interesting that there's such a difference.

For the vast majority of the ReLUs though, as far as I can tell, the neural net actually does "want" the activation to be non-monotone. In a short test run where *a* was initialized to a positive value rather than 0, the neural net over the course of the first few epochs forced all the *a*s to be negative mean again, except for the ones in the global pooled properties layer, and in the meantime, had a larger training loss, indicating that it was not fitting as well due to the positive *a*s.

I'd be very curious to hear whether this reproduces for anyone else. For now, I've been keeping the parametric ReLUs, since they do seem to be an improvement, although I'm quite mystified about why non-monotone functions are good here.

### Chain Pooling

Using the following functions:
https://www.tensorflow.org/api_docs/python/tf/unsorted_segment_max
https://www.tensorflow.org/api_docs/python/tf/gather
it's possible to implement a layer that performs max-pooling across connected chains of stones (rather than pooling across 2x2 squares as you normally see in the image literature).

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

However, when I added a single chain pooling residual block with a small number of channels (32 channels), the loss actually got worse by a significant 0.010 nats even all the way out to 128 epochs. And in self-play, the overall playing strength was not obviously better, winning 292 games out of 600 against the old neural net. I don't know exactly what's going on that makes the neural net solve races and liberty situations better in test positions but not be noticeably better, perhaps either these positions are just rare and not so important, or perhaps the chain pooling is causing the rest of the neural net to not fit as well for some reason.

When I added two chain pooling blocks each with 64 channels, this time the loss did get better, but only by about 0.005 nats at 20 epochs, which would presumably decay a little further as the neural net converged, which was still underwhelming.

One additional problem is that the chain pooling is fairly computationally expensive compared to plain convolutional layers. I haven't done detailed measurements, but it's probably at least several times more expensive, so even with an improvement it's not obvious that this is worth it over simply adding more ordinary residual blocks.

So all the results in test positions looked promising, but the the actual stats were a bit disappointing. I'll probably revisit variations on this idea later, particularly if I can find a cheap way of a similar result in Tensorflow with less computational cost.

### Redundant Parameters and Learning Rates

There is a class of kinds of twiddles you can do to the architecture of a neural net that are completely "redundant" and leave the neural net's representation ability exactly unchanged. In many contexts I've seen people completely dismiss these sorts of twiddles, but confusingly, I've noticed that they actually can have effects on the learning. And sometimes in published papers, I've noticed the authors use such architectures without any mention of the redundancy, leaving me to wonder how widespread the knowledge about these sorts of things is.

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

So in my own neural nets I tried adding an additional parameter in the center of the existing 5x5 convolution, and in fact, it was an improvement! Only about 0.01 nats in early epochs, decaying to a slight improvement of less than 0.005 nats by epoch 50 or later. But still an improvement to the overall speed of learning.

(The current neural nets in this sandbox use a 5x5 convolution rather than a 3x3 to begin the ResNet trunk since the small number of channels of the input layer it uniquely cheap to do a larger convolution here, unlike the rest of the neural net with 192 channels. It probably helps slightly in kicking things off, such as being able to detect the edge from 2 spaces away instead of only 1 space away by the start of the first residual block).

Another redundancy I've been experimenting with is the fact that due to historical accident, the current neural nets in this sandbox use scale=True in [batch norm layers](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization), even though the resulting gamma scale parameters are completely redundant with the convolution weights afterwards. This is because ReLUs (and parametric ReLUs) are scale-preserving, so a neural net having a batch norm layer with a given gamma is equivalent to one without gamma but where the next layer's convolution weights for that incoming channel are gamma-times-larger. 

Yet, despite the redundancy, the presence of those redundant parameters does influence the learning. In some quick tests I was not able to get rid of them. Although I didn't try so extensively, in a handful of tries I was unable to find a combination of setting scale=False and increasing the learning rate to compensate that did not slow down the total efficiency of learning. So the latest neural nets for now continue to use scale=True.

I haven't gotten to it yet, but in the future I want to test more changes of this flavor. For example, given that it was a good idea to increase the weight initialization and learning rate on the center weight for the initial 5x5 convolution, could it be a good idea to do the same for all the 3x3 convolutions everywhere else in the neural net?


