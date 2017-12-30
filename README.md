## Go Neural Net Sandbox

This repo is currently a sandbox for personal experimentation in neural net training in Go. I haven't made any particular attempt to make the training pipeline usable by others, but if you're interested, the rough summary is:

   * You must have HDF5 installed for C++ (https://support.hdfgroup.org/HDF5/release/obtainsrc.html)
   * Compile using "compile.sh" in writedata, which expects you to have h5c++ available. (yes, no makefiles or build system, very hacky).
   * Run the resulting "write.exe" on a directory of SGF files to generate an h5 file of preprocessed training data.
   * Run train.py using that h5 file to train the neural net.

See LICENSE for software license. License aside, informally, if do you successfully use any of the code or any wacky ideas about neural net structure explored in this repo in your own neural nets or to run any of your own experiments, I would to love hear about it and/or might also appreciate a casual acknowledgement where appropriate. Yay.

## Experimental Notes
You can see the implementations neural net structures of these in "model.py", although I may adapt and change them as time goes on.

### Ladders and Special Ladder Residual Blocks
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
   * Also apply your favorite activation function and batch norm at appropriate ponts throughout the above.
   * Add the results as residuals back to the main resnet trunk.

In the event that many of the weights are near zero, this will have the effect of propagating information potentially very long distances across the diagonals. In practice, I applied an exp-based transform to the weight channel to make it behave like an exponentially-weighted moving average, to obtain the effect that ladders care mostly about the first stone or stones they hit, and not the stones beyond them, as well as a bias to try to make it easier for the neural net to put low weight on empty spaces to encourage long-distance propagation.

Adding such a residual block to the neural net appears to greatly help long-distance ladder solving! When I trained a neural net with this to identify laddered groups, it appeared to have decently accurate ladder solving in test positions well beyond the theoretical range that its convolutions could reach alone, and I'm currently investigating whether adding this special block into a policy net helps the policy net's predictions about ladder-related tactics.

### Global Information Pooling
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

So apparently global information pooling helps a lot. I bet this could also be done as a special residual block earlier in the neural net rather than putting it only in the policy head.

