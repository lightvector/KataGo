## KataGo GTP Extensions

In addition to a basic set of [GTP commands](https://www.lysator.liu.se/~gunnar/gtp/), KataGo supports a few additional commands:

   * `rectangular_boardsize X Y`
      * Sets the board size to a potentially non-square size, width `X` and height `Y`. KataGo's official neural nets are currently not actually trained with non-square sizes, but they actually seem to generalize to them pretty well.
   * `clear-cache`
      * Clears the search tree and the NN cache. Can be used to force KataGo to re-search a position freshly, re-randomizing the search on that position, or to free up memory.
   * `stop`
      * Halts any ongoing pondering, if pondering was enabled in the gtp config.
   * `lz-analyze KEYVALUEPAIR KEYVALUEPAIR ...`
      * Begin searching and optionally outputting live analysis to stdout.
      * Possible key-value pairs:
         * `interval CENTISECONDS` - Output a line every this many centiseconds.
         * `minmoves N` - Output stats for at least N different legal moves if possible (will likely cause KataGo to output stats on 0-visit moves)
         * `allow ...` - Not currently implemented in KataGo.
         * `avoid ...` - Not currently implemented in KataGo.
      * Output format:
         * Outputted lines look like `info move C4 visits 6781 winrate 4873 prior 1090 lcb 4864 order 0 pv C4 R16 Q3 D17 E4 info move Q17 visits 5961 winrate 4872 prior 968 lcb 4862 order 1 pv Q17 Q3 C16 C4 info move R16 visits 5960 winrate 4871 prior 959 lcb 4862 order 2 pv R16 C4 C16 Q3 E16 E4 info move Q3 visits 5938 winrate 4871 prior 959 lcb 4862 order 3 pv Q3 C4 Q17 C16`
         * `info` - Indicates the start of information for a new possible move
         * `move` - The move being analyzed.
         * `visits` - The number of visits invested into the move so far.
         * `winrate` - 10000 times the winrate of the move so far, rounded to an integer in [0,10000].
         * `prior` - 10000 times the policy prior of the move, rounded to an integer in [0,10000].
         * `lcb` - 10000 times the [LCB](https://github.com/leela-zero/leela-zero/issues/2282) of the move, rounded to an integer in [0,10000].
         * `order` - KataGo's ranking of the move. 0 is the best, 1 is the next best, and so on.
         * `pv` - The principal variation following this move. May be of variable length or even empty.
      * All output values are from the perspective of the current player, unless otherwise configured in KataGo's gtp config.
      * This command will terminate upon any new GTP command being received, as well as upon a raw newline being received, including outputting the usual double-newline that signals a completed GTP response.

   * `kata-analyze KEYVALUEPAIR KEYVALUEPAIR ...`
      * Same as `lz-analyze` except a slightly different output format and some additional options and fields.
      * Additional possible key-value pairs:
         * `ownership true` - Output the predicted final ownership of every point on the board.
      * Output format:
         * Outputted lines look like `info move Q4 visits 246 utility -0.0249489 radius 0.0134198 winrate 0.491129 scoreMean -0.114924 scoreStdev 31.2765 prior 0.0272995 lcb 0.486337 utilityLcb -0.0383687 order 0 pv Q4 C4 D17 R16 D15 E4 info move R4 visits 711 utility -0.0362005 radius 0.00784969 winrate 0.487353 scoreMean -0.758136 scoreStdev 31.1881 prior 0.109013 lcb 0.48455 utilityLcb -0.0440501 order 1 pv R4 Q17 D3 C16 D5 info move R16 visits 702 utility -0.0345537 radius 0.00793677 winrate 0.487982 scoreMean -0.690915 scoreStdev 31.189 prior 0.0923564 lcb 0.485148 utilityLcb -0.0424905 order 2 pv R16 C16 R4 D3 P16 D5 E17 info move D17 visits 686 utility -0.035279 radius 0.00776143 winrate 0.487766 scoreMean -0.741424 scoreStdev 31.179 prior 0.0967651 lcb 0.484994 utilityLcb -0.0430404 order 3 pv D17 C4 Q17`
         * `info` - Indicates the start of information for a new possible move
         * `move` - The move being analyzed.
         * `visits` - The number of visits invested into the move so far.
         * `winrate` - The winrate of the move so far, as a float in [0,1].
         * `scoreMean` - The predicted average value of the final score of the game after this move, in points.
         * `scoreStdev` - The predicted standard deviation of the final score of the game after this move, in points. (NOTE: due to the mechanics of MCTS, this value will be significantly biased high currently, although it can still be informative as *relative* indicator).
         * `prior` - The policy prior of the move, as a float in [0,1].
         * `utility` - The utility of the move, combining both winrate and score, as a float in [-C,C] where C is the maximum possible utility.
         * `lcb` - The [LCB](https://github.com/leela-zero/leela-zero/issues/2282) of the move's winrate, as a float in [0,1].
         * `utilityLcb` - The LCB of the move's utility.
         * `radius` - Redundant with other values.
         * `order` - KataGo's ranking of the move. 0 is the best, 1 is the next best, and so on.
         * `pv` - The principal variation following this move. May be of variable length or even empty.
         * `ownership` - If `ownership true` was provided, then BoardHeight*BoardWidth many conecutive floats in [-1,1] separated by spaces, predicting the final ownership of every board location from the perspective of the current player. Floats are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
