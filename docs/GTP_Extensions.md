## KataGo GTP Extensions

In addition to a basic set of [GTP commands](https://www.lysator.liu.se/~gunnar/gtp/), KataGo supports a few additional commands:

   * `rectangular_boardsize X Y`
      * Sets the board size to a potentially non-square size, width `X` and height `Y`. KataGo's official neural nets are currently not actually trained with non-square sizes, but they actually seem to generalize to them pretty well.
   * `set_position COLOR VERTEX COLOR VERTEX COLOR VERTEX ...`
      * Directly specify an initial board position as a sequence of color-vertex pairs, replacing the current board.
      * The newly-set position is assumed to have no move history. Therefore:
         * There are no ko or superko prohibitions yet in the specified position.
         * If using any territory scoring rules, it is assumed that there are no captures so far.
      * Part of the reason for this command is because modern neural-net-based bots, KataGo included, may mildly bias their move suggestions based on the recent moves. This command provides a way to set initial board positions (e.g. whole-board tsumego) without using successive `play` commands which might imply a ridiculous move history.
      * It is NOT recommended to use this command to place the starting stones for handicap games. Use the standard GTP commands `fixed_handicap`, `place_free_handicap`, and/or `set_free_handicap` instead.
      * Calling `set_position` with zero arguments is equivalent to calling `clear_board`.
      * Fails, reports a normal GTP error message, and leaves the board state unchanged if any vertex is specified more than once or if the final configuration would contain stones with zero liberties.
   * `clear_cache`
      * Clears the search tree and the NN cache. Can be used to force KataGo to re-search a position freshly, re-randomizing the search on that position, or to free up memory.
   * `stop`
      * Halts any ongoing pondering, if pondering was enabled in the gtp config.
   * `get_komi`
      * Report what komi KataGo is currently set to use.
   * `kata-get-rules`
      * Returns a JSON dictionary indicating the current rules KataGo is using.
      * For example: `{"hasButton":false,"ko":"POSITIONAL","scoring":"AREA","suicide":true,"tax":"NONE","whiteHandicapBonus":"N-1","friendlyPassOk":true}`
      * See https://lightvector.github.io/KataGo/rules.html for a detailed description of the rules implemented.
      * Explanation of individual fields:
         * `ko: ("SIMPLE" | "POSITIONAL" | "SITUATIONAL")` - The rule used for preventing cycles.
         * `scoring: ("AREA" | "TERRITORY")` - The rule used for computing the score of the game.
         * `tax: ("NONE" | "SEKI" | "ALL")` - Modification to the scoring rule, indicating whether territory in SEKI is taxed, or whether ALL groups pay a tax of up to 2 points for eyes.
         * `suicide: (true | false)` - Whether multi-stone suicide is legal.
         * `hasButton: (true | false)` - Whether [button Go](https://senseis.xmp.net/?ButtonGo) is being used.
         * `whiteHandicapBonus ("0" | "N-1" | "N")` - In handicap games, whether white gets 0, N-1, or N bonus points, where N is the number of black handicap stones.
         * `friendlyPassOk: (true | false)` - When false, indicates that in this ruleset a player should capture all dead stones before passing.
      * **NOTE: It is possible that more fields and more options for these fields will be added in the future.**
   * `kata-set-rules RULES`
      * Sets the current rules KataGo should be using. Does NOT otherwise affect the board position.
      * `RULES` should either be a JSON dictionary in the same format of `kata-get-rules`, or be a shorthand string like `tromp-taylor`. Some possible shorthand strings are:
         * `tromp-taylor  : Equivalent to {"hasButton":false,"ko":"POSITIONAL", "scoring":"AREA",     "suicide":true, "tax":"NONE","whiteHandicapBonus":"0","friendlyPassOk":false}`
         * `chinese       : Equivalent to {"hasButton":false,"ko":"SIMPLE",     "scoring":"AREA",     "suicide":false,"tax":"NONE","whiteHandicapBonus":"N","friendlyPassOk":true}`
         * `chinese-ogs   : Equivalent to {"hasButton":false,"ko":"POSITIONAL", "scoring":"AREA",     "suicide":false,"tax":"NONE","whiteHandicapBonus":"N","friendlyPassOk":true}`
         * `chinese-kgs   : Equivalent to {"hasButton":false,"ko":"POSITIONAL", "scoring":"AREA",     "suicide":false,"tax":"NONE","whiteHandicapBonus":"N","friendlyPassOk":true}`
         * `japanese      : Equivalent to {"hasButton":false,"ko":"SIMPLE",     "scoring":"TERRITORY","suicide":false,"tax":"SEKI","whiteHandicapBonus":"0","friendlyPassOk":true}`
         * `korean        : Equivalent to {"hasButton":false,"ko":"SIMPLE",     "scoring":"TERRITORY","suicide":false,"tax":"SEKI","whiteHandicapBonus":"0","friendlyPassOk":true}`
         * `stone-scoring : Equivalent to {"hasButton":false,"ko":"SIMPLE",     "scoring":"AREA",     "suicide":false,"tax":"ALL", "whiteHandicapBonus":"0","friendlyPassOk":true}`
         * `aga           : Equivalent to {"hasButton":false,"ko":"SITUATIONAL","scoring":"AREA",     "suicide":false,"tax":"NONE","whiteHandicapBonus":"N-1","friendlyPassOk":true}`
         * `bga           : Equivalent to {"hasButton":false,"ko":"SITUATIONAL","scoring":"AREA",     "suicide":false,"tax":"NONE","whiteHandicapBonus":"N-1","friendlyPassOk":true}`
         * `new-zealand   : Equivalent to {"hasButton":false,"ko":"SITUATIONAL","scoring":"AREA",     "suicide":true, "tax":"NONE","whiteHandicapBonus":"0","friendlyPassOk":true}`
         * `aga-button    : Equivalent to {"hasButton":true, "ko":"SITUATIONAL","scoring":"AREA",     "suicide":false,"tax":"NONE","whiteHandicapBonus":"N-1","friendlyPassOk":true}`
      * KataGo does NOT claim that the above rules are _exactly_ a match. These are merely the _closest_ settings that KataGo has to those countries' rulesets.
      * A small number of combinations are currently not supported by even the latest neural nets, for example `scoring TERRITORY` and `hasButton true`.
      * Older neural nets for KataGo (nets released before v1.3) will also not support many of the options, and setting these rules will fail if these neural nets are being used.
      * Note the distinction between the `chinese` and `chinese-ogs, chinese-kgs`. Often in Chinese tournaments, contrary to their nominal written rules, which specify positional superko, only simple ko is used, and triple ko typically does result in the referree voiding the game. However, many servers have implemented the nominal written rules rather than the actually-used rules.
   * `kata-set-rule RULE VALUE`
      * Sets a single field of the current rules, leaving other fields unaffected.
      * For example, `kata-set-rule ko SIMPLE`.
      * May fail, if setting this field would result in a combination of rules that is not supported by the current neural net.
   * `kgs-rules RULES`
      * This is an extension for playing on KGS, via kgsGtp.
      * As specified by kgsGtp docs, `RULES` should be one of `chinese | aga | japanese | new_zealand`.
      * Has the same behavior as `kata-set-rules` except that `chinese` maps to `chinese-kgs` above.
   * `kgs-time_settings KIND ...`
      * This is an extension for playing on KGS, via kgsGtp.
      * It is not clear if other implementations will support floating-point values for times, however, KataGo does support them. Values must still be non-negative and finite, and values up to at least 2^31-1 are supported.
      * As specified by kgsGtp docs, `KIND` should be one of `none | absolute | canadian | byoyomi`.
         * `none` indicates no time control
         * `absolute` should be followed by a single float `MAINTIME` specifying the time in seconds.
         * `canadian` should be followed by `MAINTIME BYOYOMITIME BYOYOMISTONES` (float,float,int) specifying the main time, the length of the canadian overtime period, and the number of stones that must be played in that period.
         * `byoyomi` should be followed by `MAINTIME BYOYOMITIME BYOYOMIPERIODS` (float,float,int) specifying the main time, the length of each byo-yomi period, and the number of byo-yomi periods.
      * NOTE: As a hack, KGS also expects that when using `byoyomi`, the controller should provide the number of periods left in place of the number of stones left for the GTP `time_left` command.
   * `kata-time_settings KIND ...`
      * This is an extension for exposing KataGo's handling of more time controls besides those on KGS.
      * KataGo and any bots that also choose to implement this same extension must support floating-point values for times, as well as for the normal `time_left` command. Values must still be non-negative and finite, and values up to at least 2^31-1 must be supported.
      * Supports all of the same settings as `kgs-time_settings` except includes additional time settings:
        * `fischer` should be followed by two floats `MAINTIME INCREMENT` specifying the original main time in seconds and the number of seconds that is added to a player's clock at the end of making each move. And just as with `absolute`, when using this time setting the controller must always report `0` for the number of stones in the `time_left` command.
        * `fischer-capped` should be followed by four floats `MAINTIME INCREMENT MAINTIMELIMIT MAXTIMEPERMOVE`. Same as `fischer` except:
          * `MAINTIMELIMIT` is a cap such that after adding the increment, if the main time is now larger than the limit, it is reduced to the limit. If this limit is used it must be `>= MAINTIME`.
          * `MAXTIMEPERMOVE` is a limit on the maximum time that may be spent on any individual move.
          * Setting either of these two floats to a negative value such as `-1` indicates that the value is unused and there is no such limit.
      * Additionally, unlike `time_settings` or `kgs-time_settings` which demand integer values, for `kata-time_settings`, floating point values are explicitly allowed for all times, periods, or increments. Values must be finite and non-negative. An implementation should continue to support values at least up to 2^31-1.
      * More time settings might be added in the future.
   * `kata-list_time_settings`
      * Reports all time settings supported by `kata-time_settings`, separated by whitespace. Currently the list is `none absolute byo-yomi canadian fischer`.
      * GTP controllers that use `kata-time_settings` are advised to use this command to check if their time setting is supported.
   * `lz-analyze [player (optional)] [interval (optional)] KEYVALUEPAIR KEYVALUEPAIR ...`
      * Begin searching and optionally outputting live analysis to stdout. Assumes the normal player to move next unless otherwise specified.
      * Possible key-value pairs:
         * `interval CENTISECONDS` - Output a line every this many centiseconds. Alternate way to specify interval besides as the second argument.
         * `minmoves N` - Output stats for at least N different legal moves if possible (will likely cause KataGo to output stats on 0-visit moves)
         * `maxmoves N` - Output stats for at most N different legal moves (NOTE: Leela Zero does NOT currently support this field)
         * `avoid PLAYER VERTEX,VERTEX,... UNTILDEPTH` - Prohibit the search from exploring the specified moves for the specified player, until `UNTILDEPTH` ply deep in the search.
            * May be supplied multiple times with different `UNTILDEPTH` for different sets of moves. The behavior is unspecified if a move is specified more than once with different `UNTILDEPTH`.
         * `allow PLAYER VERTEX,VERTEX,... UNTILDEPTH` - Equivalent to `avoid` on all vertices EXCEPT for the specified vertices. Can only be specified once, and cannot be specified at the same time as `avoid`.
      * Output format:
         * Outputted lines look like `info move E4 visits 1178 winrate 4802 prior 2211 lcb 4781 order 0 pv E4 E3 F3 D3 F4 P4 P3 O3 Q3 O4 K3 Q6 S6 E16 E17 info move P16 visits 1056 winrate 4796 prior 2206 lcb 4769 order 1 pv P16 P17 O17 Q17 O16 E16 E17 F17 D17 F16 K17 D14 B14 P3 info move E3 visits 264 winrate 4752 prior 944 lcb 4722 order 2 pv E3 D5 P16 P17 O17 Q17 O16 E17 H17 D15 C15 D14 info move E16 visits 262 winrate 4741 prior 1047 lcb 4709 order 3 pv E16 P4 P3 O3 Q3 O4 P16 P17 O17 Q17 O16 Q14`
         * `info` - Indicates the start of information for a new possible move
         * `move` - The move being analyzed.
         * `visits` - The number of visits invested into the move so far.
         * `winrate` - 10000 times the winrate of the move so far, rounded to an integer in [0,10000].
         * `prior` - 10000 times the policy prior of the move, rounded to an integer in [0,10000].
         * `lcb` - 10000 times the [LCB](https://github.com/leela-zero/leela-zero/issues/2282) of the move, rounded to an integer in [0,10000], or possibly a value outside that interval, since the current implementation doesn't strictly account for the 0-1 bounds.
         * `order` - KataGo's ranking of the move. 0 is the best, 1 is the next best, and so on.
         * `pv` - The principal variation following this move. May be of variable length or even empty.
      * All output values are from the perspective of the current player, unless otherwise configured in KataGo's gtp config.
      * This command is a bit unusual for GTP in that it will run forever on its own, but asynchronously if any new GTP command or a raw newline is received, then it will terminate.
      * Upon termination, it will still output the usual double-newline that signals a completed GTP response.

   * `kata-analyze [player (optional)] [interval (optional)] KEYVALUEPAIR KEYVALUEPAIR ...`
      * Same as `lz-analyze` except a slightly different output format and some additional options and fields.
      * Additional possible key-value pairs:
         * `ownership true` - Output the predicted final ownership of every point on the board.
         * `ownershipStdev true` - Output the standard deviation of the distribution of predicted final ownerships of every point on the board across the search tree.
         * `movesOwnership true` - Output the predicted final ownership of every point on the board for every individual move.
         * `movesOwnershipStdev true` - Output the standard deviation of the distribution of predicted final ownerships of every point on the board across the search tree for every individual move.
         * `pvVisits true` - Output the number of visits spend on each move each principal variation.
      * Output format:
         * Outputted lines look like `info move E4 visits 487 utility -0.0408357 winrate 0.480018 scoreMean -0.611848 scoreStdev 24.7058 scoreLead -0.611848 scoreSelfplay -0.515178 prior 0.221121 lcb 0.477221 utilityLcb -0.0486664 order 0 pv E4 E3 F3 D3 F4 P4 P3 O3 Q3 O4 K3 Q6 S6 E16 E17 info move P16 visits 470 utility -0.0414945 winrate 0.479712 scoreMean -0.63075 scoreStdev 24.7179 scoreLead -0.63075 scoreSelfplay -0.5221 prior 0.220566 lcb 0.47657 utilityLcb -0.0502929 order 1 pv P16 P17 O17 Q17 O16 E17 H17 D15 C15 D14 C13 D13 C12 D12 info move E16 visits 143 utility -0.0534071 winrate 0.474509 scoreMean -0.729858 scoreStdev 24.7991 scoreLead -0.729858 scoreSelfplay -0.735747 prior 0.104652 lcb 0.470674 utilityLcb -0.0641425 order 2 pv E16 P4 P3 O3 Q3 O4 E3 H3 D5 C5`
         * `info` - Indicates the start of information for a new possible move, followed by key-value pairs. Current key-value pairs:
            * **NOTE: Consumers of this data should attempt to be robust to the order of these fields, as well as to possible addition of new fields in the future.**
            * `move` - The move being analyzed.
            * `visits` - The number of visits invested into the move so far.
            * `winrate` - The winrate of the move so far, as a float in [0,1].
            * `scoreMean` - Same as scoreLead. "Mean" is a slight misnomer, but this field exists to preserve compatibility with existing tools.
            * `scoreStdev` - The predicted standard deviation of the final score of the game after this move, in points. (NOTE: due to the mechanics of MCTS, this value will be **significantly biased high** currently, although it can still be informative as a *relative* indicator).
            * `scoreLead` - The predicted average number of points that the current side is leading by (with this many points fewer, it would be an even game).
            * `scoreSelfplay` - The predicted average value of the final score of the game after this move from low-playout noisy selfplay, in points. (NOTE: users should usually prefer scoreLead, since scoreSelfplay may be biased by the fact that KataGo isn't perfectly score-maximizing).
            * `prior` - The policy prior of the move, as a float in [0,1].
            * `utility` - The utility of the move, combining both winrate and score, as a float in [-C,C] where C is the maximum possible utility.
            * `lcb` - The [LCB](https://github.com/leela-zero/leela-zero/issues/2282) of the move's winrate, as a float in [0,1].
            * `utilityLcb` - The LCB of the move's utility.
            * `isSymmetryOf` - Another legal move. Possibly present if KataGo is configured to avoid searching some moves due to symmetry (`rootSymmetryPruning=true`). If present, this move was not actually searched, and all of its stats and PV are copied symmetrically from that other move.
            * `order` - KataGo's ranking of the move. 0 is the best, 1 is the next best, and so on.
            * `pv` - The principal variation ("PV") following this move. May be of variable length or even empty.
            * `pvVisits` - The number of visits used to explore the position resulting from each move in `pv`. Exists only if `includePVVisits` was requested.
            * `pvEdgeVisits` - The number of visits used to explore each move in `pv`. Exists only if `includePVVisits` was requested. Differs from pvVisits when doing graph search and multiple move sequences lead to the same position - pvVisits will count the total number of visits for the position at that point in the PV, pvEdgeVisits will count only the visits reaching the position using the move in the PV from the preceding position.
            * `movesOwnership` - this indicates the start of information about predicted board ownership. Exists only if `movesOwnership true` was requested.
            * `movesOwnershipStdev` - The standard deviation of the ownership across the search. Exists only if `movesOwnershipStdev true` was requested.
         * `ownership` - Alternatively to `info`, this indicates the start of information about predicted board ownership, which applies to every location on the board rather than only legal moves. Only present if `ownership true` was provided.
            * Following is BoardHeight*BoardWidth many consecutive floats in [-1,1] separated by spaces, predicting the final ownership of every board location from the perspective of the current player. Floats are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
         * `ownershipStdev` - The standard deviation of the ownership across the search. Only present if `ownershipStdev true` was provided.
            * Similar to `ownership`, following is BoardHeight*BoardWidth many consecutive floats in [0,1] separated by spaces. Floats are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
  * `lz-genmove_analyze [player (optional)] [interval (optional)] KEYVALUEPAIR KEYVALUEPAIR`
     * Same as `genmove` except will also produce `lz-analyze`-like output while the move is being searched.
     * Like `lz-analyze`, will immediately begin printing a partial GTP response, with a new line every `interval` centiseconds.
     * Unlike `lz-analyze`, will teriminate on its own after the normal amount of time that a `genmove` would take and will NOT terminate prematurely or asynchronously upon recipt of a newline or an additional GTP command.
     * The final move made will be reported as a single line `play <vertex or "pass" or "resign">`, followed by the usual double-newline that signals a complete GTP response.
  * `kata-genmove_analyze [player (optional)] [interval (optional)] KEYVALUEPAIR KEYVALUEPAIR`
     * Same as `lz-genmove_analyze` except with the options and fields of `kata-analyze` rather than `lz-analyze`
  * `analyze, genmove_analyze`
     * Same as `kata-analyze` and `kata-genmove_analyze`, but intended specifically for the Sabaki GUI app in that all floating point values are always formatted with a decimal point, even when a value happens to be an integer. May also have slightly less compact output in other ways (e.g. extra trailing zeros on some decimals).
     * The output format of `analyze` and `genmove_analyze` may update in future versions if Sabaki's format updates.

  * `kata-raw-nn SYMMETRY`
     * `SYMMETRY` should be an integer from 0-7 or "all".
     * Reports the result of a raw neural net evaluation from KataGo, or multiple raw evaluations in the case of "all".
     * Output format is of the form `symmetry <integer 0-7> <key> <value(s)> <key> <value(s)> ...`, possibly with additional whitespace or newlines between any tokens. In the case of "all", multiple such outputs of this form are concatenated together.
     * Possible keys are currently
     ```
     whiteWin (1 float)  - believed probability that white wins
     whiteLoss (1 float) - believed probability that black wins
     noResult (1 float)  - believed probability of noresult due to triple ko
     whiteLead (1 float) - predicted number of points that white is ahead by (this is the preferred score value for user display).
     whiteScoreSelfplay (1 float) - predicted mean score that would result from low-playout noisy selfplay (may be biased, Kata isn't fully score-maximizing).
     whiteScoreSelfplaySq (1 float) - predicted mean square of score that would result via low-playout noisy selfplay
     shorttermWinlossError (1 float) - predicted square root of the mean squared difference between (whiteWin-whiteLoss) and the MCTS (whiteWin-whiteLoss) in low-playout noisy selfplay after a few turns. Generally unavailable for nets prior to December 2020, in which case this value will always equal -1.
     shorttermScoreError (1 float) - predicted square root of the mean difference between whiteScoreSelfplay and the MCTS score in low-playout noisy selfplay after a few turns.  Generally unavailable for nets prior to December 2020, in which case this value will always equal -1.
     policy (boardXSize * boardYSize floats, including possibly NAN for illegal moves) - policy distribution for next move
     policyPass (1 floats) - policy probability for the pass move
     whiteOwnership (boardXSize * boardYSize floats) - predicted ownership by white (from -1 to 1).
     ```
     Any consumers of this data should attempt to be robust to any pattern of whitespace within the output, as well as possibly the future addition of new keys and values. The ordering of the keys is also not guaranteed - consumers should be capable of handling any permutation of them.
  * `kata-get-param PARAM`, `kata-set-param PARAM VALUE`
     * Get a parameter or set a parameter to a given value.
     * Current supported PARAMs are:
        * `playoutDoublingAdvantage (float)`. See documentation for this parameter in [the example config](..cpp/configs/gtp_example.cfg). Setting this via this command affects the value used for analysis, and affects play only if the config is not already set to use `dynamicPlayoutDoublingAdvantageCapPerOppLead`.
        * `analysisWideRootNoise (float)`. See documentation for this parameter in [the example config](..cpp/configs/gtp_example.cfg)
        * `numSearchThreads (int)`. The number of CPU threads to use in parallel, see documentation for this parameter in [the example config](..cpp/configs/gtp_example.cfg).
  * `kata-list-params`
     * Report all PARAMs supported by `kata-get-param`, separated by whitespace.
  * `cputime`, `gomill-cpu_time`
     * Returns the approximate total wall-clock-time spent during the handling of `genmove` or the various flavors of `genmove_analyze` commands described above so far during the entire current instance of the engine, as a floating point number of seconds. Does NOT currently count time spent during pondering or during the various `lz-analyze`, `kata-analyze`, etc.
     * Note: Gomill specifies that its variant of the command should return the total time summed across CPUs. For KataGo, this time is both unuseful and hard to measure because much of the time is spent waiting on the GPU, not on the CPU, and with different threads sometimes blocking each other through the multitheading and often exceeding the number of cores on a user's system, time spent on CPUs is hard to make sense of. So instead we report wall-clock-time, which is far more useful to record and should correspond more closely to what users may want to know for actual practical benchmarking and performance.
  * `benchmark NVISITS`
     * Run a benchmark using exactly the current search settings and board size except for any visit or playout or time limits ignored, instead using NVISITS visits.
     * Prints the result, in some user-readable format.
     * Will halt any ongoing search, may have the side effect of clearing the nn cache.
