## KataGo Parallel Analysis Engine

KataGo contains an engine that can be used to analyze large numbers of positions in parallel (entire games, or multiple games).
When properly configured and used with modern GPUs that can handle large batch sizes, this engine can be much faster than using
the GTP engine and `kata-analyze`, due to being able to take advantage of cross-position batching, and hopefully having a
nicer API. The analysis engine is primarily intended for people writing tools - for example, to run as the backend of an analysis
server or website.

This engine can be run via:

```./katago analysis -config CONFIG_FILE -model MODEL_FILE -analysis-threads NUM_ANALYSIS_THREADS```

An example config file is provided in `cpp/configs/analysis_example.cfg`. Adjusting this config is recommended, for example
setting `cudaUseFP16 = true` and `cudaUseNHWC = true` if you have a GPU with FP16 tensor core support, adjusting
`nnCacheSizePowerOfTwo` based on how much RAM you have, and adjusting `numSearchThreads` and `NUM_ANALYSIS_THREADS` as desired.

### Protocol

The engine accepts queries on stdin, and output results on stdout. Every query and every result should be a single line.
The protocol is entirely asynchronous - new requests on stdin can be accepted at any time, and results will appear on stdout
whenever those analyses finish, and possibly in a different order than the requests were provided. As described below, each query
may specify *multiple* positions to be analyzed and therefore may generate *multiple* results.

#### Queries

Each query line written to stdin should be a JSON dictionary with certain fields. Note again that every query must be a *single line* - multi-line JSON queries are NOT supported. An example query would be:

```{"id":"foo","initialStones":[["B","Q4"],["B","C4"]],"moves":[["W","P5"],["B","P6"]],"rules":"tromp-taylor","komi":7.5,"boardXSize":19,"boardYSize":19,"analyzeTurns":[0,1,2]}````

This example query specifies a 2-stone handicap game record with certain properties, and requests analysis of turns 0,1,2 of the game, which should produce three results.

Explanation of fields (including some optional fields not present in the above query):

   * `id (string)`: Required. An arbitrary string identifier for the query.
   * `moves (list of [player string, location string] tuples)`: Required. The moves that were played in the game, in the order they were played.
     * `player` should be `"B"` or `"W"`.
     * `location` should a string like `"C4"` the same as in the [GTP protocol](http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html#SECTION000311000000000000000). KataGo also supports extended column coordinates locations beyond `"Z"`, such as `"AA"`, `"AB"`, `"AC"`, ... Alternatively one can also specify strings like `"(0,13)"` that explicitly give the integer X and Y coordinates.
   * `initialStones (list of [player string, location string] tuples)`: Optional. Specifies stones already on the board at the start of the game. For example, these could be handicap stones. Or, you could use this to specify a midgame position or whole-board tsumego that does not have a move history.
   * `initialPlayer (player string)`: Optional. Specifies the player to use for analyzing the first turn of the game, which can be useful if there `moves` is an empty list.
   * `rules (string or JSON)`: Required. Specify the rules for the game using either a shorthand string or a full JSON object.
     * See the documentation of `kata-get-rules` and `kata-set-rules` in [GTP Extensions](./GTP_Extensions.md) for a description of supported rules.
     * Some older neural net versions of KataGo do not support some rules options. If this is the case, then a warning will be issued and the rules will
       automatically be converted to the nearest rules that the neural net does support.
   * `komi (integer or half-integer)`: Optional but HIGHLY recommended. Specify the komi for the game. If not specified, KataGo will guess a default value, generally 7.5 for area scoring, but 6.5 if using territory scoring, and 7.0 if area scoring with a button. Values of komi outside of [-150,150] are not supported.
   * `whiteHandicapBonus (0|N|N-1)`: Optional. See `kata-get-rules` in [GTP Extensions](./GTP_Extensions.md) for what these mean. Can be used to override the handling of handicap bonus, taking precedence over `rules`. E.g. if you want `chinese` rules but with different compensation for handicap stones. You could also always specify this as 0 and do any adjustment you like on your own, by reporting an appropriate `komi`.
   * `boardXSize (integer)`: Required. The width of the board. Sizes > 19 are NOT supported unless KataGo has been compiled to support them (cpp/game/board.h, MAX_LEN = 19). KataGo's official neural nets have also not been trained for larger boards, but should work fine for mildly larger sizes (21,23,25).
   * `boardYSize (integer)`: Required. The height of the board. Sizes > 19 are NOT supported unless KataGo has been compiled to support them (cpp/game/board.h, MAX_LEN = 19). KataGo's official neural nets have also not been trained for larger boards, but should work fine for mildly larger sizes (21,23,25).
   * `analyzeTurns (list of integers): Optional. Which turns of the game to analyze. If this field is not specified, defaults to analyzing the last turn only.
   * `maxVisits (integer)`: Optional. The maximum number of visits to use. If not specified, defaults to the value in the analysis config file. If specified, overrides it.
   * `rootPolicyTemperature (float)`: Optional. Set this to a value > 1 to make KataGo do a wider search.
   * `rootFpuReductionMax (float)`: Optional. Set this to 0 to make KataGo more willing to try a variety of moves.
   * `includeOwnership (boolean)`: Optional. If true, report ownership prediction as a result. Will double memory usage and reduce performance slightly.

#### Responses

Upon an error or a warning, responses will have one of the following formats:
```
# General error
{"error":"ERROR_MESSAGE"}
# Parsing error for a particular query field
{"error":"ERROR_MESSAGE","field":"name of the query field","id":"The id string for the query with the error"}
# Parsing warning for a particular query field
{"warning":"WARNING_MESSAGE","field":"name of the query field","id":"The id string for the query with the error"}
```
In the case of a warning, the query will still proceed to generate analysis responses.

An example successful analysis response might be:
```{"id":"foo","moveInfos":[{"lcb":0.7122210771137392,"move":"Q5","order":0,"prior":0.9710574746131897,"pv":["Q5","R5","Q6","R6","Q7","Q16","O3","Q3","R7","C16","E4"],"scoreLead":5.151662428174664,"scoreMean":5.151662428174664,"scoreSelfplay":6.876265583480511,"scoreStdev":24.963090370015202,"utility":0.4258535451266437,"utilityLcb":0.4459466007602151,"visits":998,"winrate":0.705044985816035},{"lcb":1.7784587604273838,"move":"D4","order":1,"prior":0.0046987771056592464,"pv":["D4"],"scoreLead":7.866552352905273,"scoreMean":7.866552352905273,"scoreSelfplay":10.166152000427246,"scoreStdev":25.506060917814644,"utility":0.595283210234703,"utilityLcb":2.8000000000000003,"visits":1,"winrate":0.7784587604273838}],"turnNumber":2}```

All values will be from the perspective of `reportAnalysisWinratesAs` as specified in the analysis config file.

Explanation of fields:

   * `id`: The same id string that was provided on the query.
   * `turnNumber`: The turn number being analyzed.
   * `moveInfos`: A list of JSON dictionaries, one per move that KataGo considered, with fields indicating the results of analysis. Consumers of this data should attempt to be robust to possible addition of new fields in the future. Possible fields are:
      * `move` - The move being analyzed.
      * `visits` - The number of visits invested into the move.
      * `winrate` - The winrate of the move, as a float in [0,1].
      * `scoreMean` - Same as scoreLead. "Mean" is a slight misnomer, but this field exists to preserve compatibility with existing tools.
      * `scoreStdev` - The predicted standard deviation of the final score of the game after this move, in points. (NOTE: due to the mechanics of MCTS, this value will be significantly biased high currently, although it can still be informative as *relative* indicator).
      * `scoreLead` - The predicted average number of points that the current side is leading by (with this many points fewer, it would be an even game).
      * `scoreSelfplay` - The predicted average value of the final score of the game after this move during selfplay, in points. (Note: users should usually prefer scoreLead, since scoreSelfplay may be biased by the fact that KataGo isn't perfectly score-maximizing).
      * `prior` - The policy prior of the move, as a float in [0,1].
      * `utility` - The utility of the move, combining both winrate and score, as a float in [-C,C] where C is the maximum possible utility.
      * `lcb` - The [LCB](https://github.com/leela-zero/leela-zero/issues/2282) of the move's winrate, as a float in [0,1].
      * `utilityLcb` - The LCB of the move's utility.
      * `order` - KataGo's ranking of the move. 0 is the best, 1 is the next best, and so on.
      * `pv` - The principal variation following this move. May be of variable length or even empty.
