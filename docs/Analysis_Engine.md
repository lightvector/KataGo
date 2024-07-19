# KataGo Parallel Analysis Engine

KataGo contains an engine that can be used to analyze large numbers of positions in parallel (entire games, or multiple games).
When properly configured and used with modern GPUs that can handle large batch sizes, this engine can be much faster than using
the GTP engine and `kata-analyze`, due to being able to take advantage of cross-position batching, and hopefully having a
nicer API. The analysis engine is primarily intended for people writing tools - for example, to run as the backend of an analysis
server or website.

This engine can be run via:

```./katago analysis -config CONFIG_FILE -model MODEL_FILE```

An example config file is provided in `cpp/configs/analysis_example.cfg`. Adjusting this config is recommended, for example
`nnCacheSizePowerOfTwo` based on how much RAM you have, and adjusting `numSearchThreadsPerAnalysisThread` (the number of MCTS threads operating simultaneously on the same position) and `numAnalysisThreads` (the number of positions that will be analyzed at the same time, *each* of which will use `numSearchThreadsPerAnalysisThread` many search threads).

See the [example analysis config](https://github.com/lightvector/KataGo/blob/master/cpp/configs/analysis_example.cfg#L60) for a fairly detailed discussion of how to tune these parameters.

## Example Code

For example code demonstrating how to invoke the analysis engine from Python, see [here](https://github.com/lightvector/KataGo/blob/master/python/query_analysis_engine_example.py)!

## Protocol

The engine accepts queries on stdin, and output results on stdout. Every query and every result should be a single line.
The protocol is entirely asynchronous - new requests on stdin can be accepted at any time, and results will appear on stdout
whenever those analyses finish, and possibly in a different order than the requests were provided. As described below, each query
may specify *multiple* positions to be analyzed and therefore may generate *multiple* results.

If stdin is closed, then the engine will finish the analysis of all queued queries before exiting, unless `-quit-without-waiting` was
provided on the initial command line, in which case it will attempt to stop all threads and still exit cleanly but without
necessarily finishing the analysis of whatever queries are open at the time.

### Queries

Each query line written to stdin should be a JSON dictionary with certain fields. Note again that every query must be a *single line* - multi-line JSON queries are NOT supported. An example query would be:

```json
{"id":"foo","initialStones":[["B","Q4"],["B","C4"]],"moves":[["W","P5"],["B","P6"]],"rules":"tromp-taylor","komi":7.5,"boardXSize":19,"boardYSize":19,"analyzeTurns":[0,1,2]}
```

<details>
<summary>
See formatted query for readability (but note that this is not valid input for KataGo, since it spans multiple lines).
</summary>

```json
{
    "id": "foo",
    "initialStones": [
        ["B", "Q4"],
        ["B", "C4"]
    ],
    "moves": [
        ["W", "P5"],
        ["B", "P6"]
    ],
    "rules": "tromp-taylor",
    "komi": 7.5,
    "boardXSize": 19,
    "boardYSize": 19,
    "analyzeTurns": [0, 1, 2]
}
```
</details>

This example query specifies a 2-stone handicap game record with certain properties, and requests analysis of turns 0,1,2 of the game, which should produce three results.

Explanation of fields (including some optional fields not present in the above query):

   * `id (string)`: Required. An arbitrary string identifier for the query.
   * `moves (list of [player string, location string] tuples)`: Required. The moves that were played in the game, in the order they were played.
     * `player string` should be `"B"` or `"W"`.
     * `location` should a string like `"C4"` the same as in the [GTP protocol](http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html#SECTION000311000000000000000). KataGo also supports extended column coordinates locations beyond `"Z"`, such as `"AA"`, `"AB"`, `"AC"`, ... Alternatively one can also specify strings like `"(0,13)"` that explicitly give the integer X and Y coordinates.
     * Leave this array empty if you have an initial position with no move history (do not make up an arbitrary or "fake" order of moves).
   * `initialStones (list of [player string, location string] tuples)`: Optional. Specifies stones already on the board at the start of the game. For example, these could be handicap stones. Or, you could use this to specify a midgame position or whole-board tsumego that does not have a move history.
     * If you know the real game moves that reached a position, using `moves` is usually preferable to specifying all the stones here while leaving `moves` as an empty array, since using `moves` ensures correct ko/superko handling, and the neural net may also take into account the move history in its future predictions.
   * `initialPlayer (player string)`: Optional. Specifies the player to use for analyzing the first turn (turn 0) of the game, which can be useful if `moves` is an empty list.
   * `rules (string or JSON)`: Required. Specify the rules for the game using either a shorthand string or a full JSON object.
     * See the documentation of `kata-get-rules` and `kata-set-rules` in [GTP Extensions](./GTP_Extensions.md) for a description of supported rules.
     * Some older neural net versions of KataGo do not support some rules options. If this is the case, then a warning will be issued and the rules will
       automatically be converted to the nearest rules that the neural net does support.
   * `komi (integer or half-integer)`: Optional but HIGHLY recommended. Specify the komi for the game. If not specified, KataGo will guess a default value, generally 7.5 for area scoring, but 6.5 if using territory scoring, and 7.0 if area scoring with a button. Values of komi outside of [-150,150] are not supported.
   * `whiteHandicapBonus (0|N|N-1)`: Optional. See `kata-get-rules` in [GTP Extensions](./GTP_Extensions.md) for what these mean. Can be used to override the handling of handicap bonus, taking precedence over `rules`. E.g. if you want `chinese` rules but with different compensation for handicap stones than Chinese rules normally use. You could also always specify this as 0 and do any adjustment you like on your own, by reporting an appropriate `komi`.
   * `boardXSize (integer)`: Required. The width of the board. Sizes > 19 are NOT supported unless KataGo has been compiled to support them (cpp/game/board.h, MAX_LEN = 19). KataGo's official neural nets have also not been trained for larger boards, but should work fine for mildly larger sizes (21,23,25).
   * `boardYSize (integer)`: Required. The height of the board. Sizes > 19 are NOT supported unless KataGo has been compiled to support them (cpp/game/board.h, MAX_LEN = 19). KataGo's official neural nets have also not been trained for larger boards, but should work fine for mildly larger sizes (21,23,25).
   * `analyzeTurns (list of integers)`: Optional. Which turns of the game to analyze. 0 is the initial position, 1 is the position after `moves[0]`, 2 is the position after `moves[1]`, etc. If this field is not specified, defaults to analyzing only the last turn, which is the position after all specified `moves` are made.
   * `maxVisits (integer)`: Optional. The maximum number of visits to use. If not specified, defaults to the value in the analysis config file. If specified, overrides it.
   * `rootPolicyTemperature (float)`: Optional. Set this to a value > 1 to make KataGo do a wider search.
   * `rootFpuReductionMax (float)`: Optional. Set this to 0 to make KataGo more willing to try a variety of moves.
   * `analysisPVLen (integer)`: Optional. The maximum length of the PV to send for each move (not including the first move).
   * `includeOwnership (boolean)`: Optional. If true, report ownership prediction as a result. Will double memory usage and reduce performance slightly.
   * `includeOwnershipStdev (boolean)`: Optional. If true, report standard deviation of ownership predictions across the search as well.
   * `includeMovesOwnership (boolean)`: Optional. If true, report ownership prediction for every individual move too.
   * `includeMovesOwnershipStdev (boolean)`: Optional. If true, report stdev of ownership prediction for every individual move too.
   * `includePolicy (boolean)`: Optional. If true, report neural network raw policy as a result. Will not signficiantly affect performance.
   * `includePVVisits (boolean)`: Optional. If true, report the number of visits for each move in any reported pv.
   * `avoidMoves (list of dicts)`: Optional. Prohibit the search from exploring the specified moves for the specified player, until a certain number of ply deep in the search. Each dict must contain these fields:
      * `player` - the player to prohibit, `"B"` or `"W"`.
      * `moves` - an array of move locations to prohibit, such as `["C3","Q4","pass"]`
      * `untilDepth` - a positive integer, indicating the ply such that moves are prohibited before that ply.
      * Multiple dicts can specify different `untilDepth` for different sets of moves. The behavior is unspecified if a move is specified more than once with different `untilDepth`.
   * `allowMoves (list of dicts)`: Optional. Same as `avoidMoves` except prohibits all moves EXCEPT the moves specified. Currently, the list of dicts must also be length 1.
   * `overrideSettings (object)`: Optional. Specify any number of `"paramName":value` entries in this object to override those params from command line `CONFIG_FILE` for this query. Most search parameters can be overriden: `cpuctExploration`, `winLossUtilityFactor`, etc. Some notable parameters include:
      * `playoutDoublingAdvantage (float)`. A value of PDA from -3 to 3 will adjust KataGo's evaluation to assume that the opponent is NOT of equal strength/compte, but rather that the current player has 2^(PDA) times as many playouts as the opponent. Dynamic versions of this are used to significant effect in handicap games in GTP mode, see [GTP example config](../cpp/configs/gtp_example.cfg).
        * `wideRootNoise (float)`. See documentation for this parameter in [the example config](../cpp/configs/analysis_example.cfg)
        * `ignorePreRootHistory (boolean)`. Whether to ignore pre-root history during analysis.
        * `antiMirror (boolean)`. Whether to enable anti-mirror play during analysis. Off by default. Will probably result in biased and nonsensical winrates and other analysis values, but moves may detect and crudely respond to mirror play.
        * `rootNumSymmetriesToSample (int from 1 to 8)`. How many of the 8 possible random symmetries to evaluate the neural net with and average. Defaults to 1, but if you set this to 2, or 8 you might get a slightly higher-quality policy at the root due to noise reduction.
        * `humanSLProfile (string)`. Set the human-like play that KataGo should imitate. Requires that a human SL model like `b18c384nbt-humanv0.bin.gz` is being used, typically via the command line parameter `-human-model`. Available profiles include:
           * `preaz_20k` through `preaz_9d`: Imitate human players of the given rank. (based on 2016 pre-AlphaZero opening style).
           * `rank_20k` through `rank_9d`: Imitate human players of the given rank (modern opening style).
           * `preaz_{BR}_{WR}` or `rank_{BR}_{WR}`: Same, but predict how black with the rank BR and white with the rank WR would play against each other, *knowing* that the other player is stronger/weaker than them. Warning: for rank differences > 9 ranks, or drastically mis-matched to the handicap used in the game, this may be out of distribution due to lack of training data and the model might not behave well! Experiment with care.
           * `proyear_1800` through `proyear_2023`: Imitate pro and strong insei moves based on historical game records from the specified year and surrounding years.
           * See also section below, "Human SL Analysis Guide" for various other parameters that are interesting to set in conjunction with this.
   * `reportDuringSearchEvery (float)`: Optional. Specify a number of seconds such that while this position is being searched, KataGo will report the partial analysis every that many seconds.
   * `priority (int)`: Optional. Analysis threads will prefer handling queries with the highest priority unless already started on another task, breaking ties in favor of earlier queries. If not specified, defaults to 0.
   * `priorities (list of integers)`: Optional. When using analyzeTurns, you can use this instead of `priority` if you want a different priority per turn. Must be of same length as `analyzeTurns`, `priorities[0]` is the priority for `analyzeTurns[0]`, `priorities[1]` is the priority for `analyzeTurns[1]`, etc.


### Responses

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
```json
{"id":"foo","isDuringSearch":false,"moveInfos":[{"lcb":0.8740855166489953,"move":"Q5","order":0,"prior":0.8934692740440369,"pv":["Q5","R5","Q6","P4","O5","O4","R6","S5","N4","N5","N3"],"scoreLead":8.18535151076558,"scoreMean":8.18535151076558,"scoreSelfplay":10.414442461570038,"scoreStdev":23.987067985850913,"utility":0.7509536097709347,"utilityLcb":0.7717092488727239,"visits":495,"edgeVisits":495,"winrate":0.8666727883983563},{"lcb":1.936558574438095,"move":"D4","order":1,"prior":0.021620146930217743,"pv":["D4","Q5"],"scoreLead":12.300520420074463,"scoreMean":12.300520420074463,"scoreSelfplay":15.386500358581543,"scoreStdev":24.661467510313432,"utility":0.9287495791972984,"utilityLcb":2.8000000000000003,"visits":2,"edgeVisits":2,"winrate":0.9365585744380951},{"lcb":1.9393062554299831,"move":"Q16","order":2,"prior":0.006689758971333504,"pv":["Q16"],"scoreLead":12.97426986694336,"scoreMean":12.97426986694336,"scoreSelfplay":16.423904418945313,"scoreStdev":25.34494674587838,"utility":0.9410896213959669,"utilityLcb":2.8000000000000003,"visits":1,"edgeVisits":1,"winrate":0.9393062554299831},{"lcb":1.9348860532045364,"move":"D16","order":3,"prior":0.0064553022384643555,"pv":["D16"],"scoreLead":12.066888809204102,"scoreMean":12.066888809204102,"scoreSelfplay":15.591397285461426,"scoreStdev":25.65390196745236,"utility":0.9256971928661066,"utilityLcb":2.8000000000000003,"visits":1,"edgeVisits":1,"winrate":0.9348860532045364}],"rootInfo":{"currentPlayer":"B","lcb":0.8672585456293346,"scoreLead":8.219540952281882,"scoreSelfplay":10.456476293719811,"scoreStdev":23.99829921716391,"symHash":"1D25038E8FC8C26C456B8DF2DBF70C02","thisHash":"F8FAEDA0E0C89DDC5AA5CCBB5E7B859D","utility":0.7524437705003542,"visits":500,"winrate":0.8672585456293346},"turnNumber":2}
```
<details>
<summary>
See formatted response.
</summary>

```json
{
    "id": "foo",
    "isDuringSearch": false,
    "moveInfos": [{
        "lcb": 0.8740855166489953,
        "move": "Q5",
        "order": 0,
        "prior": 0.8934692740440369,
        "pv": ["Q5", "R5", "Q6", "P4", "O5", "O4", "R6", "S5", "N4", "N5", "N3"],
        "scoreLead": 8.18535151076558,
        "scoreMean": 8.18535151076558,
        "scoreSelfplay": 10.414442461570038,
        "scoreStdev": 23.987067985850913,
        "utility": 0.7509536097709347,
        "utilityLcb": 0.7717092488727239,
        "visits": 495,
        "edgeVisits": 495,
        "winrate": 0.8666727883983563
    }, {
        "lcb": 1.936558574438095,
        "move": "D4",
        "order": 1,
        "prior": 0.021620146930217743,
        "pv": ["D4", "Q5"],
        "scoreLead": 12.300520420074463,
        "scoreMean": 12.300520420074463,
        "scoreSelfplay": 15.386500358581543,
        "scoreStdev": 24.661467510313432,
        "utility": 0.9287495791972984,
        "utilityLcb": 2.8000000000000003,
        "visits": 2,
        "edgeVisits": 2,
        "winrate": 0.9365585744380951
    }, {
        "lcb": 1.9393062554299831,
        "move": "Q16",
        "order": 2,
        "prior": 0.006689758971333504,
        "pv": ["Q16"],
        "scoreLead": 12.97426986694336,
        "scoreMean": 12.97426986694336,
        "scoreSelfplay": 16.423904418945313,
        "scoreStdev": 25.34494674587838,
        "utility": 0.9410896213959669,
        "utilityLcb": 2.8000000000000003,
        "visits": 1,
        "edgeVisits": 1,
        "winrate": 0.9393062554299831
    }, {
        "lcb": 1.9348860532045364,
        "move": "D16",
        "order": 3,
        "prior": 0.0064553022384643555,
        "pv": ["D16"],
        "scoreLead": 12.066888809204102,
        "scoreMean": 12.066888809204102,
        "scoreSelfplay": 15.591397285461426,
        "scoreStdev": 25.65390196745236,
        "utility": 0.9256971928661066,
        "utilityLcb": 2.8000000000000003,
        "visits": 1,
        "edgeVisits": 1,
        "winrate": 0.9348860532045364
    }],
    "rootInfo": {
        "currentPlayer": "B",
        "lcb": 0.8672585456293346,
        "scoreLead": 8.219540952281882,
        "scoreSelfplay": 10.456476293719811,
        "scoreStdev": 23.99829921716391,
        "symHash":"1D25038E8FC8C26C456B8DF2DBF70C02",
        "thisHash":"F8FAEDA0E0C89DDC5AA5CCBB5E7B859D",
        "utility": 0.7524437705003542,
        "visits": 500,
        "winrate": 0.8672585456293346
    },
    "turnNumber": 2
}
```
</details>


**All values will be from the perspective of `reportAnalysisWinratesAs` as specified in the analysis config file.**

Consumers of this data should attempt to be robust to possible addition of both new top-level fields in the future, as well as additions to fields in `moveInfos` or `rootInfo`.

The various "human" fields are available if -human-model is provided and humanSLProfile is set.

Current fields are:

   * `id`: The same id string that was provided on the query.
   * `isDuringSearch`: Normally false. If `reportDuringSearchEvery` is provided, then will be true on the reports during the middle of the search before the search is complete. Every position searched will still always conclude with exactly one final response when the search is completed where this field is false.
   * `turnNumber`: The turn number being analyzed.
   * `moveInfos`: A list of JSON dictionaries, one per move that KataGo considered, with fields indicating the results of analysis. Current fields are:
      * `move` - The move being analyzed.
      * `visits` - The number of visits that the child node received.
      * `edgeVisits` - The number of visits that the root node "wants" to invest in the move, due to thinking it's a plausible or search-worthy move. Might differ from `visits` due to human SL weightless exploration, or graph search transpositions.
      * `winrate` - The winrate of the move, as a float in [0,1].
      * `scoreMean` - Same as scoreLead. "Mean" is a slight misnomer, but this field exists to preserve compatibility with existing tools.
      * `scoreStdev` - The predicted standard deviation of the final score of the game after this move, in points. (NOTE: due to the mechanics of MCTS, this value will be **significantly biased high** currently, although it can still be informative as *relative* indicator).
      * `scoreLead` - The predicted average number of points that the current side is leading by (with this many points fewer, it would be an even game).
      * `scoreSelfplay` - The predicted average value of the final score of the game after this move during selfplay, in points. (NOTE: users should usually prefer scoreLead, since scoreSelfplay may be biased by the fact that KataGo isn't perfectly score-maximizing).
      * `prior` - The policy prior of the move, as a float in [0,1].
      * `humanPrior` - The human policy for the move, as a float in [0,1], if available.
      * `utility` - The utility of the move, combining both winrate and score, as a float in [-C,C] where C is the maximum possible utility. The maximum winrate utility can be set by `winLossUtilityFactor` in the config, while the maximum score utility is the sum of `staticScoreUtilityFactor` and `dynamicScoreUtilityFactor`.
      * `lcb` - The [LCB](https://github.com/leela-zero/leela-zero/issues/2282) of the move's winrate. Has the same units as winrate, but might lie outside of [0,1] since the current implementation doesn't strictly account for the 0-1 bounds.
      * `utilityLcb` - The LCB of the move's utility.
      * `weight` - The total weight of the visits that the child node received. The average weight of visits may be lower when less certain, and larger when more certain.
      * `edgeWeight` - The total weight of the visits the parent wants to invest into the move. The average weight of visits may be lower when less certain, and larger when more certain.
      * `order` - KataGo's ranking of the move. 0 is the best, 1 is the next best, and so on.
      * `isSymmetryOf` - Another legal move. Possibly present if KataGo is configured to avoid searching some moves due to symmetry (`rootSymmetryPruning=true`). If present, this move was not actually searched, and all of its stats and PV are copied symmetrically from that other move.
      * `pv` - The principal variation ("PV") following this move. May be of variable length or even empty.
      * `pvVisits` - The number of visits used to explore the position resulting from each move in `pv`. Exists only if `includePVVisits` is true.
      * `pvEdgeVisits` - The number of visits used to explore each move in `pv`. Exists only if `includePVVisits` is true. Differs from pvVisits when doing graph search and multiple move sequences lead to the same position - pvVisits will count the total number of visits for the position at that point in the PV, pvEdgeVisits will count only the visits reaching the position using the move in the PV from the preceding position.
      * `ownership` - If `includeMovesOwnership` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from -1 to 1 indicating the predicted ownership after this move. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
      * `ownershipStdev` - If `includeMovesOwnershipStdev` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from 0 to 1 indicating the per-location standard deviation of predicted ownership in the search tree after this move. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
   * `rootInfo`: A JSON dictionary with fields containing overall statistics for the requested turn itself calculated in the same way as they would be for the next moves. Current fields are: `winrate`, `scoreLead`, `scoreSelfplay`, `utility`, `visits`. And additional fields:
      * `thisHash` - A string that will with extremely high probability be unique for each distinct (board position, player to move, simple ko ban) combination.
      * `symHash` - Like `thisHash` except the string will be the same between positions that are symmetrically equivalent. Does NOT necessarily take into account superko.
      * `currentPlayer` - The current player whose possible move choices are being analyzed, `"B"` or `"W"`.
      * `rawWinrate` - The winrate prediction of the neural net by itself, without any search.
      * `rawLead` - The lead prediction of the neural net by itself, without any search.
      * `rawScoreSelfplay` - The selfplay score prediction of the neural net by itself, without any search.
      * `rawScoreSelfplayStdev` - The standard deviation of the final game score predicted by the net itself, without any search.
      * `rawNoResultProb` - The raw predicted probability of a no result game in Japanese-like rules.
      * `rawStWrError` - The short-term uncertainty the raw neural net believed there would be in the winrate of the position, prior to searching it.
      * `rawStScoreError` - The short-term uncertainty the raw neural net believed there would be in the score of the position, prior to searching it.
      * `rawVarTimeLeft` - The raw neural net's guess of "how long of a meaningful game is left?", in no particular units. A large number when expected that it will be a long game before the winner becomes clear. A small number when the net believes the winner is already clear, or that the winner is unclear but will become clear soon.
      * `humanWinrate` - Same as `rawWinrate` but using the human model, if available.
      * `humanScoreMean` - Same as `rawScoreSelfplay` but using the human model, if available.
      * `humanScoreStdev` - Same as `rawScoreSelfplayStdev` but using the human model, if available.
      * `humanStWrError` - The short-term uncertainty the raw neural net believes there will be in the winrate of the position as it gets played out by players of the configured profile, using the human model, if available.
      * `humanStScoreError` - The short-term uncertainty the raw neural net believes there will be in the score evaluation of the position as it gets played out by players of the configured profile, using the human model, if available.
      * Note that properties of the root like "winrate" and score will vary more smoothly and a bit more sluggishly than the corresponding property of the best move, since the rootInfo averages smoothly across all visits even while the top move may fluctuate rapidly. This may or may not be preferable over reporting the stats of the top move, depending on the purpose.
   * `ownership` - If `includeOwnership` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from -1 to 1 indicating the predicted ownership. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
   * `ownershipStdev` - If `includeOwnershipStdev` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from 0 to 1 indicating the per-location standard deviation of predicted ownership in the search tree. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
   * `policy` - If `includePolicy` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize + 1` with positive values summing to 1 indicating the neural network's prediction of the best move before any search, and `-1` indicating illegal moves. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1). The last value in the array is the policy value for passing.
   * `humanPolicy` - If `includePolicy` was true, and a human model is available, then this field will be included. The format is the same as `policy`, but it reports the policy from the human model based on the configured `humanSLProfile`. See also section below, "Human SL Analysis Guide".


### Special Action Queries

Currently a few special action queries are supported that direct the analysis engine to do something other than enqueue a new position or set of positions for analysis.
A special action query is also sent as a JSON object, but with a different set of fields depending on the query.

#### query_version
Requests that KataGo report its current version. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `query_version`.

Example:
```
{"id":"foo","action":"query_version"}
```

The response to this query is to echo back a json object with exactly the same data and fields of the query, but with two additional fields:

   * `version (string)`: A string indicating the most recent KataGo release version that this version is a descendant of, such as `1.6.1`.
   * `git_hash (string)`: The precise git hash this KataGo version was compiled from, or the string `<omitted>` if KataGo was compiled separately from its repo or without Git support.

Example:
```
{"action":"query_version","git_hash":"0b0c29750fd351a8364440a2c9c83dc50195c05b","id":"foo","version":"1.6.1"}
```

#### clear_cache
Requests that KataGo empty its neural net cache. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `clear_cache`.

Example:
```
{"id":"foo","action":"clear_cache"}
```
The response to this query is to simply echo back a json object with exactly the same data and fields of the query. This response is sent after the cache is successfully cleared. If there are also any ongoing analysis queries at the time, those queries will of course be concurrently refilling the cache even as the response is being sent.

Explanation: KataGo uses a cache of neural net query results to skip querying the neural net when it encounters within its search tree a position whose stone configuration, player to move, ko status, komi, rules, and other relevant options are all identical a position it has seen before. For example, this may happen if the search trees for some queries overlap due to being on nearby moves of the same game, or it may happen even within a single analysis query if the search explores differing orders of moves that lead to the same positions (often, about 20% of search tree nodes hit the cache due transposing to order of moves, although it may be vastly higher or lower depending on the position and search depth). Reasons for wanting to clear the cache may include:

* Freeing up RAM usage - emptying the cache should release the memory used for the results in the cache, which is typically the largest memory usage in KataGo. Memory usage will of course rise again as the cache refills.

* Testing or studying the variability of KataGo's search results for a given number of visits. Analyzing a position again after a cache clear will give a "fresh" look on that position that better matches the variety of possible results KataGo may return, simliar to if the analysis engine were entirely restarted. Each query will re-randomize the symmetry of the neural net used for that query instead of using the cached result, giving a new and more varied opinion.


#### terminate

Requests that KataGo terminate zero or more analysis queries without waiting for them to finish normally. When a query is terminated, the engine will make a best effort to halt their analysis as soon as possible, reporting the results of whatever number of visits were performed up to that point. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `terminate`.
   * `terminateId (string)`: Required. Terminate queries that were submitted with this `id` field without analyzing or finishing analyzing them.
   * `turnNumbers (array of ints)`: Optional. If provided, restrict only to terminating the queries with that id that were for these turn numbers.

Examples:
```
{"id":"bar","action":"terminate","terminateId":"foo"}
{"id":"bar","action":"terminate","terminateId":"foo","turnNumbers":[1,2]}
```

Responses to terminated queries may be missing their data fields if no analysis at all was performed before termination. In such a case, the only fields guaranteed to be on the response are `id` and `turnNumber`, and `isDuringSearch` (which will always be false), as well as one additional boolean field unique to terminated queries that did not analyze at all, `noResults` (which will always be true). Example:
```
{"id":"foo","isDuringSearch":false,"noResults":true,"turnNumber":2}
```

The terminate query itself will result in a response as well, to acknowledge receipt and processing of the action. The response consists of echoing a json object back with exactly the same fields and data of the query.

The response will NOT generally wait for all of the effects of the action to take place - it may take a small amount of additional time for ongoing searches to actually terminate and report their partial results. A client of this API that wants to wait for all terminated queries to finish should on its own track the set of queries that it has sent for analysis, and wait for all of them to have finished. This can be done by relying on the property that every analysis query, whether terminated or not, and regardless of `reportDuringSearchEvery`, will conclude with exactly one reply where `isDuringSearch` is `false` - such a reply can therefore be used as a marker that an analysis query has finished. (Except during shutdown of the engine if `-quit-without-waiting` was specified).

#### terminate_all

The same as terminate but does not require providing a `terminateId` field and applies to all queries, regardless of their `id`. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `terminate_all`.
   * `turnNumbers (array of ints)`: Optional. If provided, restrict only to terminating the queries for these turn numbers.

Examples:
```
{"id":"bar","action":"terminate_all"}
{"id":"bar","action":"terminate_all","turnNumbers":[1,2]}
```
The terminate_all query itself will result in a response as well, to acknowledge receipt and processing of the action. The response consists of echoing a json object back with exactly the same fields and data of the query.

See the documentation for terminate above regarding the output from terminated queries. As with terminate, the response to terminate_all will NOT wait for all of the effects of the action to take place, and the results of all the old queries as they are terminated will be reported back asynchronously.

#### query_models

Requests that KataGo report information about the loaded models. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `query_models`.

Example:
```json
{"id":"foo","action":"query_models"}
```

The response to this query will echo back the same keys passed in, along with a key "models" containing an array of the models loaded. Each model in the array includes details such as the model name, internal name, maximum batch size, whether it uses a human SL profile, version, and FP16 usage. Example:

```json
{
  "id": "foo",
  "action": "query_models",
  "models": [
    {
      "name": "kata1-b18c384nbt-s9732312320-d4245566942.bin.gz",
      "internalName": "kata1-b18c384nbt-s9732312320-d4245566942",
      "maxBatchSize": 256,
      "usesHumanSLProfile": false,
      "version": 14,
      "usingFP16": "auto"
    },
    {
      "name": "b18c384nbt-humanv0.bin.gz",
      "internalName": "b18c384nbt-humanv0",
      "maxBatchSize": 256,
      "usesHumanSLProfile": true,
      "version": 15,
      "usingFP16": "auto"
    }
  ]
}
```

## Human SL Analysis Guide

As of version 1.15.0, released July 2024, KataGo supports a new human supervised learning ("human SL") model `b18c384nbt-humanv0.bin.gz` that was trained on a large number of human games to predict moves by players of all different ranks and the outcomes of those games. People have only just started to experiment with the model and there might be many creative possibilities for analysis or play.

See also the notes on "humanSL" and other parameters within the [GTP human 5k example config](../cpp/configs/gtp_human5k_example.cfg). Although this is a GTP config, not an analysis engine config, the inline documentation about how the "humanSL" parameters behave is just as applicable to the analysis engine.

Similarly, for GTP users, most of the below notes are just as applicable to GTP play and analysis (used by engines like Lizzie or Sabaki) despite being written from the perspective of the analysis engine.

Below are some notes and suggestions for starting points on playing with the human SL model.

### Setting Up to Use the Model

There are two ways to pass in the human SL model.

* The basic intended way: pass an additional argument `-human-model b18c384nbt-humanv0.bin.gz` in addition to still passing in KataGo's normal model.
   * For example: `./katago analysis -config configs/analysis_example.cfg -model models/kata1-b28c512nbt-s7382862592-d4374571218.bin.gz -human-model models/b18c384nbt-humanv0.bin.gz`.
   * Additionally, provide `humanSLProfile` via `overrideSettings` on queries. See documentation above for `overrideSettings`.
   * Additionally, make sure to request `"includePolicy":true` in the query.
   * Then, a new `humanPolicy` field will be reported on the result, indicating KataGo's prediction of how random human players matching the given humanSLProfile (e.g. 5 kyu rank) might play.
   * If no further parameters are set, KataGo's main model will still be used for all other analysis.
   * If further parameters are set, interesting *blended* usages of the KataGo's main model and the human SL model are possible. See some "recipes" below.

* An alternative way: pass `-model b18c384nbt-humanv0.bin.gz` instead of KataGo's normal model, using the human model exclusively.
   * For example: `./katago analysis -config configs/analysis_example.cfg -model models/b18c384nbt-humanv0.bin.gz`.
   * Additionally, provide `humanSLProfile` via `overrideSettings` on queries. See documentation above for `overrideSettings`.
   * Then, KataGo will use the human model at the configured profile for all analysis, rather than its normal typically-superhuman analysis.
   * Note that if you are searching with many visits (or even just a few visits!), typically you can expect that KataGo will NOT match the strength of a player of the given humanSLProfile, but will still be stronger because the search will probably solve a lot of tactics that players of a weaker rank would not solve.
      * The human SL model is trained such that using only *one* visit, and full temperature (i.e. choosing random moves from the policy proportionally often, rather than always choosing the top move), will give the closest match to how players of the given rank might play. This should be true up to mid-high dan level, at which point the raw model might start to fall short and need more than 1 visit to keep up in strength.

   * If used as the main model, the human SL model may have significantly more pathologies and biases in its reported winrates and scores than KataGo's normal model, due to the SGF data it trained on.
      * For example, in handicap games it might not report accurate scores or winrates because in recorded human handicap SGFs, White is usually a stronger player than Black and/or some servers may underhandicap games, biasing the result.
      * For example, in even games, it might report erroneous scores and winrates after a big swing or in extreme positions, due to how human players may resign or go on tilt, or due to inaccurately recorded player ranks in the training data, or due to some fraction of sandbagger/airbagger games or AI cheating games in the training data.


### Recipes for Various HumanSL Usages

Here is a brief guide to some example usages, and hopefully a bit of inspiration for possible things to try.

Except for parameters explicitly documented earlier as belonging on the outer json query object (e.g. `includePolicy`, `maxVisits`), the parameters described below should be set within the `overrideSettings` of a query. E.g:

```
"overrideSettings":{"humanSLProfile":"rank_3d","ignorePreRootHistory":false,"humanSLRootExploreProbWeightless":0.5,"humanSLCpuctPermanent":2.0}```

Do NOT set such parmeters as a key of the outer json query object, as that will have no effect. KataGo should issue a warning if you accidentally do. If desired, you can also hardcode parameters within the analysis config file, e.g. `humanSLProfile = rank_3d`.

This guide is also applicable for GTP users, for configuring play and GTP-based analysis (e.g. kata-analyze). For GTP, set parameters within the GTP config file, and optionally change then dynamically via `kata-set-param` ([GTP Extensions](./GTP_Extensions.md)).

#### Human-like play

For simply imitating how a player of a given rank would play, the recommended way is:

* Set `humanSLProfile` appropriately.
* Set `ignorePreRootHistory` to `false` (normally analysis ignores history to be unbiased by move order, but humans definitely behave differently based on recent moves!).
* Send a query with any number of visits (even 1 visit) with `"includePolicy":true` specified on the outer json query object.
* Read `humanPolicy` from the result and pick a random move according to the policy probabilities.

Note that since old historical human games from training might vary in whether they record passes at all, it's possible the human SL net could have trouble passing appropriately in some board positions for some humanSLProfiles. For some weaker ranks, it's possible the human SL net may pass too early and leave positions unfinished in an undesirable way. If so, then the following should work well:

* Set `humanSLProfile` appropriately.
* Set `ignorePreRootHistory` to `false`.
* Send a query with at least a few visits so KataGo can search the position itself (e.g. > 50 visits), still with `"includePolicy":true`.
* If the top moveInfo from the result (the moveInfo with `"order":0`) is a pass, then pass.
* Otherwise, read `humanPolicy` and pick a random move proportional to the policy probabilities, except excluding passing.

(Note: For GTP users, [gtp_human5k_example.cfg](../cpp/configs/gtp_human5k_example.cfg) already does human imitation play by default, with some GTP-specific hacks and parameters to get KataGo's move selection to use the human SL model in the above kind of way. See documentation in that config.)

Optionally, also you can set `rootNumSymmetriesToSample` to `2`, or to `8` instead of the default `1`. This will slightly add latency but improve the quality of the human policy by averaging more symmetries, which might be good when relying so heavily on the raw human policy without any search.

#### Ensuring all likely human moves are analyzed

For analysis and game review, if you want to ensure all moves with high human policy get plenty of visits, you can try settings like the following:

* Set `humanSLProfile` and `ignorePreRootHistory` and `rootNumSymmetriesToSample` as desired.
* Set `humanSLRootExploreProbWeightless` to `0.5` (spend about 50% of playouts to explore human moves, in a weightless way that doesn't bias KataGo's evaluations).
* Set `humanSLCpuctPermanent` to `2.0` or similar (when exploring human moves, ensure high-human-policy moves get many visits even if they lose a lot). Set it to something lower if you want to reduce visits for moves that are judged to be very bad.
* Make sure to use plenty of visits overall.

#### Possible metrics that might be interesting

If you've ensured that all likely human moves are analyzed, there might be some interesting kinds of metrics to consider that can be derived from the human policy:

* Mean score that a player would have after the current move, if sampling from the human policy, `sum(scoreLead * humanPrior) / sum(humanPrior)`.
* Standard deviation of score change due to current move, if sampling from the human policy, `sqrt(sum((scoreLead-m)**2 * humanPrior) / sum(humanPrior))` where m is the above mean.
* Difference in the human policy of the move played between the current rank and a player several ranks higher (send 1-visit queries with other humanSLProfiles to obtain the humanPolicy for other ranks).
* Is something like "(actual score - mean score) / standard deviation of score" an interesting alternative to simply the absolute score loss for judging a mistake?
* Is sorting or weighting mistakes by the amount that a player 4 ranks higher would be less likely to play that move, or other similar kinds of metrics, a good way to bias a game review towards mistakes that are more level-appropriate for a player to review?

#### How to get stronger human-style play

If you want to obtain human *style* moves, but playing stronger than a given human level in strength (i.e. match just the style, but not necessarily the strength), you can try this:

* Ensure all human likely moves are analyzed, as described in an earlier section.
* Choose a random move among all `moveInfos` with probability proportional to `humanPrior * exp(utility / 0.5)`. This will follow the humanPrior, but smoothly attenuate the probability of a move as it starts to lose more than 0.5 utility (about 25% winrate and/or some amount of score). Adjust the divisor 0.5 as desired.
* Optionally, also set `staticScoreUtilityFactor` to `0.5`. (significantly increase how much score affects the utility, compared to just winrate).
* A method like this, with adjusted numbers, might also be used to compensate for the gap that starts to open up in the human SL model no longer being able to match the strength of very top players at only 1 visit.

(Note: For GTP users, the parameter `humanSLChosenMovePiklLambda` does precisely this exp-based probability scaling.)

#### Heavily bias the search to anticipate human-like sequences rather than KataGo sequences.

Not many people have experimented with this yet, but in theory this could have very interesting effects! This will influence the winrates and scores that KataGo assigns to various moves to be much closer to the winrates and scores it would anticipate for various variations if they were played out the way it thinks human players of the configured profile might play them out, but still judging the endpoints of those variations using KataGo's own judgments.

* Set `humanSLProfile` and `ignorePreRootHistory` and `rootNumSymmetriesToSample` as desired.
* Set `humanSLPlaExploreProbWeightful` and `humanSLOppExploreProbWeightful` to `0.9` (spend about 90% of visits at every node using the human policy, in a weightful way that does bias KataGo's evaluations).
* Set `humanSLRootExploreProbWeightful` to `0.5` (at the root use about 50% of playouts to explore human moves, and the other 50% use KataGo's normal policy).
* Set `humanSLCpuctPermanent` to `1.0` or as otherwise desired (when using the human policy, attenuate the policy from too many visits to things that are on the order of 1.0 utility, or 50% winrate worse).
* Set `useUncertainty` to `false` and `subtreeValueBiasFactor` to `0.0` and `useNoisePruning` to `false` (important, disables a few search features that add strength but are highly likely to interfere with this kind of weightful biasing).

#### Bias the search to anticipate human-like sequences rather than KataGo sequences, but only for the opponent.

This kind of setting could be interesting for handicap games or trying to elicit trick plays and other kinds of opponent-aware tactics. Of course, experimentation and tuning may be needed for it to work well, and it might not work well, or might work "too" well and backfire.

* Set `humanSLProfile` and `ignorePreRootHistory` and `rootNumSymmetriesToSample` as desired.
   * This is also probably a really interesting place to experiment with the various `preaz_{BR}_{WR}` or `rank_{BR}_{WR}` settings with asymmetric ranks.
* Set `humanSLOppExploreProbWeightful` to `0.8` (spend about 80% of visits at every node using the human policy, in a weightful way that does bias KataGo's values, but only for the opponent!).
* Set `humanSLCpuctPermanent` to `0.5` or as desired (when using the human policy, do attenuate the policy from putting *too* many visits to things that are on the order of 0.5 utility, or 25% winrate worse).
* Set `playoutDoublingAdvantage` also as desired or as typical for handicap games.
* Set `useUncertainty` to `false` and `subtreeValueBiasFactor` to `0.0` and `useNoisePruning` to `false` (important, disables a few search features that add strength but are highly likely to interfere with this kind of weightful biasing).
   * Setting `useNoisePruning` to `false` is probably the most important of these - it adds the least strength in normal usage but might interfere the most. One could experiment with still enabling the other two for strength.

