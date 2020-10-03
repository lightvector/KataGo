## KataGo Parallel Analysis Engine

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

### Protocol

The engine accepts queries on stdin, and output results on stdout. Every query and every result should be a single line.
The protocol is entirely asynchronous - new requests on stdin can be accepted at any time, and results will appear on stdout
whenever those analyses finish, and possibly in a different order than the requests were provided. As described below, each query
may specify *multiple* positions to be analyzed and therefore may generate *multiple* results.

If stdin is closed, then the engine will finish the analysis of all queued queries before exiting, unless `-quit-without-waiting` was
provided on the initial command line, in which case it will attempt to stop all threads and still exit cleanly but without
necessarily finishing the analysis of whatever queries are open at the time.

#### Queries

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
     * `player` should be `"B"` or `"W"`.
     * `location` should a string like `"C4"` the same as in the [GTP protocol](http://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html#SECTION000311000000000000000). KataGo also supports extended column coordinates locations beyond `"Z"`, such as `"AA"`, `"AB"`, `"AC"`, ... Alternatively one can also specify strings like `"(0,13)"` that explicitly give the integer X and Y coordinates.
   * `initialStones (list of [player string, location string] tuples)`: Optional. Specifies stones already on the board at the start of the game. For example, these could be handicap stones. Or, you could use this to specify a midgame position or whole-board tsumego that does not have a move history.
   * `initialPlayer (player string)`: Optional. Specifies the player to use for analyzing the first turn of the game, which can be useful if there `moves` is an empty list.
   * `rules (string or JSON)`: Required. Specify the rules for the game using either a shorthand string or a full JSON object.
     * See the documentation of `kata-get-rules` and `kata-set-rules` in [GTP Extensions](./GTP_Extensions.md) for a description of supported rules.
     * Some older neural net versions of KataGo do not support some rules options. If this is the case, then a warning will be issued and the rules will
       automatically be converted to the nearest rules that the neural net does support.
   * `komi (integer or half-integer)`: Optional but HIGHLY recommended. Specify the komi for the game. If not specified, KataGo will guess a default value, generally 7.5 for area scoring, but 6.5 if using territory scoring, and 7.0 if area scoring with a button. Values of komi outside of [-150,150] are not supported.
   * `whiteHandicapBonus (0|N|N-1)`: Optional. See `kata-get-rules` in [GTP Extensions](./GTP_Extensions.md) for what these mean. Can be used to override the handling of handicap bonus, taking precedence over `rules`. E.g. if you want `chinese` rules but with different compensation for handicap stones than Chinese rules normally use. You could also always specify this as 0 and do any adjustment you like on your own, by reporting an appropriate `komi`.
   * `boardXSize (integer)`: Required. The width of the board. Sizes > 19 are NOT supported unless KataGo has been compiled to support them (cpp/game/board.h, MAX_LEN = 19). KataGo's official neural nets have also not been trained for larger boards, but should work fine for mildly larger sizes (21,23,25).
   * `boardYSize (integer)`: Required. The height of the board. Sizes > 19 are NOT supported unless KataGo has been compiled to support them (cpp/game/board.h, MAX_LEN = 19). KataGo's official neural nets have also not been trained for larger boards, but should work fine for mildly larger sizes (21,23,25).
   * `analyzeTurns (list of integers)`: Optional. Which turns of the game to analyze. If this field is not specified, defaults to analyzing the last turn only.
   * `maxVisits (integer)`: Optional. The maximum number of visits to use. If not specified, defaults to the value in the analysis config file. If specified, overrides it.
   * `rootPolicyTemperature (float)`: Optional. Set this to a value > 1 to make KataGo do a wider search.
   * `rootFpuReductionMax (float)`: Optional. Set this to 0 to make KataGo more willing to try a variety of moves.
   * `includeOwnership (boolean)`: Optional. If true, report ownership prediction as a result. Will double memory usage and reduce performance slightly.
   * `includeMovesOwnership (boolean)`: Optional. If true, report ownership prediction for every individual move too.
   * `includePolicy (boolean)`: Optional. If true, report neural network raw policy as a result. Will not signficiantly affect performance.
   * `includePVVisits (boolean)`: Optional. If true, report the number of visits for each move in any reported pv.
   * `avoidMoves (list of dicts)`: Optional. Prohibit the search from exploring the specified moves for the specified player, until a certain number of ply deep in the search. Each dict must contain these fields:
      * `player` - the player to prohibit, `"B"` or `"W"`.
      * `moves` - an array of move locations to prohibit, such as `["C3","Q4","pass"]`
      * `untilDepth` - a positive integer, indicating the ply such that moves are prohibited before that ply.
      * Multiple dicts can specify different `untilDepth` for different sets of moves. The behavior is unspecified if a move is specified more than once with different `untilDepth`.
   * `allowMoves (list of dicts)`: Optional. Same as `avoidMoves` except prohibits all moves EXCEPT the moves specified. Currently, the list of dicts must also be length 1.
   * `overrideSettings (object)`: Optional. Specify any number of `paramName:value` entries in this object to override those params from command line `CONFIG_FILE` for this query. Most search parameters can be overriden: `cpuctExploration`, `winLossUtilityFactor`, etc.
   * `priority (int)`: Optional. Analysis threads will prefer handling queries with the highest priority unless already started on another task, breaking ties in favor of earlier queries. If not specified, defaults to 0.

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
```json
{"id":"foo","moveInfos":[{"lcb":0.8740855166489953,"move":"Q5","order":0,"prior":0.8934692740440369,"pv":["Q5","R5","Q6","P4","O5","O4","R6","S5","N4","N5","N3"],"scoreLead":8.18535151076558,"scoreMean":8.18535151076558,"scoreSelfplay":10.414442461570038,"scoreStdev":23.987067985850913,"utility":0.7509536097709347,"utilityLcb":0.7717092488727239,"visits":495,"winrate":0.8666727883983563},{"lcb":1.936558574438095,"move":"D4","order":1,"prior":0.021620146930217743,"pv":["D4","Q5"],"scoreLead":12.300520420074463,"scoreMean":12.300520420074463,"scoreSelfplay":15.386500358581543,"scoreStdev":24.661467510313432,"utility":0.9287495791972984,"utilityLcb":2.8000000000000003,"visits":2,"winrate":0.9365585744380951},{"lcb":1.9393062554299831,"move":"Q16","order":2,"prior":0.006689758971333504,"pv":["Q16"],"scoreLead":12.97426986694336,"scoreMean":12.97426986694336,"scoreSelfplay":16.423904418945313,"scoreStdev":25.34494674587838,"utility":0.9410896213959669,"utilityLcb":2.8000000000000003,"visits":1,"winrate":0.9393062554299831},{"lcb":1.9348860532045364,"move":"D16","order":3,"prior":0.0064553022384643555,"pv":["D16"],"scoreLead":12.066888809204102,"scoreMean":12.066888809204102,"scoreSelfplay":15.591397285461426,"scoreStdev":25.65390196745236,"utility":0.9256971928661066,"utilityLcb":2.8000000000000003,"visits":1,"winrate":0.9348860532045364}],"rootInfo":{"lcb":0.8672585456293346,"scoreLead":8.219540952281882,"scoreSelfplay":10.456476293719811,"scoreStdev":23.99829921716391,"utility":0.7524437705003542,"visits":500,"winrate":0.8672585456293346},"turnNumber":2}
```
<details>
<summary>
See formatted response.
</summary>

```json
{
    "id": "foo",
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
        "winrate": 0.9348860532045364
    }],
    "rootInfo": {
        "lcb": 0.8672585456293346,
        "scoreLead": 8.219540952281882,
        "scoreSelfplay": 10.456476293719811,
        "scoreStdev": 23.99829921716391,
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

Current fields are:

   * `id`: The same id string that was provided on the query.
   * `turnNumber`: The turn number being analyzed.
   * `moveInfos`: A list of JSON dictionaries, one per move that KataGo considered, with fields indicating the results of analysis. Current fields are:
      * `move` - The move being analyzed.
      * `visits` - The number of visits invested into the move.
      * `winrate` - The winrate of the move, as a float in [0,1].
      * `scoreMean` - Same as scoreLead. "Mean" is a slight misnomer, but this field exists to preserve compatibility with existing tools.
      * `scoreStdev` - The predicted standard deviation of the final score of the game after this move, in points. (NOTE: due to the mechanics of MCTS, this value will be **significantly biased high** currently, although it can still be informative as *relative* indicator).
      * `scoreLead` - The predicted average number of points that the current side is leading by (with this many points fewer, it would be an even game).
      * `scoreSelfplay` - The predicted average value of the final score of the game after this move during selfplay, in points. (NOTE: users should usually prefer scoreLead, since scoreSelfplay may be biased by the fact that KataGo isn't perfectly score-maximizing).
      * `prior` - The policy prior of the move, as a float in [0,1].
      * `utility` - The utility of the move, combining both winrate and score, as a float in [-C,C] where C is the maximum possible utility.
      * `lcb` - The [LCB](https://github.com/leela-zero/leela-zero/issues/2282) of the move's winrate, as a float in [0,1].
      * `utilityLcb` - The LCB of the move's utility.
      * `order` - KataGo's ranking of the move. 0 is the best, 1 is the next best, and so on.
      * `pv` - The principal variation following this move. May be of variable length or even empty.
      * `pvVisits` - The number of visits for each move in `pv`. Exists only if `includePVVisits` is true.
      * `ownership` - If `includeMovesOwnership` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from -1 to 1 indicating the predicted ownership after this move. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
   * `rootInfo`: A JSON dictionary with fields containing overall statistics for the requested turn itself calculated in the same way as they would be for the next moves. Current fields are: `winrate`, `scoreLead`, `scoreSelfplay`, `utility`, `visits`.
   * `ownership` - If `includeOwnership` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from -1 to 1 indicating the predicted ownership. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
   * `policy` - If `includePolicy` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize + 1` with positive values summing to 1 indicating the neural network's prediction of the best move before any search, and `-1` indicating illegal moves. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1). The last value in the array is the policy value for passing.
