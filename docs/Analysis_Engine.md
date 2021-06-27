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
   * `reportDuringSearchEvery (float)`: Optional. Specify a number of seconds such that while this position is being searched, KataGo will report the partial analysis every that many seconds.
   * `priority (int)`: Optional. Analysis threads will prefer handling queries with the highest priority unless already started on another task, breaking ties in favor of earlier queries. If not specified, defaults to 0.
   * `priorities (list of integers)`: Optional. When using analyzeTurns, you can use this instead of `priority` if you want a different priority per turn. Must be of same length as `analyzeTurns`, `priorities[0]` is the priority for `analyzeTurns[0]`, `priorities[1]` is the priority for `analyzeTurns[1]`, etc.

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
{"id":"foo","isDuringSearch":false,"moveInfos":[{"lcb":0.8740855166489953,"move":"Q5","order":0,"prior":0.8934692740440369,"pv":["Q5","R5","Q6","P4","O5","O4","R6","S5","N4","N5","N3"],"scoreLead":8.18535151076558,"scoreMean":8.18535151076558,"scoreSelfplay":10.414442461570038,"scoreStdev":23.987067985850913,"utility":0.7509536097709347,"utilityLcb":0.7717092488727239,"visits":495,"winrate":0.8666727883983563},{"lcb":1.936558574438095,"move":"D4","order":1,"prior":0.021620146930217743,"pv":["D4","Q5"],"scoreLead":12.300520420074463,"scoreMean":12.300520420074463,"scoreSelfplay":15.386500358581543,"scoreStdev":24.661467510313432,"utility":0.9287495791972984,"utilityLcb":2.8000000000000003,"visits":2,"winrate":0.9365585744380951},{"lcb":1.9393062554299831,"move":"Q16","order":2,"prior":0.006689758971333504,"pv":["Q16"],"scoreLead":12.97426986694336,"scoreMean":12.97426986694336,"scoreSelfplay":16.423904418945313,"scoreStdev":25.34494674587838,"utility":0.9410896213959669,"utilityLcb":2.8000000000000003,"visits":1,"winrate":0.9393062554299831},{"lcb":1.9348860532045364,"move":"D16","order":3,"prior":0.0064553022384643555,"pv":["D16"],"scoreLead":12.066888809204102,"scoreMean":12.066888809204102,"scoreSelfplay":15.591397285461426,"scoreStdev":25.65390196745236,"utility":0.9256971928661066,"utilityLcb":2.8000000000000003,"visits":1,"winrate":0.9348860532045364}],"rootInfo":{"currentPlayer":"B","lcb":0.8672585456293346,"scoreLead":8.219540952281882,"scoreSelfplay":10.456476293719811,"scoreStdev":23.99829921716391,"symHash":"1D25038E8FC8C26C456B8DF2DBF70C02","thisHash":"F8FAEDA0E0C89DDC5AA5CCBB5E7B859D","utility":0.7524437705003542,"visits":500,"winrate":0.8672585456293346},"turnNumber":2}
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

Current fields are:

   * `id`: The same id string that was provided on the query.
   * `isDuringSearch`: Normally false. If `reportDuringSearchEvery` is provided, then will be true on the reports during the middle of the search before the search is complete. Every position searched will still always conclude with exactly one final response when the search is completed where this field is false.
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
      * `lcb` - The [LCB](https://github.com/leela-zero/leela-zero/issues/2282) of the move's winrate. Has the same units as winrate, but might lie outside of [0,1] since the current implementation doesn't strictly account for the 0-1 bounds.
      * `utilityLcb` - The LCB of the move's utility.
      * `order` - KataGo's ranking of the move. 0 is the best, 1 is the next best, and so on.
      * `pv` - The principal variation following this move. May be of variable length or even empty.
      * `pvVisits` - The number of visits for each move in `pv`. Exists only if `includePVVisits` is true.
      * `ownership` - If `includeMovesOwnership` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from -1 to 1 indicating the predicted ownership after this move. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
   * `rootInfo`: A JSON dictionary with fields containing overall statistics for the requested turn itself calculated in the same way as they would be for the next moves. Current fields are: `winrate`, `scoreLead`, `scoreSelfplay`, `utility`, `visits`. And additional fields:
      * `thisHash` - A string that will with extremely high probability be unique for each distinct (board position, player to move, simple ko ban) combination.
      * `symHash` - Like `thisHash` except the string will be the same between positions that are symmetrically equivalent. Does NOT necessarily take into account superko.
      * `symmetriesUsed` - Present if KataGo is configured to avoid searching some moves due to symmetry (`rootSymmetryPruning=true`). If present, a list of integers from 0-7 indicating the symmetries used. Bit 0 is "flip y", bit 1 is "flip x", bit 2 is "swap x and y by reflecting along the diagonal from the upper-left to lower-right corner, e.g. A19 to T1", and these are applied in this order (so for example symmetry 5 is the transformation "flip y and THEN swap x and y"). If present, the list always includes symmetry 0.
      * `currentPlayer` - The current player whose possible move choices are being analyzed, `"B"` or `"W"`.
   * `ownership` - If `includeOwnership` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize` with values from -1 to 1 indicating the predicted ownership. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1).
   * `policy` - If `includePolicy` was true, then this field will be included. It is a JSON array of length `boardYSize * boardXSize + 1` with positive values summing to 1 indicating the neural network's prediction of the best move before any search, and `-1` indicating illegal moves. Values are in row-major order, starting at the top-left of the board (e.g. A19) and going to the bottom right (e.g. T1). The last value in the array is the policy value for passing.

#### Special Action Queries

Currently a few special action queries are supported that direct the analysis engine to do something other than enqueue a new position or set of positions for analysis.
A special action query is also sent as a JSON object, but with a different set of fields depending on the query.

##### query_version
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

##### clear_cache
Requests that KataGo empty its neural net cache. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `clear_cache`.

Example:
```
{"id":"foo","action":"clear_cache"}
```
The response to this query is to simply echo back a json object with exactly the same data and fields of the query. This response is sent after the cache is successfully cleared. If there are also any ongoing analysis queries at the time, those queries will of course be concurrently refilling the the cache even as the response is being sent.

Explanation: KataGo uses a cache of neural net query results to skip querying the neural net when it encounters within its search tree a position whose stone configuration, player to move, ko status, komi, rules, and other relevant options are all identical a position it has seen before. For example, this may happen if the search trees for some queries overlap due to being on nearby moves of the same game, or it may happen even within a single analysis query if the search explores differing orders of moves that lead to the same positions (often, about 20% of search tree nodes hit the cache due transposing to order of moves, although it may be vastly higher or lower depending on the position and search depth). Reasons for wanting to clear the cache may include:

* Freeing up RAM usage - emptying the cache should release the memory used for the results in the cache, which is typically the largest memory usage in KataGo. Memory usage will of course rise again as the cache refills.

* Testing or studying the variability of KataGo's search results for a given number of visits. Analyzing a position again after a cache clear will give a "fresh" look on that position that better matches the variety of possible results KataGo may return, simliar to if the analysis engine were entirely restarted. Each query will re-randomize the symmetry of the neural net used for that query instead of using the cached result, giving a new and more varied opinion.


##### terminate

Requests that KataGo terminate zero or more analysis queries without waiting for them to finish normally. When a query is terminated, the engine will make a best effort to halt their analysis as soon as possible, reporting the results of whatever number of visits were performed up to that point. Required fields:

   * `id (string)`: Required. An arbitrary string identifier for this query.
   * `action (string)`: Required. Should be the string `terminate`.
   * `terminateId (string)`: Required. Terminate queries that were submitted with this `id` field without analyzing or finishing analyzing them.
   * `turnNumbers (array of ints)`: Optional. If provided, restrict only to terminating the queries with that id that were for these turn numbers.

Example:
```
{"id":"bar","action":"terminate","terminateId":"foo","turnNumbers":[1,2]}
```

Responses to terminated queries may be missing their data fields if no analysis at all was performed before termination. In such a case, the only fields guaranteed to be on the response are `id` and `turnNumber`, and `isDuringSearch` (which will always be false), as well as one additional boolean field unique to terminated queries that did not analyze at all, `noResults` (which will always be true). Example:
```
{"id":"foo","isDuringSearch":false,"noResults":true,"turnNumber":2}
```

The terminate query itself will result in a response as well, to acknowledge receipt and processing of the action. The response consists of echoing a json object back with exactly the same fields and data of the query.

The response will NOT generally wait for all of the effects of the action to take place - it may take a small amount of additional time for ongoing searches to actually terminate and report their partial results. A client of this API that wants to wait for all terminated queries to finish should on its own track the set of queries that it has sent for analysis, and wait for all of them to have finished. This can be done by relying on the property that every analysis query, whether terminated or not, and regardless of `reportDuringSearchEvery`, will conclude with exactly one reply where `isDuringSearch` is `false` - such a reply can therefore be used as a marker that an analysis query has finished. (Except during shutdown of the engine if `-quit-without-waiting` was specified).



