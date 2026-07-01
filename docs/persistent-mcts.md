# Persistent MCTS model

This note records the model implemented for persistent root switching, export, and import.

## State

Let game states be keyed by collision-free graph hashes. A node can be used as a root at different
times, so every persistent primitive contribution is tagged by the root hash that generated it.
The implementation stores only primitive facts:

- root-tagged direct NN/terminal leaf statistics for a node;
- root-tagged counted edge visits for an edge;
- pending inherited visit counts keyed by possible future roots.

Aggregated `NodeStats` are a cache. They are rebuilt by post-order materialization from the
currently visible primitive facts.

For a current root `R`, the visible primitive facts are exactly the facts tagged by `R`. Ancestor
information is not read directly. Instead, when a real playout rooted at `A` traverses a descendant
state `D`, the implementation increments `pending[D]`. When `D` later becomes the current root,
those pending visits are consumed by running `D`'s own ordinary MCTS transition and writing
`D`-tagged primitive facts. Pending-consumption playouts do not create more pending visits; the
original ancestor playout already credited every descendant it actually traversed.

Therefore each root `R` has a canonical root-local state

```text
S_R = ordinary_mcts(root = R, samples = own(R) + inherited(R))
```

where `own(R)` is the number of real searches rooted at `R`, and `inherited(R)` is the number of
real ancestor-root playouts whose path traversed `R`.

## Isolation theorem

For any root-switch sequence and any two states `X` and `Y`, searches rooted inside the strict
subtree of `Y` cannot affect the materialized MCTS result at `Y`.

Proof: a real search rooted at `X` writes primitive facts only with tag `X`, and it increments
pending only for states traversed by that real playout. The materialized view at `Y` reads only
`Y`-tagged facts. If `X` is a strict descendant of `Y`, then `X != Y`, so `X`-tagged direct stats
and edge visits are invisible at `Y`. The playout rooted at `X` also never traverses its strict
ancestor `Y`, so it cannot increment `pending[Y]`. Thus neither the visible primitives nor the
pending count of `Y` can change.

## Inheritance theorem

For any state `C`, every real playout rooted at an ancestor of `C` that traverses `C` contributes
exactly one useful sample to `C`'s canonical MCTS state.

Proof: when the ancestor playout traverses `C`, `addPersistentDescendantCredit` increments
`pending[C]` once. No descendant-root contribution is read while the ancestor is selecting moves,
because only the ancestor's tag is visible during that search. When `C` is later materialized,
`consumePersistentPendingVisits` removes those pending visits and runs exactly that many playouts
from root `C`, with descendant credit propagation disabled. Those playouts write only `C`-tagged
primitive facts, so they are visible when `C` is current and invisible to ancestors. Since they use
the same transition function as ordinary MCTS from `C`, `S_C` is exactly ordinary MCTS with
`own(C) + inherited(C)` samples.

The statement is independent of switch order. A sequence such as `A, B, A, E, F, B, G, A, B`
only changes the counters `own(R)` and `inherited(R)` for each root `R`; materializing `R` depends
on those counters and `R`-tagged primitives, not on the previous root.

## Why primitive facts are serialized

KataGo parent stats are not linear deltas. They are recomputed from child edge visits, child
weights, value weighting, pruning, and direct NN values. Storing only aggregate `NodeStats` would
not be proof-preserving after root switches. The persistent layer serializes tagged direct stats,
tagged edge visits, NN outputs, root copies, graph-table nodes, the current root position/history,
and pending inherited visits. Import reconstructs the graph and materializes the current root from
those primitives.

## Implementation invariants

- Root copies are separate from graph-table nodes because KataGo normally keeps the root outside
  the graph table. Direct stats and edge visits are mirrored into an existing root copy when an
  ancestor search reaches the matching graph-table node.
- When a persistent search touches an already-expanded transposition node, it first ensures the
  current root has a direct NN contribution for that node. This prevents import from rebuilding a
  different state than the in-memory search.
- At the end of a persistent `runWholeSearch`, the current root subtree is materialized post-order.
  This flushes graph-search parent caches so exported/imported state and reported root values agree.
- Root move filtering is non-destructive in persistent mode. Disallowed root children are
  materialized with zero visible edge visits for the current root, while their tagged facts remain
  available for future root switches.
