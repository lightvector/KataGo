"""
Unit tests for katago.utils.training_data_generator.TrainingDataGenerator.

These check:
  1. The restart gap fix: a file deferred at a subepoch boundary (peeked but not popped) is NOT recorded as
     used, so a checkpoint+restart re-serves it instead of silently skipping it.
  2. The peek/pop and refill-cycling serving semantics in both modes.
  3. The gap-delaying reshuffle: when a new epoch reshuffles, a file may not recur within ~1/3 of the dataset
     of its previous occurrence, while everything is otherwise re-randomized.
  4. set_data_dir reconciliation of the persisted queue against the filesystem (filter-out + interleave-in).
  5. Backwards compatibility: a legacy set-typed data_files_used is converted to a (shuffled) list.

Note: there is deliberately no byte-for-byte equivalence oracle against the old PushBackGenerator logic; the
shuffle algorithm was intentionally changed to add the gap-delay, so the sequences no longer match the old one.
"""

import os
import json
import random

import pytest

from katago.utils.training_data_generator import TrainingDataGenerator


# ---- helpers ----------------------------------------------------------------------------------------

def make_data_dir(root, rows_each):
    """Create a fake train/ dir with one .npz (empty) + .json ({"num_rows": n}) per entry. Returns its path."""
    tdatadir = os.path.join(str(root), "train")
    os.makedirs(tdatadir, exist_ok=True)
    for i, num_rows in enumerate(rows_each):
        p = os.path.join(tdatadir, f"data{i}.npz")
        open(p, "w").close()
        with open(os.path.splitext(p)[0] + ".json", "w") as f:
            json.dump({"num_rows": num_rows}, f)
    return tdatadir


def all_npz(tdatadir):
    return [os.path.join(tdatadir, f) for f in os.listdir(tdatadir) if f.endswith(".npz")]


def fresh_state():
    return {"data_files_used": [], "rev_data_files_remaining": [], "old_train_data_dirs": []}


def num_batches_per_subepoch(samples_per_epoch, batch_size, sub_epochs):
    return int(round(samples_per_epoch / batch_size)) / sub_epochs


# ---- driver mirroring train.py get_files_for_subepoch (peek/pop) ------------------------------------

def subepoch_via_peek_pop(g, batch_size, samples_per_epoch, sub_epochs):
    nbatches_sub = num_batches_per_subepoch(samples_per_epoch, batch_size, sub_epochs)
    train_files_to_use = []
    batches_so_far = 0
    found_enough = False
    while True:
        filename = g.peek()
        if filename is None:
            break
        with open(os.path.splitext(filename)[0] + ".json") as f:
            nbatches = json.load(f)["num_rows"] // batch_size
        if nbatches <= 0:
            g.pop()
            continue
        if batches_so_far + nbatches > nbatches_sub:
            if batches_so_far > 0 and random.random() <= (batches_so_far + nbatches - nbatches_sub) / nbatches:
                found_enough = True
                break
        g.pop()
        train_files_to_use.append(filename)
        batches_so_far += nbatches
        if batches_so_far >= nbatches_sub or len(train_files_to_use) > 100000:
            found_enough = True
            break
    return train_files_to_use if found_enough else None


def serve_n(g, n):
    """Pop up to n files (stopping early on None), returning the list actually served."""
    out = []
    for _ in range(n):
        f = g.pop()
        if f is None:
            break
        out.append(f)
    return out


PARAMS = dict(batch_size=100, samples_per_epoch=4000, sub_epochs=2, num_draws=10)
ROWS = [1000, 1500, 800, 2000, 1200, 900, 1700, 1100]


# ---- basic serving semantics ------------------------------------------------------------------------

def test_norepeat_eventually_stops(tmp_path):
    """no_repeat_files serves each file once, then returns None forever, with no repeats or losses."""
    td = make_data_dir(tmp_path, [1000] * 6)
    g = TrainingDataGenerator(fresh_state(), True)
    g.set_data_dir_if_has_remaining_files(td)
    out = [subepoch_via_peek_pop(g, batch_size=100, samples_per_epoch=2000, sub_epochs=2) for _ in range(20)]
    assert out[-1] is None
    assert any(x is None for x in out)
    served = [f for sub in out if sub for f in sub]
    assert len(served) == len(set(served)) == 6


def test_norepeat_all_used_waits_and_serves_nothing(tmp_path):
    """no_repeat_files loading an all-already-used dir should decline (return False) and serve nothing."""
    td = make_data_dir(tmp_path, [1000] * 6)
    g = TrainingDataGenerator(
        {"data_files_used": list(all_npz(td)), "rev_data_files_remaining": [], "old_train_data_dirs": []}, True
    )
    assert g.set_data_dir_if_has_remaining_files(td) is False
    assert g.has_any_remaining_data() is False
    assert g.peek() is None
    assert g.pop() is None


def test_declined_dir_does_not_mutate_history(tmp_path):
    """When set_data_dir_if_has_remaining_files declines a dir, it must not record it in old_train_data_dirs
    nor prune data_files_used."""
    td = make_data_dir(tmp_path / "a", [1000] * 4)
    used = list(all_npz(td))
    dirs = ["/preexisting/dir"]
    ts = {"data_files_used": list(used), "rev_data_files_remaining": [], "old_train_data_dirs": list(dirs)}
    g = TrainingDataGenerator(ts, no_repeat_files=True)
    assert g.set_data_dir_if_has_remaining_files(td) is False
    assert ts["old_train_data_dirs"] == dirs        # not appended
    assert ts["data_files_used"] == used            # not pruned/altered
    assert g.has_any_files() is False               # nothing loaded

    # default mode with a truly empty dir: also declined, same no-mutation guarantee.
    empty = make_data_dir(tmp_path / "b", [])
    ts2 = {"data_files_used": list(used), "rev_data_files_remaining": [], "old_train_data_dirs": list(dirs)}
    g2 = TrainingDataGenerator(ts2, no_repeat_files=False)
    assert g2.set_data_dir_if_has_remaining_files(empty) is False
    assert ts2["old_train_data_dirs"] == dirs
    assert ts2["data_files_used"] == used


def test_loaded_dir_records_history(tmp_path):
    """The positive (load) path returns True and still performs the history bookkeeping."""
    td = make_data_dir(tmp_path, [1000] * 3)
    ts = fresh_state()
    g = TrainingDataGenerator(ts, no_repeat_files=False)
    assert g.set_data_dir_if_has_remaining_files(td) is True
    assert ts["old_train_data_dirs"] == [td]
    assert g.has_any_remaining_data() is True


def test_has_any_remaining_data_timing_independent(tmp_path):
    """has_any_remaining_data is valid at any call time, not just right after set_data_dir."""
    td = make_data_dir(tmp_path, [1000] * 3)

    g = TrainingDataGenerator(fresh_state(), no_repeat_files=True)
    g.set_data_dir_if_has_remaining_files(td)
    assert g.has_any_remaining_data() is True
    assert g.pop() is not None
    assert g.has_any_remaining_data() is True  # mid-stream, files still remain
    assert g.pop() is not None
    assert g.pop() is not None
    assert g.has_any_remaining_data() is False  # drained, not "started empty"
    assert g.peek() is None

    g2 = TrainingDataGenerator(fresh_state(), no_repeat_files=False)
    g2.set_data_dir_if_has_remaining_files(td)
    for _ in range(3):
        assert g2.pop() is not None
    # Pass drained, but default mode cycles, so there is still more data.
    assert g2.has_any_remaining_data() is True
    assert g2.peek() is not None


def test_peek_does_not_consume_or_mutate(tmp_path):
    """peek() must be idempotent and must not add anything to data_files_used."""
    td = make_data_dir(tmp_path, [1000] * 4)
    ts = fresh_state()
    g = TrainingDataGenerator(ts, True)
    g.set_data_dir_if_has_remaining_files(td)
    first = g.peek()
    assert first is not None
    assert g.peek() == first  # repeatable
    assert ts["data_files_used"] == []  # no mutation
    assert g.pop() == first  # pop returns the peeked file
    assert ts["data_files_used"] == [first]  # and only now is it marked used


def test_pop_records_usage_order(tmp_path):
    """data_files_used is a list recording consumption order, with no duplicates within an epoch."""
    td = make_data_dir(tmp_path, [1000] * 5)
    ts = fresh_state()
    g = TrainingDataGenerator(ts, True)
    g.set_data_dir_if_has_remaining_files(td)
    served = serve_n(g, 5)
    assert ts["data_files_used"] == served            # exact order, as a list
    assert len(served) == len(set(served)) == 5       # all five, no dups


# ---- queue persistence / reconciliation -------------------------------------------------------------

def test_queue_persisted_in_train_state(tmp_path):
    """The serve queue lives in train_state so it survives a checkpoint: a fresh generator over the same
    train_state continues the exact same order rather than reshuffling."""
    td = make_data_dir(tmp_path, [1000] * 8)
    ts = fresh_state()
    g = TrainingDataGenerator(ts, True)
    g.set_data_dir_if_has_remaining_files(td)
    g.pop(); g.pop()
    remaining_snapshot = list(ts["rev_data_files_remaining"])

    # "Restart": brand new generator, same (checkpointed) train_state, re-load the same dir.
    g2 = TrainingDataGenerator(ts, True)
    g2.set_data_dir_if_has_remaining_files(td)
    # Reconciliation against an unchanged filesystem with no new files must leave the queue order untouched.
    assert ts["rev_data_files_remaining"] == remaining_snapshot
    assert serve_n(g2, 6) == list(reversed(remaining_snapshot))


def test_reconcile_drops_vanished_files(tmp_path):
    """Files queued but no longer present on disk are filtered out of rev_data_files_remaining, order-preserved."""
    td = make_data_dir(tmp_path, [1000] * 6)
    ts = fresh_state()
    g = TrainingDataGenerator(ts, True)
    g.set_data_dir_if_has_remaining_files(td)
    queued = list(ts["rev_data_files_remaining"])

    # Delete two files that are still queued (pick the two at the front of the serve order = end of the list).
    gone = [queued[-1], queued[-2]]
    for f in gone:
        os.remove(f)
        os.remove(os.path.splitext(f)[0] + ".json")

    g2 = TrainingDataGenerator(ts, True)
    g2.set_data_dir_if_has_remaining_files(td)
    survivors = ts["rev_data_files_remaining"]
    assert set(survivors) == set(queued) - set(gone)
    # Order among survivors preserved (the filter is order-preserving).
    assert survivors == [f for f in queued if f not in set(gone)]
    assert all(f not in survivors for f in gone)


def test_reconcile_interleaves_new_files(tmp_path):
    """New, not-yet-used files appearing on disk get interleaved into the existing queue, preserving the
    queue's relative order among its retained files."""
    td = make_data_dir(tmp_path, [1000] * 4)
    ts = fresh_state()
    g = TrainingDataGenerator(ts, True)
    g.set_data_dir_if_has_remaining_files(td)
    old_queue = list(ts["rev_data_files_remaining"])

    # Add four brand-new files to the same dir, then re-load.
    for i in range(4, 8):
        p = os.path.join(td, f"data{i}.npz")
        open(p, "w").close()
        with open(os.path.splitext(p)[0] + ".json", "w") as fh:
            json.dump({"num_rows": 1000}, fh)
    new_files = set(all_npz(td)) - set(old_queue)
    assert len(new_files) == 4

    g2 = TrainingDataGenerator(ts, True)
    g2.set_data_dir_if_has_remaining_files(td)
    merged = ts["rev_data_files_remaining"]
    assert set(merged) == set(old_queue) | new_files          # everything present, once
    assert len(merged) == len(set(merged))
    # The retained old-queue files keep their relative order within the merged queue.
    assert [f for f in merged if f in set(old_queue)] == old_queue


def test_fresh_dir_with_empty_queue_shuffles(tmp_path):
    """On a brand-new dir with an empty persisted queue, the interleave-into-empty path must still randomize
    (not serve in os.listdir order)."""
    td = make_data_dir(tmp_path, [1000] * 50)
    listdir_order = all_npz(td)

    saw_non_listdir_order = False
    for seed in range(10):
        random.seed(seed)
        ts = fresh_state()
        g = TrainingDataGenerator(ts, True)
        g.set_data_dir_if_has_remaining_files(td)
        served = serve_n(g, 50)
        assert set(served) == set(listdir_order)  # same files
        if served != listdir_order:
            saw_non_listdir_order = True
    assert saw_non_listdir_order  # at least one seed reorders; we are not just echoing listdir order


# ---- gap-delaying reshuffle -------------------------------------------------------------------------

def drain_epoch_order(g, n):
    """Pop a full epoch of n files (default mode), returning them in serve order. Stops if the generator
    wipes data_files_used (i.e. it cycled into the next epoch) after n pops."""
    return serve_n(g, n)


def min_gap_floor(n):
    """The hard minimum gap guaranteed by the reshuffle for a dataset of n files: n - k0 + 1."""
    k0 = (n * 2 + 1) // 3
    return n - k0 + 1


@pytest.mark.parametrize("seed", range(25))
def test_reshuffle_respects_min_gap(tmp_path, seed):
    """Across the epoch1->epoch2 boundary, no file recurs within the min-gap floor of its previous occurrence.

    We measure the gap as the distance between a file's index in the concatenated (epoch1 ++ epoch2) serve
    order. The floor is n - k0 + 1.
    """
    random.seed(seed)
    n = 12
    td = make_data_dir(tmp_path, [1000] * n)
    g = TrainingDataGenerator(fresh_state(), no_repeat_files=False)
    g.set_data_dir_if_has_remaining_files(td)

    epoch1 = drain_epoch_order(g, n)   # consumes exactly one epoch; refill happens on the next peek/pop
    epoch2 = drain_epoch_order(g, n)
    assert sorted(epoch1) == sorted(epoch2) == sorted(all_npz(td))  # each epoch is a full permutation

    combined = epoch1 + epoch2
    floor = min_gap_floor(n)
    last_seen = {}
    for pos, f in enumerate(combined):
        if f in last_seen:
            assert pos - last_seen[f] >= floor, (
                f"file {os.path.basename(f)} recurred at gap {pos - last_seen[f]} < floor {floor} (seed {seed})"
            )
        last_seen[f] = pos


def test_reshuffle_tail_third_held_out_of_epoch_start(tmp_path):
    """The last 1/3 of the previous epoch's order must not appear in the first (n - k0) positions of the new
    epoch (equivalently: those positions are drawn only from new files + the first k0 of the previous order)."""
    n = 15
    k0 = (n * 2 + 1) // 3
    td = make_data_dir(tmp_path, [1000] * n)

    for seed in range(25):
        random.seed(seed)
        g = TrainingDataGenerator(fresh_state(), no_repeat_files=False)
        g.set_data_dir_if_has_remaining_files(td)
        epoch1 = drain_epoch_order(g, n)
        epoch2 = drain_epoch_order(g, n)

        tail_third = set(epoch1[k0:])
        early_new = set(epoch2[: n - k0])
        # P[i] can first appear at new-epoch position i-k0+1, so the earliest a tail file (i>=k0) can show up
        # is position 1; but the strongest clean statement is that the count of tail-third files in the first
        # (n-k0) positions is bounded by how many have "unlocked" - assert none of P[k0:] appears before its
        # unlock index.
        pos_in_epoch2 = {f: i for i, f in enumerate(epoch2)}
        for i in range(k0, n):
            f = epoch1[i]
            earliest = i - k0 + 1
            assert pos_in_epoch2[f] >= earliest, (
                f"prev-epoch file at index {i} appeared at new-epoch pos {pos_in_epoch2[f]} < {earliest} (seed {seed})"
            )
        # And the very first new-epoch slot can never be a tail-third file (its unlock index is >= 1).
        assert epoch2[0] not in tail_third


def test_reshuffle_rerandomizes(tmp_path):
    """The reshuffle is not the identity: successive epochs are not always in the same order."""
    n = 12
    td = make_data_dir(tmp_path, [1000] * n)
    random.seed(0)
    g = TrainingDataGenerator(fresh_state(), no_repeat_files=False)
    g.set_data_dir_if_has_remaining_files(td)
    epochs = [drain_epoch_order(g, n) for _ in range(4)]
    # No two consecutive epochs identical, and not all four identical.
    assert any(epochs[i] != epochs[i + 1] for i in range(len(epochs) - 1))
    assert len({tuple(e) for e in epochs}) > 1


def test_reshuffle_new_files_eligible_immediately(tmp_path):
    """New files (present this epoch, absent from the previous-epoch order) can land anywhere, including the
    very first position of the new epoch."""
    # Start with a dataset, run one epoch so data_files_used is the full previous order.
    n = 8
    td = make_data_dir(tmp_path, [1000] * n)

    saw_new_first = False
    for seed in range(40):
        random.seed(seed)
        ts = fresh_state()
        g = TrainingDataGenerator(ts, no_repeat_files=False)
        g.set_data_dir_if_has_remaining_files(td)
        drain_epoch_order(g, n)  # epoch1: establishes previous-epoch order in data_files_used

        # Add new files, re-load (reconcile interleaves them into the still-draining/empty queue), then force
        # a reshuffle by draining to the boundary.
        for i in range(n, n + 3):
            p = os.path.join(td, f"data{i}.npz")
            open(p, "w").close()
            with open(os.path.splitext(p)[0] + ".json", "w") as fh:
                json.dump({"num_rows": 1000}, fh)
        prev_order = list(ts["data_files_used"])
        new_files = set(all_npz(td)) - set(prev_order)

        g.set_data_dir_if_has_remaining_files(td)
        # Drain whatever epoch0 remainder is queued, then the reshuffled epoch begins.
        epoch2 = serve_n(g, len(all_npz(td)) * 2)
        # Find the first reshuffled-epoch position: the reshuffle wipes data_files_used, but here we just check
        # that across the served files a new file appears at least once very early across seeds.
        if epoch2 and epoch2[0] in new_files:
            saw_new_first = True
    assert saw_new_first


# ---- gap fix (deferred file survives restart) -------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_gap_fix_deferred_file_survives_restart(tmp_path, seed):
    """Core bug fix: a file deferred (peeked, not popped) at a subepoch boundary must not be marked used, so a
    checkpoint+restart re-serves it rather than skipping it forever."""
    td = make_data_dir(tmp_path, [600] * 8)

    # First process: draw exactly one subepoch (which ends by deferring the next file).
    ts = fresh_state()
    random.seed(seed)
    g = TrainingDataGenerator(ts, False)
    g.set_data_dir_if_has_remaining_files(td)
    subepoch_via_peek_pop(g, batch_size=100, samples_per_epoch=2000, sub_epochs=2)
    deferred = g.peek()  # the file the first process would serve next
    used_snapshot = list(ts["data_files_used"])  # the "checkpoint"
    remaining_snapshot = list(ts["rev_data_files_remaining"])

    # Only meaningful when the deferred file is one not already consumed this pass.
    assert deferred is not None
    assert deferred not in used_snapshot

    # Restart: fresh generator/process, only train_state persisted.
    ts2 = {
        "data_files_used": list(used_snapshot),
        "rev_data_files_remaining": list(remaining_snapshot),
        "old_train_data_dirs": list(ts["old_train_data_dirs"]),
    }
    random.seed(seed + 99999)
    g2 = TrainingDataGenerator(ts2, False)
    g2.set_data_dir_if_has_remaining_files(td)

    # Walk the resumed first pass; the deferred file must appear (not be silently skipped).
    served, seen = [], set()
    while True:
        f = g2.peek()
        if f is None or f in seen:
            break
        served.append(f)
        seen.add(f)
        g2.pop()
        if len(ts2["data_files_used"]) == 0:  # default mode wiped + reseeded => left the first pass
            break
    assert deferred in served


# ---- backwards compatibility ------------------------------------------------------------------------

def test_legacy_set_converted_to_list():
    """A checkpoint predating the list change stores data_files_used as a set; __init__ converts it to a list."""
    used = {"/d/a.npz", "/d/b.npz", "/d/c.npz"}
    ts = {"data_files_used": set(used), "old_train_data_dirs": ["/d"]}
    TrainingDataGenerator(ts, no_repeat_files=False)
    assert isinstance(ts["data_files_used"], list)
    assert set(ts["data_files_used"]) == used          # same contents
    assert ts.get("rev_data_files_remaining") == []    # defaulted in


def test_init_defaults_train_state_fields():
    """The generator defaults its train_state fields when absent and leaves already-present values untouched."""
    ts = {}
    TrainingDataGenerator(ts, no_repeat_files=False)
    assert ts["data_files_used"] == []
    assert ts["rev_data_files_remaining"] == []
    assert ts["old_train_data_dirs"] == []

    used = ["/some/dir/a.npz"]
    remaining = ["/some/dir/b.npz"]
    dirs = ["/some/dir"]
    ts2 = {"data_files_used": used, "rev_data_files_remaining": remaining, "old_train_data_dirs": dirs}
    TrainingDataGenerator(ts2, no_repeat_files=False)
    assert ts2["data_files_used"] is used
    assert ts2["rev_data_files_remaining"] is remaining
    assert ts2["old_train_data_dirs"] is dirs
