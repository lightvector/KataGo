"""
Unit tests for katago.utils.training_data_generator.TrainingDataGenerator.

These check two things:
  1. Equivalence: the new peek/pop generator produces byte-identical file sequences to the previous
     train_files_gen + PushBackGenerator logic (reproduced here as the oracle), in both modes.
  2. The restart gap fix: a file deferred at a subepoch boundary (peeked but not popped) is NOT recorded as
     used, so a checkpoint+restart re-serves it instead of silently skipping it.
"""

import os
import json
import random

import pytest

from katago.utils.training_data_generator import TrainingDataGenerator
from katago.utils.push_back_generator import PushBackGenerator


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


def fresh_state():
    return {"data_files_used": set(), "old_train_data_dirs": []}


def num_batches_per_subepoch(samples_per_epoch, batch_size, sub_epochs):
    return int(round(samples_per_epoch / batch_size)) / sub_epochs


# ---- oracle: exact copy of the OLD train.py file-selection logic -----------------------------------

def reference_run(tdatadir, no_repeat_files, train_state, batch_size, samples_per_epoch, sub_epochs, num_draws, seed):
    random.seed(seed)
    train_files = [os.path.join(tdatadir, f) for f in os.listdir(tdatadir) if f.endswith(".npz")]
    epoch0_train_files = [p for p in train_files if p not in train_state["data_files_used"]]
    if tdatadir not in train_state["old_train_data_dirs"]:
        train_state["old_train_data_dirs"].append(tdatadir)

    def train_files_gen():
        train_files_shuffled = epoch0_train_files.copy()
        while True:
            random.shuffle(train_files_shuffled)
            for filename in train_files_shuffled:
                train_state["data_files_used"].add(filename)
                yield filename
            if no_repeat_files:
                break
            else:
                train_files_shuffled = train_files.copy()
                train_state["data_files_used"] = set()

    gen = PushBackGenerator(train_files_gen())
    nbatches_sub = num_batches_per_subepoch(samples_per_epoch, batch_size, sub_epochs)

    def get_files_for_subepoch():
        train_files_to_use = []
        batches_so_far = 0
        found_enough = False
        for filename in gen:
            with open(os.path.splitext(filename)[0] + ".json") as f:
                nbatches = json.load(f)["num_rows"] // batch_size
            if nbatches <= 0:
                continue
            if batches_so_far + nbatches > nbatches_sub:
                if batches_so_far > 0 and random.random() <= (batches_so_far + nbatches - nbatches_sub) / nbatches:
                    gen.push_back(filename)
                    found_enough = True
                    break
            train_files_to_use.append(filename)
            batches_so_far += nbatches
            if batches_so_far >= nbatches_sub or len(train_files_to_use) > 100000:
                found_enough = True
                break
        return train_files_to_use if found_enough else None

    return [get_files_for_subepoch() for _ in range(num_draws)]


# ---- driver mirroring the NEW train.py get_files_for_subepoch (peek/pop) ----------------------------

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


def new_run(tdatadir, no_repeat_files, train_state, batch_size, samples_per_epoch, sub_epochs, num_draws, seed):
    random.seed(seed)
    g = TrainingDataGenerator(train_state, no_repeat_files)
    g.set_data_dir_if_has_remaining_files(tdatadir)
    out = [subepoch_via_peek_pop(g, batch_size, samples_per_epoch, sub_epochs) for _ in range(num_draws)]
    return out, g


PARAMS = dict(batch_size=100, samples_per_epoch=4000, sub_epochs=2, num_draws=10)
ROWS = [1000, 1500, 800, 2000, 1200, 900, 1700, 1100]


# ---- tests ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_equivalence_default(tmp_path, seed):
    """Default (cycling) mode: new sequence must exactly match the old generator's, including wrap-around."""
    td = make_data_dir(tmp_path, ROWS)
    ref = reference_run(td, False, fresh_state(), seed=seed, **PARAMS)
    new, _ = new_run(td, False, fresh_state(), seed=seed, **PARAMS)
    assert ref == new


@pytest.mark.parametrize("seed", range(10))
def test_equivalence_norepeat(tmp_path, seed):
    """no_repeat_files mode: new sequence must exactly match the old generator's."""
    td = make_data_dir(tmp_path, ROWS)
    ref = reference_run(td, True, fresh_state(), seed=seed, **PARAMS)
    new, _ = new_run(td, True, fresh_state(), seed=seed, **PARAMS)
    assert ref == new


@pytest.mark.parametrize("seed", range(10))
def test_equivalence_default_resume_all_used(tmp_path, seed):
    """Default mode resuming with every file already used (empty epoch0): must wipe + cycle, still matching."""
    td = make_data_dir(tmp_path, ROWS)
    allfiles = [os.path.join(td, f) for f in os.listdir(td) if f.endswith(".npz")]
    ref = reference_run(td, False, {"data_files_used": set(allfiles), "old_train_data_dirs": []}, seed=seed, **PARAMS)
    new, _ = new_run(td, False, {"data_files_used": set(allfiles), "old_train_data_dirs": []}, seed=seed, **PARAMS)
    assert ref == new


def test_norepeat_eventually_stops(tmp_path):
    """no_repeat_files serves each file once, then returns None forever, with no repeats or losses."""
    td = make_data_dir(tmp_path, [1000] * 6)
    out, _ = new_run(td, True, fresh_state(), batch_size=100, samples_per_epoch=2000, sub_epochs=2, num_draws=20, seed=1)
    assert out[-1] is None
    assert any(x is None for x in out)
    served = [f for sub in out if sub for f in sub]
    assert len(served) == len(set(served)) == 6


def test_norepeat_all_used_waits_and_serves_nothing(tmp_path):
    """no_repeat_files adopting an all-already-used dir should decline (return False) and serve nothing."""
    td = make_data_dir(tmp_path, [1000] * 6)
    allfiles = [os.path.join(td, f) for f in os.listdir(td) if f.endswith(".npz")]
    g = TrainingDataGenerator({"data_files_used": set(allfiles), "old_train_data_dirs": []}, True)
    assert g.set_data_dir_if_has_remaining_files(td) is False
    assert g.has_any_remaining_data() is False
    assert g.peek() is None
    assert g.pop() is None


def test_declined_dir_does_not_mutate_history(tmp_path):
    """When set_data_dir_if_has_remaining_files declines a dir, it must not record it in old_train_data_dirs
    nor prune data_files_used. This is the property that preserves the old no-data-guard ordering: a dir we
    end up waiting on / quitting on leaves the used-file history exactly as it was.
    """
    # no_repeat with an all-already-used dir: declined.
    td = make_data_dir(tmp_path / "a", [1000] * 4)
    allfiles = [os.path.join(td, f) for f in os.listdir(td) if f.endswith(".npz")]
    used = set(allfiles)
    dirs = ["/preexisting/dir"]
    ts = {"data_files_used": set(used), "old_train_data_dirs": list(dirs)}
    g = TrainingDataGenerator(ts, no_repeat_files=True)
    assert g.set_data_dir_if_has_remaining_files(td) is False
    assert ts["old_train_data_dirs"] == dirs        # not appended
    assert ts["data_files_used"] == used            # not pruned/altered
    assert g.has_any_files() is False               # nothing adopted

    # default mode with a truly empty dir: also declined, same no-mutation guarantee.
    empty = make_data_dir(tmp_path / "b", [])
    ts2 = {"data_files_used": set(used), "old_train_data_dirs": list(dirs)}
    g2 = TrainingDataGenerator(ts2, no_repeat_files=False)
    assert g2.set_data_dir_if_has_remaining_files(empty) is False
    assert ts2["old_train_data_dirs"] == dirs
    assert ts2["data_files_used"] == used


def test_adopted_dir_records_history(tmp_path):
    """The positive (adopt) path returns True and still performs the history bookkeeping."""
    td = make_data_dir(tmp_path, [1000] * 3)
    ts = fresh_state()
    g = TrainingDataGenerator(ts, no_repeat_files=False)
    assert g.set_data_dir_if_has_remaining_files(td) is True
    assert ts["old_train_data_dirs"] == [td]
    assert g.has_any_remaining_data() is True


def test_has_any_remaining_data_timing_independent(tmp_path):
    """has_any_remaining_data is valid at any call time, not just right after set_data_dir.

    no_repeat_files: True while the epoch0 pass has files, False once it drains (terminal).
    default mode: stays True across a drained pass, because the pass is refilled by cycling.
    """
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


def test_init_defaults_train_state_fields():
    """The generator owns initialization of its two train_state fields: it defaults them when absent (fresh run
    or a checkpoint predating them) and leaves already-present (restored) values untouched."""
    # Absent on a fresh/old state: both get created.
    ts = {}
    TrainingDataGenerator(ts, no_repeat_files=False)
    assert ts["data_files_used"] == set()
    assert ts["old_train_data_dirs"] == []

    # Present on a resume: existing values are not clobbered.
    used = {"/some/dir/a.npz"}
    dirs = ["/some/dir"]
    ts2 = {"data_files_used": used, "old_train_data_dirs": dirs}
    TrainingDataGenerator(ts2, no_repeat_files=False)
    assert ts2["data_files_used"] is used
    assert ts2["old_train_data_dirs"] is dirs


def test_peek_does_not_consume_or_mutate(tmp_path):
    """peek() must be idempotent and must not add anything to data_files_used."""
    td = make_data_dir(tmp_path, [1000] * 4)
    ts = fresh_state()
    g = TrainingDataGenerator(ts, True)
    g.set_data_dir_if_has_remaining_files(td)
    first = g.peek()
    assert first is not None
    assert g.peek() == first  # repeatable
    assert ts["data_files_used"] == set()  # no mutation
    assert g.pop() == first  # pop returns the peeked file
    assert first in ts["data_files_used"]  # and only now is it marked used


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
    used_snapshot = set(ts["data_files_used"])  # the "checkpoint"

    # Only meaningful when the deferred file is one not already consumed this pass.
    assert deferred is not None
    assert deferred not in used_snapshot

    # Restart: fresh generator/process, only train_state persisted.
    ts2 = {"data_files_used": set(used_snapshot), "old_train_data_dirs": list(ts["old_train_data_dirs"])}
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
