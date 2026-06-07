import os
import random
import logging
from typing import Dict, List, Optional, Any


class TrainingDataGenerator:
    """
    Selects which shuffled training data (.npz) files to train on, in order.

    This class holds a shared reference to the run's train_state dict, MUTATING IT IN PLACE.
    The contract is that train_state always reflects the correct serving state, so that a checkpoint
    taken of train_state at any point can be resumed correctly:
      - train_state["data_files_used"]: list of absolute file paths already consumed (popped) for training, in
        the order they were consumed. The ordering matters: it's the "previous epoch order" that the
        gap-delaying reshuffle (see _reshuffle_for_new_epoch) consumes when it builds the next epoch's order.
      - train_state["rev_data_files_remaining"]: the queue of files still to be served in the current pass,
        in REVERSE serve order, for O(1) pop. Saved in train_state so the shuffle order survives checkpoint+resume.
      - train_state["old_train_data_dirs"]: bounded history of data directories seen, used to prune stale
        entries out of data_files_used once a directory has rotated far enough into the past.

    Note on data_files_used being a list - it was historically a set but we changed it to a list since we needed
    to track ordering. We locally convert to a set in some places for performance. The list itself never contains
    duplicates within an epoch: it's reset to empty at each reshuffle, and within an epoch each file is served
    (and thus appended) at most once because rev_data_files_remaining holds no dups.
    """

    def __init__(self, train_state: Dict[str, Any], no_repeat_files: bool):
        """
        no_repeat_files=True: When training data runs out, stop. peek()/pop() return None.
        no_repeat_files=False: When data runs out, reshuffle the full file list, wipe data_files_used, and keep going.
        data_files_used is still used in that it will prevent repeats of the files used in the same epoch so far,
        when resuming from a checkpoint mid-epoch.

        Owns initialization of the three train_state fields it manages: on a fresh run (or a checkpoint predating
        them) they may be absent, so we default them here. On a resume they are already present and restored from
        the checkpoint, so setdefault leaves them untouched.

        Backwards compatibility: a checkpoint written before data_files_used became a list stores it as a set.
        We convert any such set into a list here. Order within the old set is meaningless, so we shuffle it to
        give the reshuffle a well-defined (if arbitrary) "previous epoch order" to work from.
        """

        self.train_state = train_state
        self.no_repeat_files = no_repeat_files

        # Backwards compat: convert a legacy set into a shuffled list before defaulting.
        if isinstance(train_state.get("data_files_used"), set):
            converted = list(train_state["data_files_used"])
            random.shuffle(converted)
            train_state["data_files_used"] = converted

        # See class docstring for the meaning of these fields.
        train_state.setdefault("data_files_used", [])
        train_state.setdefault("rev_data_files_remaining", [])
        train_state.setdefault("old_train_data_dirs", [])

        # All .npz files in the current data directory. Recomputed each set_data_dir, not part of train_state.
        self._all_files: List[str] = []

    @staticmethod
    def _uniform_interleave(a: List[str], b: List[str]) -> List[str]:
        """
        Interleave two lists into one, preserving the relative order WITHIN each input, with the items of a and
        b spread uniformly through the result. Linear pass: at each step take the next item from a or from b,
        choosing a with probability len(remaining_a)/(len(remaining_a)+len(remaining_b)).
        """
        result: List[str] = []
        i = 0
        j = 0
        while i < len(a) or j < len(b):
            rem_a = len(a) - i
            rem_b = len(b) - j
            if random.random() < rem_a / (rem_a + rem_b):
                result.append(a[i])
                i += 1
            else:
                result.append(b[j])
                j += 1
        return result

    def _reshuffle_for_new_epoch(self) -> List[str]:
        """
        Prepare the shuffle order for a fresh epoch, taking into account the previous epoch's consumption order
        (train_state["data_files_used"]) and the current dataset (self._all_files).

        Uses a reservoir-sampling-based shuffle where a file is forbidden from recurring within ~1/3 of the
        dataset of its previous occurrence.

        Returns the new order in FORWARD serve order (caller reverses it).

        Algorithm:

        Let P = previous-epoch order filtered to files still present,
        Let N = files present this epoch but not in P
        Let k begin at (len(P)*2+1)//3)

        Begin with reservoir = N + P[:k]
        Repeatedly pop a uniform-random item out of the reservoir into the output, and replace it with
        P[k], then increment k. When P runs out, then shuffle whatever is left in the reservoir and append it.
        """
        all_files_set = set(self._all_files)
        # Filter the previous-epoch order to only files still in the dataset, preserving order.
        # Handles if the dataset somehow changed underneath us. This is generally illegal / undefined behavior
        # while training is live, but it's good to support between runs in case someone wants to try it.
        prev = [f for f in self.train_state["data_files_used"] if f in all_files_set]
        prev_set = set(prev)
        # New files: present this epoch but not seen last epoch. These have no "recently seen" constraint, so
        # they are eligible from the very start. Shuffle them so they are not emitted in os.listdir order.
        new_files = [f for f in self._all_files if f not in prev_set]
        random.shuffle(new_files)

        n = len(prev)
        k = (n * 2 + 1) // 3

        reservoir = new_files + prev[:k]
        order: List[str] = []
        while k < n:
            idx = random.randrange(len(reservoir))
            # Swap-remove for O(1). Reservoir order is irrelevant since we always draw uniformly at random.
            reservoir[idx], reservoir[-1] = reservoir[-1], reservoir[idx]
            order.append(reservoir.pop())
            reservoir.append(prev[k])
            k += 1
        random.shuffle(reservoir)
        order.extend(reservoir)
        return order

    def set_data_dir_if_has_remaining_files(self, tdatadir: str) -> bool:
        """
        Load a new shuffled data directory and reconcile the train_state against the files the
        directory actually provides, but ONLY if it can serve at least one file right now
        (same condition as has_any_remaining_data())

        Returns True if the directory was set, False if it was declined for lack of data.
        """
        all_files = [os.path.join(tdatadir, fname) for fname in os.listdir(tdatadir) if fname.endswith(".npz")]
        all_files_set = set(all_files)

        # "epoch0" = the files to use for the first pass through this data dir in this process
        # On a fresh start data_files_used is empty so this is all files, on a resume it's the leftover files of an epoch
        # that was in progress when we were killed (data_files_used having been restored from the checkpoint).
        used_set = set(self.train_state["data_files_used"])
        epoch0_train_files = [path for path in all_files if path not in used_set]
        num_already_used = len(all_files) - len(epoch0_train_files)
        if self.no_repeat_files:
            logging.info(f"Dropping {num_already_used}/{len(all_files)} already-used files in: {tdatadir} (no_repeat_files=True, never reused)")
        else:
            logging.info(f"Deferring {num_already_used}/{len(all_files)} already-used files in: {tdatadir} until the next epoch")

        rev_remaining = self.train_state["rev_data_files_remaining"]
        # Order-preservingly filter out anything no longer in the dataset.
        rev_remaining = [f for f in rev_remaining if f in all_files_set]
        # Find not-yet-used files that are present interleave them in.
        # (On a fresh dir this just becomes a shuffle of the whole list while the rev_remaining is empty)
        queued_set = set(rev_remaining)
        new_queue_files = [f for f in epoch0_train_files if f not in queued_set]
        random.shuffle(new_queue_files)
        rev_remaining = self._uniform_interleave(rev_remaining, new_queue_files)

        # Decide whether this directory can serve anything before mutating any history.
        # An empty epoch0 is fine (the full list cycles), so only a dir with no .npz at all is unservable.
        # In no_repeat mode an empty epoch0 is also unservable since nothing will ever be repeated.
        if len(all_files) <= 0 or (self.no_repeat_files and len(rev_remaining) <= 0):
            return False

        self._all_files = all_files
        self.train_state["rev_data_files_remaining"] = rev_remaining

        # Update history of what training data we used
        if tdatadir not in self.train_state["old_train_data_dirs"]:
            self.train_state["old_train_data_dirs"].append(tdatadir)
        # Clear out tracking of sufficiently old files
        while len(self.train_state["old_train_data_dirs"]) > 20:
            old_dir = self.train_state["old_train_data_dirs"][0]
            self.train_state["old_train_data_dirs"] = self.train_state["old_train_data_dirs"][1:]
            # Order-preservingly drop data files that live under the removed directory.
            self.train_state["data_files_used"] = [
                f for f in self.train_state["data_files_used"] if not f.startswith(old_dir)
            ]

        return True

    def has_any_files(self) -> bool:
        return len(self._all_files) > 0

    def has_any_remaining_data(self) -> bool:
        """Whether peek()/pop() can still yield at least one file, given the current directory and mode."""
        if not self.has_any_files():
            return False
        if self.no_repeat_files and len(self.train_state["rev_data_files_remaining"]) <= 0:
            return False
        return True

    def _maybe_refill_remaining(self) -> None:
        if self.train_state["rev_data_files_remaining"]:
            return
        if self.no_repeat_files:
            # Never repeat files, even across restarts (data_files_used persists in the checkpoint).
            return

        order = self._reshuffle_for_new_epoch()

        # Reversed so the front of the serve order ends up at the end of the list (served via pop()).
        order.reverse()
        self.train_state["rev_data_files_remaining"] = order
        self.train_state["data_files_used"] = []

    def peek(self) -> Optional[str]:
        """
        Return the next file to be served without consuming it, or None if no more files are available.
        """
        self._maybe_refill_remaining()
        if not self.train_state["rev_data_files_remaining"]:
            return None
        return self.train_state["rev_data_files_remaining"][-1]

    def pop(self) -> Optional[str]:
        """
        Consume and return the next file, marking it used in train_state, or None if no more are available.
        """
        self._maybe_refill_remaining()
        if not self.train_state["rev_data_files_remaining"]:
            return None
        filename = self.train_state["rev_data_files_remaining"].pop()
        logging.info("Yielding training file for dataset: " + filename)
        self.train_state["data_files_used"].append(filename)
        return filename
