import os
import random
import logging
from typing import Dict, List, Optional, Any


class TrainingDataGenerator:
    """
    Selects which shuffled training data (.npz) files to train on, in order.

    This class holds a shared reference to the run's train_state dict, MUTATING IT IN PLACE.
    The contract is that train_state always reflects the correct "used" status, so that a checkpoint
    taken of train_state at any point can be resumed correctly:
      - train_state["data_files_used"]: set of absolute file paths already consumed (popped) for training.
      - train_state["old_train_data_dirs"]: bounded history of data directories seen, used to prune stale
        entries out of data_files_used once a directory has rotated far enough into the past.
    """

    def __init__(self, train_state: Dict[str, Any], no_repeat_files: bool):
        """
        no_repeat_files=True: When training data runs out, stop. peek()/pop() return None.
        no_repeat_files=False: When data runs out, reshuffle the full file list, wipe data_files_used, and keep going.
        data_files_used is still used in that it will prevent repeats of the files used in the same epoch so far,
        when resuming from a checkpoint mid-epoch.

        Owns initialization of the two train_state fields it manages: on a fresh run (or a checkpoint predating
        them) they may be absent, so we default them here. On a resume they are already present and restored from
        the checkpoint, so setdefault leaves them untouched.
        """

        self.train_state = train_state
        self.no_repeat_files = no_repeat_files
        # See class docstring for the meaning of these two fields.
        train_state.setdefault("data_files_used", set())
        train_state.setdefault("old_train_data_dirs", [])

        # All .npz files in the current data directory.
        self._all_files: List[str] = []
        # Files remaining to be served in the current pass, stored in REVERSE serve order for O(1) pop.
        self._rev_remaining: List[str] = []

    def set_data_dir_if_has_remaining_files(self, tdatadir: str) -> bool:
        """
        Load a new shuffled data directory and (re)initialize the file queue for it, but ONLY if it can
        actually serve at least one file right now (same condition has_any_remaining_data() reports).

        Returns True if the directory was set, False if it was declined for lack of data.
        """
        all_files = [os.path.join(tdatadir, fname) for fname in os.listdir(tdatadir) if fname.endswith(".npz")]
        # "epoch0" = the files to use for the first pass through this data dir in this process
        # On a fresh start data_files_used is empty so this is all files, on a resume it's the leftover files of an epoch
        # that was in progress when we were killed (data_files_used having been restored from the checkpoint).
        epoch0_train_files = [path for path in all_files if path not in self.train_state["data_files_used"]]
        num_already_used = len(all_files) - len(epoch0_train_files)
        if self.no_repeat_files:
            logging.info(f"Dropping {num_already_used}/{len(all_files)} already-used files in: {tdatadir} (no_repeat_files=True, never reused)")
        else:
            logging.info(f"Deferring {num_already_used}/{len(all_files)} already-used files in: {tdatadir} until the next epoch")

        # Decide whether this directory can serve anything before mutating any history.
        # An empty epoch0 is fine (the full list cycles), so only a dir with no .npz at all is unservable.
        # In no_repeat mode an empty epoch0 is also unservable since nothing will ever be repeated.
        if len(all_files) <= 0 or (self.no_repeat_files and len(epoch0_train_files) <= 0):
            return False

        self._all_files = all_files

        # Update history of what training data we used
        if tdatadir not in self.train_state["old_train_data_dirs"]:
            self.train_state["old_train_data_dirs"].append(tdatadir)
        # Clear out tracking of sufficiently old files
        while len(self.train_state["old_train_data_dirs"]) > 20:
            old_dir = self.train_state["old_train_data_dirs"][0]
            self.train_state["old_train_data_dirs"] = self.train_state["old_train_data_dirs"][1:]
            for filename in list(self.train_state["data_files_used"]):
                if filename.startswith(old_dir):
                    self.train_state["data_files_used"].remove(filename)

        # Seed the queue with the first pass over the not-yet-used subset.
        # Reversed so the front of the serve order ends up at the end of the list (served via pop()).
        shuffled = epoch0_train_files.copy()
        random.shuffle(shuffled)
        shuffled.reverse()
        self._rev_remaining = shuffled
        return True

    def has_any_files(self) -> bool:
        return len(self._all_files) > 0

    def has_any_remaining_data(self) -> bool:
        """Whether peek()/pop() can still yield at least one file, given the current directory and mode."""
        if not self.has_any_files():
            return False
        if self.no_repeat_files and len(self._rev_remaining) <= 0:
            return False
        return True

    def _maybe_refill_remaining(self) -> None:
        if self._rev_remaining:
            return
        if self.no_repeat_files:
            # Never repeat files, even across restarts (data_files_used persists in the checkpoint).
            return

        # Refill remaining files from a fresh shuffle.
        self.train_state["data_files_used"] = set()
        # Reversed so the front of the serve order ends up at the end of the list (served via pop()).
        shuffled = self._all_files.copy()
        random.shuffle(shuffled)
        shuffled.reverse()
        self._rev_remaining = shuffled

    def peek(self) -> Optional[str]:
        """
        Return the next file to be served without consuming it, or None if no more files are available.
        """
        self._maybe_refill_remaining()
        if not self._rev_remaining:
            return None
        return self._rev_remaining[-1]

    def pop(self) -> Optional[str]:
        """
        Consume and return the next file, marking it used in train_state, or None if no more are available.
        """
        self._maybe_refill_remaining()
        if not self._rev_remaining:
            return None
        filename = self._rev_remaining.pop()
        logging.info("Yielding training file for dataset: " + filename)
        self.train_state["data_files_used"].add(filename)
        return filename
