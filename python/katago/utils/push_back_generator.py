from collections import deque
from typing import Generator, TypeVar, Generic

T = TypeVar('T')

class PushBackGenerator(Generic[T]):
    """Wrapper for generators that allows pushing items back on to the generator to be yielded next."""
    def __init__(self, generator: Generator[T, None, None]):
        self._generator = generator
        self._stack = deque()
        self._orig_generator_empty = False

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if self._stack:
            return self._stack.pop()
        if self._orig_generator_empty:
            raise StopIteration()
        try:
            return next(self._generator)
        except StopIteration:
            self._orig_generator_empty = True
            raise

    def push_back(self, item: T) -> None:
        self._stack.append(item)
