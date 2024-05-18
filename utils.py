from typing import Iterable, TypeVar


def b_nand(a: int, b: int) -> int:
    return a ^ (b & a)


T = TypeVar("T")


def everything_but_my_index(values: Iterable[T], idx: int) -> Iterable[T]:
    return map(
        lambda v: v[1],
        filter(lambda j: j[0] != idx, enumerate(values)),
    )
