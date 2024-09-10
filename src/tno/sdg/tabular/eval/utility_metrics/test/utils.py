"""
Module with utility functions and precomputed test input for the tests.
"""

from __future__ import annotations

from typing import Any, Generator, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

HALF_VECTOR_LENGTH = 500
VECTOR_LENGTH = 2 * HALF_VECTOR_LENGTH
NR_OF_VECTORS = 28
RANDOM_RANGE: tuple[int, int] = (1, 900)
BINS = 3
BIN_RANGE = RANDOM_RANGE[1] // BINS
CHUNK_SIZE = 4
CAT_COLS = [f"col{i+1}" for i in range(CHUNK_SIZE)]
CATS_FULL = list(range(BINS))
CATS_HALF = CATS_FULL[: len(CATS_FULL) // 2]


def add_arguments(
    arguments: Sequence[tuple[Any, ...]] | Sequence[pd.DataFrame],
    extra_arg: Any | tuple[Any],
) -> list[tuple[Any, ...]]:
    """
    Function that takes a list of tuples used as test input and adds one or more arguments to each
    of those tuples.

    :param arguments: List of tuples with test inputs.
    :param extra_arg: One or more arguments that need to be added to each tuple.
    :return: List with updated tuples.
    """
    if not isinstance(extra_arg, tuple):
        extra_arg = (extra_arg,)

    if isinstance(arguments[0], pd.DataFrame):
        return [(df,) + extra_arg for df in arguments]
    return [tup + extra_arg for tup in arguments]


def _random_correlated_vectors(
    numerical: bool = False,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Sample two random vectors that are perfectly correlated
    :return: Two correlated vectors.
    """
    random_vector = np.random.randint(low=0, high=BINS, size=VECTOR_LENGTH)

    if numerical:
        vector_1 = random_vector + abs(np.random.normal(0, 1 / 100, size=VECTOR_LENGTH))
    else:
        vector_1 = random_vector
    return vector_1, random_vector


def _random_uncorrelated_vectors(
    numerical: bool = False,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Sample two random vectors that are completely uncorrelated.

    :return: Two uncorrelated vectors.
    """
    rng = np.random.default_rng()
    v1 = np.repeat(CATS_FULL, len(CATS_FULL))
    v2 = np.tile(CATS_FULL, len(CATS_FULL))
    if numerical:
        v1 = np.add(v1, abs(rng.normal(0, 1 / 100, v1.shape)), casting="unsafe")
    perm = rng.permutation(len(v1))
    return v1[perm], v2[perm]


def _to_single_dataframe(*vectors: npt.NDArray[Any]) -> pd.DataFrame:
    """
    Convert an arbitrary number of equal-size vectors to a Dataframe.
    :return: The dataframe.
    """
    return pd.DataFrame(data=dict(zip(CAT_COLS, vectors)))


def _to_multiple_dataframes(
    vector_tuples: Sequence[tuple[npt.NDArray[Any], npt.NDArray[Any]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Turn a sequence of multiple tuples of vectors into two dataframes, where the first entries of
    the tuples are inserted into the first dataframe and the second entries of the tuples are
    inserted into the second dataframe.

    :param vector_tuples: Sequence of tuples of vectors.
    :return: A tuple of two dataframes.
    """
    data1 = {}
    data2 = {}
    for c, vector_tuple in zip(CAT_COLS, vector_tuples):
        v1, v2 = vector_tuple
        data1[c] = v1
        data2[c] = v2
    return pd.DataFrame(data=data1), pd.DataFrame(data=data2)


def _chunks(lst: list[Any]) -> Generator[list[Any]]:
    """
    Cut a list of arbitrary elements into chunks.

    :param lst: A list of elements.
    :return: A generator for lists of elements, each list being a chunk.
    """
    for i in range(0, len(lst), CHUNK_SIZE):
        yield lst[i : i + CHUNK_SIZE]


def _to_orthogonal_distributions(
    random_vector: npt.NDArray[Any],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Turn a vector into two orthogonal distributions.

    :param random_vector: A vector with random elements.
    :return: A tuple of two orthogonal distributions.
    """
    vector_1 = np.concatenate(
        (random_vector[:HALF_VECTOR_LENGTH], np.array([0] * HALF_VECTOR_LENGTH))
    )
    vector_2 = np.concatenate(
        (np.array([0] * HALF_VECTOR_LENGTH), random_vector[HALF_VECTOR_LENGTH:])
    )
    return vector_1 / vector_1.sum(), vector_2 / vector_2.sum()


CORRELATED_VECTORS_CAT = [
    _random_correlated_vectors(False) for _ in range(NR_OF_VECTORS)
]
CORRELATED_VECTORS_NUM = [
    _random_correlated_vectors(True) for _ in range(NR_OF_VECTORS)
]
UNCORRELATED_VECTORS_CAT = [
    _random_uncorrelated_vectors(False) for _ in range(NR_OF_VECTORS)
]
UNCORRELATED_VECTORS_NUM = [
    _random_uncorrelated_vectors(True) for _ in range(NR_OF_VECTORS)
]
CORRELATED_SINGLE_DFS_CAT = [_to_single_dataframe(*vs) for vs in CORRELATED_VECTORS_CAT]
UNCORRELATED_SINGLE_DFS_CAT = [
    _to_single_dataframe(*vs) for vs in UNCORRELATED_VECTORS_CAT
]

CORRELATED_MULTIPLE_DFS_CAT = [
    _to_multiple_dataframes(vs) for vs in _chunks(CORRELATED_VECTORS_CAT)
]
UNCORRELATED_MULTIPLE_DFS_CAT = [
    _to_multiple_dataframes(vs) for vs in _chunks(UNCORRELATED_VECTORS_CAT)
]
CORRELATED_SINGLE_DFS_NUM = [_to_single_dataframe(*vs) for vs in CORRELATED_VECTORS_NUM]
UNCORRELATED_SINGLE_DFS_NUM = [
    _to_single_dataframe(*vs) for vs in UNCORRELATED_VECTORS_NUM
]


CORRELATED_MULTIPLE_DFS_NUM = [
    _to_multiple_dataframes(vs) for vs in _chunks(CORRELATED_VECTORS_NUM)
]
UNCORRELATED_MULTIPLE_DFS_NUM = [
    _to_multiple_dataframes(vs) for vs in _chunks(UNCORRELATED_VECTORS_NUM)
]

RANDOM_NUM_VECTORS: list[npt.NDArray[Any]] = [
    np.concatenate(
        (
            np.asarray(RANDOM_RANGE),
            np.random.randint(
                low=RANDOM_RANGE[0], high=RANDOM_RANGE[1], size=VECTOR_LENGTH - 2
            ),
        )
    )
    for _ in range(NR_OF_VECTORS)
]

RANDOM_CAT_VECTORS: list[npt.NDArray[Any]] = [
    np.concatenate(
        (
            np.asarray(CATS_FULL),
            np.random.choice(CATS_FULL, size=VECTOR_LENGTH - len(CATS_FULL)),
        )
    )
    for _ in range(NR_OF_VECTORS)
]

ORTHOGONAL_DISTRIBUTIONS = [_to_orthogonal_distributions(r) for r in RANDOM_NUM_VECTORS]
IDENTICAL_DISTRIBUTIONS = [(r, r) for r in RANDOM_NUM_VECTORS]
ORTHOGONAL_DFS = [
    _to_multiple_dataframes(vs) for vs in _chunks(ORTHOGONAL_DISTRIBUTIONS)
]

DUPLICATED_DFS_NUM = [
    pd.DataFrame(data={c: v for c in CAT_COLS[:2]}) for v in RANDOM_NUM_VECTORS
]
DUPLICATED_DFS_CAT = [
    pd.DataFrame(data={c: v for c in CAT_COLS[:2]}) for v in RANDOM_CAT_VECTORS
]

RANDOM_NUM_DFS = [_to_single_dataframe(*cols) for cols in _chunks(RANDOM_NUM_VECTORS)]
RANDOM_CAT_DFS = [_to_single_dataframe(*cols) for cols in _chunks(RANDOM_CAT_VECTORS)]
