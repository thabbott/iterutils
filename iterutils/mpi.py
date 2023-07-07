import itertools

def partition_contiguous(iterable, rank, size):
    """
    Extract a subset of an iterable for processing by a given MPI rank.
    Iterable partitioning is contiguous: e.g., with 5 ranks and an iterable 
    of length 12, ranks are assigned to iterable elements as follows:

    Element: | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
    Rank:    |     0     |     1     |   2   |   3   |    4    |

    Constructing a contiguous partition requires knowledge of the length 
    of the iterable, so the input iterable is converted to a list, and 
    slices of this list are returned. If this causes performance issues
    (e.g., for long iterators, or iterators that yield large objects),
    consider using `partition_striped` instead.

    Parameters
    ----------
    iterable: iterable 
        Full (non-partitioned) iterable 

    rank: int 
        MPI rank (between 0 and size - 1, inclusive)

    size: int 
        Total number of MPI ranks

    Returns
    -------
    list
        Subset of input iterable assigned to given MPI rank, as a list 
    """
    iterable = list(iterable)
    length = len(iterable) // size
    remainder = len(iterable) % size
    if rank < remainder:
        length = length + 1
        start = length*rank
    else:
        start = remainder*(length + 1) + (rank - remainder)*length

    return iterable[start:start+length]

def partition_striped(iterable, rank, size):
    """
    Extract a subset of an iterable for processing by a given MPI rank.
    Iterable partitioning is striped: e.g., with 5 ranks and an iterable 
    of length 12, ranks are assigned to iterable elements as follows:

    Element: | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
    Rank:    | 0 | 1 | 2 | 3 | 4 | 0 | 1 | 2 | 3 | 4 | 0  | 1  |

    Unlike `partition_contiguous`, `partition_striped` does not require 
    knowledge of the length of the iterable, so the input iterable is 
    not not converted to a list, and subsets are returned as `itertool.islice`
    objects. These can be convert to list with `list()` if desired.

    Parameters
    ----------
    iterable: iterable 
        Full (non-partitioned) iterable 

    rank: int 
        MPI rank (between 0 and size - 1, inclusive)

    size: int 
        Total number of MPI ranks

    Returns
    -------
    itertools.islice
        Subset of input iterable assigned to given MPI rank
    """
    return itertools.islice(iterable, rank, None, size)

if __name__ == '__main__':

    print('Contiguous')
    iterable = range(12)
    print(list(iterable))
    for rank in range(5):
        print(rank, list(partition_contiguous(iterable, rank, 5)))
    
    print('Contiguous')
    iterable = range(12)
    print(list(iterable))
    for rank in range(3):
        print(rank, list(partition_contiguous(iterable, rank, 3)))

    print('Striped')
    iterable = range(12)
    print(list(iterable))
    for rank in range(5):
        print(rank, list(partition_striped(iterable, rank, 5)))

