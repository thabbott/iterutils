import pytest
import iterutils.mpi as mpi

class TestPartitionContiguous:
    
    def test_validation(self):
        iterable = range(12)
        with pytest.raises(ValueError):
            mpi.partition_contiguous(iterable, -1, 5)
        with pytest.raises(ValueError):
            mpi.partition_contiguous(iterable, 0, 0)
        with pytest.raises(ValueError):
            mpi.partition_contiguous(iterable, 0, -1)
        with pytest.raises(ValueError):
            mpi.partition_contiguous(iterable, 5, 5)
        with pytest.raises(ValueError):
            mpi.partition_contiguous(iterable, 6, 5)
    
    def test_empty(self):
        iterable = range(0)
        size = 5
        for rank in range(size):
            assert mpi.partition_contiguous(iterable, rank, size) == []

    def test_single(self):
        iterable = range(12)
        size = 1
        assert mpi.partition_contiguous(iterable, 0, size) == list(iterable)

    def test_no_remainder(self):
        iterable = range(12)
        size = 3
        assert mpi.partition_contiguous(iterable, 0, size) == [0, 1, 2, 3]
        assert mpi.partition_contiguous(iterable, 1, size) == [4, 5, 6, 7]
        assert mpi.partition_contiguous(iterable, 2, size) == [8, 9, 10, 11]

    def test_remainder(self):
        iterable = range(12)
        size = 5
        assert mpi.partition_contiguous(iterable, 0, size) == [0, 1, 2]
        assert mpi.partition_contiguous(iterable, 1, size) == [3, 4, 5]
        assert mpi.partition_contiguous(iterable, 2, size) == [6, 7]
        assert mpi.partition_contiguous(iterable, 3, size) == [8, 9]
        assert mpi.partition_contiguous(iterable, 4, size) == [10, 11]

class TestPartitionStriped:
    
    def test_validation(self):
        iterable = range(12)
        with pytest.raises(ValueError):
            mpi.partition_striped(iterable, -1, 5)
        with pytest.raises(ValueError):
            mpi.partition_striped(iterable, 0, 0)
        with pytest.raises(ValueError):
            mpi.partition_striped(iterable, 0, -1)
        with pytest.raises(ValueError):
            mpi.partition_striped(iterable, 5, 5)
        with pytest.raises(ValueError):
            mpi.partition_striped(iterable, 6, 5)
    
    def test_empty(self):
        iterable = range(0)
        size = 5
        for rank in range(size):
            assert list(mpi.partition_striped(iterable, rank, size)) == []

    def test_single(self):
        iterable = range(12)
        size = 1
        assert list(mpi.partition_striped(iterable, 0, size)) == list(iterable)

    def test_no_remainder(self):
        iterable = range(12)
        size = 3
        assert list(mpi.partition_striped(iterable, 0, size)) == [0, 3, 6, 9]
        assert list(mpi.partition_striped(iterable, 1, size)) == [1, 4, 7, 10]
        assert list(mpi.partition_striped(iterable, 2, size)) == [2, 5, 8, 11]

    def test_remainder(self):
        iterable = range(12)
        size = 5
        assert list(mpi.partition_striped(iterable, 0, size)) == [0, 5, 10]
        assert list(mpi.partition_striped(iterable, 1, size)) == [1, 6, 11]
        assert list(mpi.partition_striped(iterable, 2, size)) == [2, 7]
        assert list(mpi.partition_striped(iterable, 3, size)) == [3, 8]
        assert list(mpi.partition_striped(iterable, 4, size)) == [4, 9]
