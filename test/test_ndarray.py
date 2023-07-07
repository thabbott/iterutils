import iterutils.ndarray as ndarray 
import pytest

class TestIterCartesian:

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            list(ndarray.iter_cartesian(2, order='F'))
        with pytest.raises(ValueError):
            list(ndarray.iter_cartesian(2, blocking=(1, 3)))
        with pytest.raises(TypeError):
            list(ndarray.iter_cartesian('shape'))
        with pytest.raises(TypeError):
            list(ndarray.iter_cartesian(slice(1, None)))
        with pytest.raises(TypeError):
            list(ndarray.iter_cartesian(slice('start', 12, 1)))
        with pytest.raises(TypeError):
            list(ndarray.iter_cartesian(slice(0, 'stop', 1)))
        with pytest.raises(TypeError):
            list(ndarray.iter_cartesian(slice(0, 12, 'step')))

    def test_empty(self):
        assert list(ndarray.iter_cartesian(0)) == []
        assert list(ndarray.iter_cartesian(0, 0, 0)) == []

    def test_1d(self):
        assert list(ndarray.iter_cartesian(3)) == [(0,), (1,), (2,)]
        assert list(ndarray.iter_cartesian(slice(3))) == [(0,), (1,), (2,)]
        assert list(ndarray.iter_cartesian(slice(1, 4))) == [(1,), (2,), (3,)]
        assert list(ndarray.iter_cartesian(slice(1, 8, 3))) == [(1,), (4,), (7,)]

    def test_1d_blocking(self):
        assert list(ndarray.iter_cartesian(3, blocking=(1,))) == \
                [(slice(0, 1, 1),), (slice(1, 2, 1),), (slice(2, 3, 1),)]
        assert list(ndarray.iter_cartesian(6, blocking=(3,))) == \
                [(slice(0, 3, 1),), (slice(3, 6, 1),)]
        assert list(ndarray.iter_cartesian(slice(1, 19, 3), blocking=(3,))) == \
                [(slice(1, 10, 3),), (slice(10, 19, 3),)]
        assert list(ndarray.iter_cartesian(slice(1, 21, 3), blocking=(3,))) == \
                [(slice(1, 10, 3),), (slice(10, 19, 3),), (slice(19, 21, 3),)]

    def test_nd_scalar(self):
        
        ind = list(ndarray.iter_cartesian(2, 3, 4, 5))
        assert ind[0] == (0, 0, 0, 0)
        assert ind[1] == (0, 0, 0, 1)
        assert ind[5] == (0, 0, 1, 0)
        assert ind[6] == (0, 0, 1, 1)
        assert ind[20] == (0, 1, 0, 0)
        assert ind[26] == (0, 1, 1, 1)
        assert ind[60] == (1, 0, 0, 0)
        assert ind[86] == (1, 1, 1, 1)
        assert ind[119] == (1, 2, 3, 4)

    def test_nd_slice(self):

        ind = list(ndarray.iter_cartesian(slice(1, 3), slice(2, 4)))
        assert ind == [(1, 2), (1, 3), (2, 2), (2, 3)]
        
    def test_nd_slice_step(self):

        ind = list(ndarray.iter_cartesian(slice(1, 8, 3), slice(2, 9, 2)))
        assert ind == [(1, 2), (1, 4), (1, 6), (1, 8),
                       (4, 2), (4, 4), (4, 6), (4, 8),
                       (7, 2), (7, 4), (7, 6), (7, 8)]
    
    def test_nd_blocking_scalar(self):
        
        ind = list(ndarray.iter_cartesian(1, 2, 3, blocking=(1, 1, 1)))
        assert ind[0] == (slice(0, 1, 1),)*3
        assert ind[1] == (slice(0, 1, 1), slice(0, 1, 1), slice(1, 2, 1))
        assert ind[5] == (slice(0, 1, 1), slice(1, 2, 1), slice(2, 3, 1))

        ind = list(ndarray.iter_cartesian(1, 2, 3, blocking=(1, 2, 3)))
        assert ind == [(slice(0, 1, 1), slice(0, 2, 1), slice(0, 3, 1))]
        
        ind = list(ndarray.iter_cartesian(1, 2, 3, blocking=(1, 2, 2)))
        assert ind == [
            (slice(0, 1, 1), slice(0, 2, 1), slice(0, 2, 1)),
            (slice(0, 1, 1), slice(0, 2, 1), slice(2, 3, 1))
        ]

    def test_nd_blocking_slice(self):

        ind = list(ndarray.iter_cartesian(
            slice(1, 3), slice(2, 4), slice(3, 5),
            blocking=(1, 1, 1)))
        assert ind[0] == (slice(1, 2, 1), slice(2, 3, 1), slice(3, 4, 1))
        assert ind[1] == (slice(1, 2, 1), slice(2, 3, 1), slice(4, 5, 1))
        assert ind[7] == (slice(2, 3, 1), slice(3, 4, 1), slice(4, 5, 1))
        
        ind = list(ndarray.iter_cartesian(
            slice(1, 3), slice(2, 4), slice(3, 5),
            blocking=(2, 2, 2)))
        assert ind == [(slice(1, 3, 1), slice(2, 4, 1), slice(3, 5, 1))]
        
        ind = list(ndarray.iter_cartesian(
            slice(1, 3), slice(2, 4), slice(3, 6),
            blocking=(2, 2, 2)))
        assert ind == [
            (slice(1, 3, 1), slice(2, 4, 1), slice(3, 5, 1)),
            (slice(1, 3, 1), slice(2, 4, 1), slice(5, 6, 1)),
        ]
    
    def test_nd_blocking_slice_step(self):

        ind = list(ndarray.iter_cartesian(
            slice(1, 3, 1), slice(2, 6, 2), slice(3, 9, 3),
            blocking=(1, 1, 1)))
        assert ind[0] == (slice(1, 2, 1), slice(2, 4, 2), slice(3, 6, 3))
        assert ind[1] == (slice(1, 2, 1), slice(2, 4, 2), slice(6, 9, 3))
        assert ind[7] == (slice(2, 3, 1), slice(4, 6, 2), slice(6, 9, 3))
        
        ind = list(ndarray.iter_cartesian(
            slice(1, 3, 1), slice(2, 6, 2), slice(3, 9, 3),
            blocking=(2, 2, 2)))
        assert ind == [(slice(1, 3, 1), slice(2, 6, 2), slice(3, 9, 3))]
        
        ind = list(ndarray.iter_cartesian(
            slice(1, 3, 1), slice(2, 6, 2), slice(3, 12, 3),
            blocking=(2, 1, 2)))
        assert ind == [
            (slice(1, 3, 1), slice(2, 4, 2), slice(3, 9, 3)),
            (slice(1, 3, 1), slice(2, 4, 2), slice(9, 12, 3)),
            (slice(1, 3, 1), slice(4, 6, 2), slice(3, 9, 3)),
            (slice(1, 3, 1), slice(4, 6, 2), slice(9, 12, 3)),
        ]
        
        ind = list(ndarray.iter_cartesian(
            slice(1, 3, 1), slice(2, 5, 2), slice(3, 10, 3),
            blocking=(2, 1, 2)))
        assert ind == [
            (slice(1, 3, 1), slice(2, 4, 2), slice(3, 9, 3)),
            (slice(1, 3, 1), slice(2, 4, 2), slice(9, 10, 3)),
            (slice(1, 3, 1), slice(4, 5, 2), slice(3, 9, 3)),
            (slice(1, 3, 1), slice(4, 5, 2), slice(9, 10, 3)),
        ]


