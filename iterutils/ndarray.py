def iter_cartesian(*args, order='C', blocking=None):
    """
    Iterate over a Cartesian product of slices.

    The set of indices that this iterator yields is completely defined by 
    the Cartesian product of slices, but the manner in which it iterates 
    over them is not, and this can be important for performance. The iteration
    order (including blocking, if desired) is controlled by keyword arguments.

    Parameters
    ----------
    ind[, ind, ...]: one or more slices or integers 
        Indices along each iteration dimension. If a slice is provided,
        slice.stop must not be None. slice.start and slice.step can be
        None, in which case they are replaced with 0 and 1. If an integer 
        N is provided, iteration indices are from 0 to N - 1.

    order: 'C' or 'F', optional, default 'C'
        Iteration order. 'C' (C order) means iteration over last dimension first;
        'F' (Fortran order) means iteration over first dimension first. If blocking 
        is specified, the iterator always yields indices for entire blocks, and this argument
        controls the order in which the iterator traverses blocks.

    blocking: iterable of integers, optional 
        If provided, yield slices of indices with the given block size. In this case, the 
        number of elements in the tuple must match the number of index dimensions.

    Yields 
    ------
    Tuple of integers (if blocking=None) or tuple of slices (if blocking is provided).
    The set of indices yielded by the iterator will include each index in the Cartesian 
    product of input slices exactly once.

    """


    # Validate indices 
    slices = tuple(_validate_indices(arg) for arg in args)

    # TODO: convert initial implementation for order='C' to implementation that also 
    # handles order='F'. Should be able to do this by reversing the order of a few tuples 
    if order != 'C':
        raise ValueError("Iteration order must be one of 'C' or 'F'" + 
        " (only 'C' currently supported).")

    # Validate blocking if provided 
    if blocking is None:
        blocksize = tuple(1 for s in slices)
    else:
        blocksize = tuple(int(b) for b in blocking)
    if len(blocksize) != len(slices):
        raise ValueError(
            'Number of block dimensions (%d) must match number of input slices (%d)'
            % (len(blocking), len(slices)))
    
    # Implement iteration by shifting origin. Initially at start of each slice 
    origin = [s.start for s in slices]

    # Handle shifts past the end of slices 
    # Must be done before beginning iteration to detect early termination condition
    _handle_past_end(origin, slices, blocksize)

    # Simple termination condition: first index of origin is at or beyond end of first slice 
    while origin[0] < slices[0].stop:

        # Extract current index or block
        if blocking is None:
            yield tuple(origin)
        else:
            yield _extract_block(origin, slices, blocksize)

        # Shift origin 
        origin[-1] += blocksize[-1]*slices[-1].step 
        _handle_past_end(origin, slices, blocksize)

# Validate representation of input indices
def _validate_indices(arg):

    if isinstance(arg, int):
        return slice(0, arg, 1)

    if isinstance(arg, slice):
        if arg.stop is None:
            raise TypeError("Slice stop must not be None.")
        start = 0 if arg.start is None else arg.start 
        if not isinstance(start, int):
            raise TypeError("Slices must contain integers.")
        stop = arg.stop
        if not isinstance(stop, int):
            raise TypeError("Slices must contain integers.")
        step = 1 if arg.step is None else arg.step
        if not isinstance(step, int):
            raise TypeError("Slices must contain integers.")
        return slice(start, stop, step)

    raise TypeError("Indices must be defined by integers or integer slices.")

# Handle origin indices past end of corresponding slice
# Requires resetting this index to slice start and
# incrementing next-earlier origin index.
# Nothing is done if the first origin index is past the end 
# of the first slice, as this is a termination condition.
def _handle_past_end(origin, slices, blocksize):

    # Iteration intentionally excludes 0
    for i in range(len(origin) - 1, 0, -1):
        if origin[i] >= slices[i].stop:
            origin[i] = slices[i].start
            origin[i-1] += blocksize[i-1]*slices[i-1].step
    return origin
            
# Extract indices of block with current origin 
def _extract_block(origin, slices, blocksize):
    return tuple(slice(
            o,                              # block starts at origin
            min(o + b*s.step, s.stop),      # stop after blocksize unless ended by slice stop 
            s.step)                         # step set by slice set
        for o, s, b in zip(origin, slices, blocksize))
