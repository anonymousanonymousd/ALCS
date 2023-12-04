import numpy as np

def pad_slice(array, slice_r, slice_c):
    assert len(array.shape) >= 2

    r1, r2 = slice_r
    c1, c2 = slice_c
    assert r2 > r1
    assert c2 > c1

    pr1 = max(r1, 0)
    pc1 = max(c1, 0)

    sl = array[int(pr1):int(r2), int(pc1):int(c2), :]
    slr, slc = sl.shape[:2]

    padded_sl = np.zeros((int(r2) - int(r1), int(c2) - int(c1)) + array.shape[2:])
    pad_fr_r = pr1 - r1
    pad_to_r = pad_fr_r + slr
    pad_fr_c = pc1 - c1
    pad_to_c = pad_fr_c + slc

    padded_sl[int(pad_fr_r):int(pad_to_r), int(pad_fr_c):int(pad_to_c), :] = sl

    return padded_sl
