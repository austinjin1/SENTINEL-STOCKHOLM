# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""EMP BIOM parsing on a tiny SYNTHETIC HDF5 table (code-path test only)."""

import numpy as np

from sentinel.data.sources import emp as EMP


def test_read_biom_hdf5_long_table(tmp_path):
    import h5py

    p = tmp_path / "t.biom"
    # 3 observations x 2 samples, sample-major CSC: s1 = {ACG: 2, TTT: 5}, s2 = {GGG: 7}
    with h5py.File(p, "w") as f:
        f["observation/ids"] = np.array([b"ACG", b"TTT", b"GGG"])
        f["sample/ids"] = np.array([b"s1", b"s2"])
        f["sample/matrix/indptr"] = np.array([0, 2, 3])
        f["sample/matrix/indices"] = np.array([0, 1, 2])
        f["sample/matrix/data"] = np.array([2.0, 5.0, 7.0])
    d = EMP.read_biom_hdf5(p, ["s2"])
    assert d.to_dict("list") == {"sample_id": ["s2"], "sequence": ["GGG"], "count": [7]}
    assert EMP.read_biom_hdf5(p).shape == (3, 3)
