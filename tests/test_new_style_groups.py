""" Test new style groups in pyfive. """
import os
import pyfive

DIRNAME = os.path.dirname(__file__)
NEW_STYLE_GROUPS_HDF5_FILE = os.path.join(DIRNAME, "new_style_groups.hdf5")


def test_groups():
    with pyfive.File(NEW_STYLE_GROUPS_HDF5_FILE) as hfile:
        assert len(hfile) == 9
        # test that the objects are stored in the correct order
        # (file was created with track_order=True)
        for i, grp in enumerate(hfile):
            assert grp == "group{:d}".format(i)
