""" Test new style groups in pyfive. """
import os

import pyfive

DIRNAME = os.path.dirname(__file__)
NEW_STYLE_GROUPS_HDF5_FILE = os.path.join(DIRNAME, 'new_style_groups.hdf5')


def test_groups():

    with pyfive.File(NEW_STYLE_GROUPS_HDF5_FILE) as hfile:

        assert len(hfile) == 9
        grp0 = hfile['group0']
        grp1 = hfile['group1']
        grp2 = hfile['group2']
        grp3 = hfile['group3']
        grp4 = hfile['group4']
        grp5 = hfile['group5']
        grp6 = hfile['group6']
        grp7 = hfile['group7']
        grp8 = hfile['group8']
