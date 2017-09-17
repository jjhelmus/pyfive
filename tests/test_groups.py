""" Test groups and sub-groups in pyfive. """
import os

import pyfive

DIRNAME = os.path.dirname(__file__)
GROUPS_HDF5_FILE = os.path.join(DIRNAME, 'groups.hdf5')


def test_groups():

    with pyfive.File(GROUPS_HDF5_FILE) as hfile:

        # groups
        assert len(hfile) == 2
        grp1 = hfile['group1']
        grp2 = hfile['group2']

        # sub-groups
        assert len(grp1) == 0
        assert len(grp2) == 2
        sub1 = grp2['subgroup1']
        sub2 = grp2['subgroup2']

        # sub-sub-groups
        assert len(sub1) == 0
        assert len(sub2) == 3
        sub2['sub_subgroup1']
        sub2['sub_subgroup2']
        sub2['sub_subgroup3']
