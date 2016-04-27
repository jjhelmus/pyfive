""" Test groups and sub-groups in pyfive. """
import os

import pyfive

DIRNAME = os.path.dirname(__file__)
GROUPS_HDF5_FILE = os.path.join(DIRNAME, 'groups.hdf5')


def test_groups():

    hfile = pyfive.HDF5File(GROUPS_HDF5_FILE)

    # groups
    assert len(hfile.groups) == 2
    grp1 = hfile.groups['group1']
    grp2 = hfile.groups['group2']

    # sub-groups
    assert len(grp1.groups) == 0
    assert len(grp2.groups) == 2
    sub1 = grp2.groups['subgroup1']
    sub2 = grp2.groups['subgroup2']

    # sub-sub-groups
    assert len(sub1.groups) == 0
    assert len(sub2.groups) == 3
    sub2.groups['sub_subgroup1']
    sub2.groups['sub_subgroup2']
    sub2.groups['sub_subgroup3']
