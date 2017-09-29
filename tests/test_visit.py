""" Test visit and visititems methods. """
from __future__ import print_function

import os

import pyfive

DIRNAME = os.path.dirname(__file__)
GROUPS_HDF5_FILE = os.path.join(DIRNAME, 'groups.hdf5')


def test_visit_method():
    with pyfive.File(GROUPS_HDF5_FILE) as hfile:

        assert hfile.visit(lambda x: print(x)) is None

        name = 'group2/subgroup1'
        assert hfile.visit(lambda x: x if x == name else None) == name

        name = '/group2/subgroup1'  # starts with /, not found
        assert hfile.visit(lambda x: x if x == name else None) is None

        group2 = hfile['group2']

        name = 'subgroup1'
        assert group2.visit(lambda x: x if x == name else None) == name

        name = 'group2/subgroup1'  # rooted at group2
        assert group2.visit(lambda x: x if x == name else None) is None


def test_visititems_method():
    with pyfive.File(GROUPS_HDF5_FILE) as hfile:

        assert hfile.visititems(lambda x, y: print(x, y.name)) is None

        name = 'group2/subgroup1'
        ret = hfile.visititems(lambda x, y: x if x == name else None)
        assert ret == name

        name = '/group2/subgroup1'  # starts with /, not found
        assert hfile.visititems(lambda x, y: x if x == name else None) is None
        ret = hfile.visititems(lambda x, y: y if y.name == name else None)
        assert ret.name == name
