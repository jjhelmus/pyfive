""" Representation and reading of HDF5 datatype messages. """

from collections import OrderedDict

from .core import _padded_size, _structure_size, _unpack_struct_from
from .core import InvalidHDF5File


class DatatypeMessage(object):
    """ Representation of a HDF5 Datatype Message. """
    # Contents and layout defined in IV.A.2.d.

    def __init__(self, buf, offset):
        self.buf = buf
        self.offset = offset
        self.dtype = self.determine_dtype()

    def determine_dtype(self):
        """ Return the dtype (often numpy-like) for the datatype message.  """
        datatype_msg = _unpack_struct_from(DATATYPE_MSG, self.buf, self.offset)
        self.offset += DATATYPE_MSG_SIZE
        # last 4 bits
        datatype_class = datatype_msg['class_and_version'] & 0x0F

        if datatype_class == DATATYPE_FIXED_POINT:
            return self._determine_dtype_fixed_point(datatype_msg)
        elif datatype_class == DATATYPE_FLOATING_POINT:
            return self._determine_dtype_floating_point(datatype_msg)
        elif datatype_class == DATATYPE_TIME:
            raise NotImplementedError("Time datatype class not supported.")
        elif datatype_class == DATATYPE_STRING:
            return self._determine_dtype_string(datatype_msg)
        elif datatype_class == DATATYPE_BITFIELD:
            raise NotImplementedError("Bitfield datatype class not supported.")
        elif datatype_class == DATATYPE_OPAQUE:
            raise NotImplementedError("Opaque datatype class not supported.")
        elif datatype_class == DATATYPE_COMPOUND:
            return self._determine_dtype_compound(datatype_msg)
        elif datatype_class == DATATYPE_REFERENCE:
            return ('REFERENCE', datatype_msg['size'])
        elif datatype_class == DATATYPE_ENUMERATED:
            raise NotImplementedError(
                "Enumerated datatype class not supported.")
        elif datatype_class == DATATYPE_ARRAY:
            raise NotImplementedError("Array datatype class not supported.")
        elif datatype_class == DATATYPE_VARIABLE_LENGTH:
            vlen_type = self._determine_dtype_vlen(datatype_msg)
            if vlen_type[0] == 'VLEN_SEQUENCE':
                base_type = self.determine_dtype()
                vlen_type = ('VLEN_SEQUENCE', base_type)
            return vlen_type
        raise InvalidHDF5File('Invalid datatype class %i' % (datatype_class))

    def _determine_dtype_fixed_point(self, datatype_msg):
        """ Return the NumPy dtype for a fixed point class. """
        # fixed-point types are assumed to follow IEEE standard format
        length_in_bytes = datatype_msg['size']
        if length_in_bytes not in [1, 2, 4, 8]:
            raise NotImplementedError("Unsupported datatype size")

        signed = datatype_msg['class_bit_field_0'] & 0x08
        if signed > 0:
            dtype_char = 'i'
        else:
            dtype_char = 'u'

        byte_order = datatype_msg['class_bit_field_0'] & 0x01
        if byte_order == 0:
            byte_order_char = '<'  # little-endian
        else:
            byte_order_char = '>'  # big-endian

        # 4-byte fixed-point property description
        # not read, assumed to be IEEE standard format
        self.offset += 4

        return byte_order_char + dtype_char + str(length_in_bytes)

    def _determine_dtype_floating_point(self, datatype_msg):
        """ Return the NumPy dtype for a floating point class. """
        # Floating point types are assumed to follow IEEE standard formats
        length_in_bytes = datatype_msg['size']
        if length_in_bytes not in [1, 2, 4, 8]:
            raise NotImplementedError("Unsupported datatype size")

        dtype_char = 'f'

        byte_order = datatype_msg['class_bit_field_0'] & 0x01
        if byte_order == 0:
            byte_order_char = '<'  # little-endian
        else:
            byte_order_char = '>'  # big-endian

        # 12-bytes floating-point property description
        # not read, assumed to be IEEE standard format
        self.offset += 12

        return byte_order_char + dtype_char + str(length_in_bytes)

    @staticmethod
    def _determine_dtype_string(datatype_msg):
        """ Return the NumPy dtype for a string class. """
        return 'S' + str(datatype_msg['size'])

    def _determine_dtype_compound(self, datatype_msg):
        """ Return the dtype of a compound class if supported. """
        bit_field_0 = datatype_msg['class_bit_field_0']
        bit_field_1 = datatype_msg['class_bit_field_1']
        n_comp = bit_field_0 + (bit_field_1 << 4)

        # read in the members of the compound datatype
        members = []
        for _ in range(n_comp):
            null_location = self.buf.index(b'\x00', self.offset)
            name_size = _padded_size(null_location - self.offset, 8)
            name = self.buf[self.offset:self.offset+name_size]
            name = name.strip(b'\x00').decode('utf-8')
            self.offset += name_size

            prop_desc = _unpack_struct_from(
                COMPOUND_PROP_DESC_V1, self.buf, self.offset)
            self.offset += COMPOUND_PROP_DESC_V1_SIZE

            comp_dtype = self.determine_dtype()
            members.append((name, prop_desc, comp_dtype))

        # determine if the compound dtype is complex64/complex128
        if len(members) == 2:
            name1, prop1, dtype1 = members[0]
            name2, prop2, dtype2 = members[1]
            names_valid = (name1 == 'r' and name2 == 'i')
            complex_dtype_map = {
                '>f4': '>c8',
                '<f4': '<c8',
                '>f8': '>c16',
                '<f8': '<c16',
            }
            dtypes_valid = (dtype1 == dtype2) and dtype1 in complex_dtype_map
            half = datatype_msg['size'] // 2
            offsets_valid = (prop1['offset'] == 0 and prop2['offset'] == half)
            props_valid = (
                prop1['dimensionality'] == 0 and
                prop1['permutation'] == 0 and
                prop1['dim_size_1'] == 0 and
                prop1['dim_size_2'] == 0 and
                prop1['dim_size_3'] == 0 and
                prop1['dim_size_4'] == 0 and
                prop2['dimensionality'] == 0 and
                prop2['permutation'] == 0 and
                prop2['dim_size_1'] == 0 and
                prop2['dim_size_2'] == 0 and
                prop2['dim_size_3'] == 0 and
                prop2['dim_size_4'] == 0
            )
            if names_valid and dtypes_valid and offsets_valid and props_valid:
                return complex_dtype_map[dtype1]

        raise NotImplementedError("Compond dtype not supported.")

    @staticmethod
    def _determine_dtype_vlen(datatype_msg):
        """ Return the dtype information for a variable length class. """
        vlen_type = datatype_msg['class_bit_field_0'] & 0x01
        if vlen_type != 1:
            return ('VLEN_SEQUENCE', 0, 0)
        padding_type = datatype_msg['class_bit_field_0'] >> 4  # bits 4-7
        character_set = datatype_msg['class_bit_field_1'] & 0x01
        return ('VLEN_STRING', padding_type, character_set)


# IV.A.2.d The Datatype Message

DATATYPE_MSG = OrderedDict((
    ('class_and_version', 'B'),
    ('class_bit_field_0', 'B'),
    ('class_bit_field_1', 'B'),
    ('class_bit_field_2', 'B'),
    ('size', 'I'),
))
DATATYPE_MSG_SIZE = _structure_size(DATATYPE_MSG)


COMPOUND_PROP_DESC_V1 = OrderedDict((
    ('offset', 'I'),
    ('dimensionality', 'B'),
    ('reserved_0', 'B'),
    ('reserved_1', 'B'),
    ('reserved_2', 'B'),
    ('permutation', 'I'),
    ('reserved_3', 'I'),
    ('dim_size_1', 'I'),
    ('dim_size_2', 'I'),
    ('dim_size_3', 'I'),
    ('dim_size_4', 'I'),
))
COMPOUND_PROP_DESC_V1_SIZE = _structure_size(COMPOUND_PROP_DESC_V1)


# Datatype message, datatype classes
DATATYPE_FIXED_POINT = 0
DATATYPE_FLOATING_POINT = 1
DATATYPE_TIME = 2
DATATYPE_STRING = 3
DATATYPE_BITFIELD = 4
DATATYPE_OPAQUE = 5
DATATYPE_COMPOUND = 6
DATATYPE_REFERENCE = 7
DATATYPE_ENUMERATED = 8
DATATYPE_VARIABLE_LENGTH = 9
DATATYPE_ARRAY = 10
