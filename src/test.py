from utils import santa

coder = santa.SantaCoder()

prefix = """# automatically generated by the FlatBuffers compiler, do not modify

# namespace: flatbuf

import flatbuffers

class FloatingPoint(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFloatingPoint(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FloatingPoint()
        x.Init(buf, n + offset)
        return x

    # FloatingPoint
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FloatingPoint
    def Precision(self):"""

suffix = """if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 0"""

middle = coder.infill((prefix, suffix))

print("\033[92m" + prefix + "\033[93m" + middle + "\033[92m" + suffix)
"""
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: flatbuf

import flatbuffers

class FloatingPoint(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFloatingPoint(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FloatingPoint()
        x.Init(buf, n + offset)
        return x

    # FloatingPoint
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FloatingPoint
    def Precision(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 0
"""