from ir import *

class BlockIdy(IR):
    def __init__(self):
        super().__init__()

class BlockIdx(IR):
    def __init__(self):
        super().__init__()

class BlockDimy(IR):
    def __init__(self):
        super().__init__()

class BlockDimx(IR):
    def __init__(self):
        super().__init__()

class ThreadIdy(IR):
    def __init__(self):
        super().__init__()

class ThreadIdx(IR):
    def __init__(self):
        super().__init__()

class SyncThreads(IR):
    def __init__(self):
        super().__init__()

class SyncWarps(IR):
    def __init__(self):
        super().__init__()

class ShuffleDown(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class ShuffleUp(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class ShuffleXor(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class SaveAtThread(IR):
    def __init__(self, src, dst, threadid):
        super().__init__()
        self.src = src
        self.dst = dst
        self.threadid = threadid

class BroadCast(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class Shared(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class Uniq(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class Buffer(IR):
    def __init__(self, dobject):
        super().__init__()
        self.dobject = dobject

class IF(IR):
    def __init__(self, left, condition: Expr, true_var, false_var):
        super().__init__()
        self.left = left
        self.condition = condition
        self.true_var = true_var
        self.false_var = false_var

class Pointer(DObject):
    def __init__(self, dtype, size):
        super().__init__(dtype, size)
        self.__name__ = f'ptr{self.dobject_id}'
        self.dtype = dtype
        self.size = size

    def name(self):
        return self.__name__

class Access_ptr():
    def __init__(self, dobject:Pointer, idx):
        self.dobject = dobject
        self.idx = idx