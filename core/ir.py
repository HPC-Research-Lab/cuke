class IR:
    def __init__(self) -> None:
        self.ast_ref = None

class DOject(IR):
    nobjects = 0

    def __init__(self, dtype: str):
        super(IR, self).__init__()
        self.dobject_id = DOject.nobjects
        DOject.nobjects += 1
        self.dtype = dtype


class Expr(IR):
    def __init__(self, left, right, op: str):
        super(IR, self).__init__()
        self.left = left
        self.right = right
        self.op = op



class Assignment(IR):
    def __init__(self, lhs, rhs, op=None):
        super(IR, self).__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.op = op



class Loop(IR):
    loop_id = 0

    def __init__(self, start, end, step, body: list):
        super(IR, self).__init__()
        self.lid = Loop.loop_id
        Loop.loop_id += 1
        self.start = start
        self.end = end
        self.step = step
        self.body = body
        self.iterate = Scalar('int', f'_l{self.lid}')


class Scalar(DOject):
    def __init__(self, dtype: str, name: str = None, is_arg = False, val = None):
        super().__init__(dtype)
        self.__name__ = name if name else f's{self.dobject_id}'
        self.size = []
        self.val = val
        self.is_arg = is_arg


    def name(self):
        return self.__name__

    def addr(self):
        return self.name()


class Slice(IR):
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step


class Ndarray(DOject):
    def __init__(self, dtype: str, size: tuple, name: str = None, is_arg = False, val = None):
        super().__init__(dtype)
        self.size = size
        self.__name__ = name if name else f'arr{self.dobject_id}'
        self.val = val # val is None, 0, or 1
        self.is_arg = is_arg

    def __getitem__(self, item):
        return f'{self.__name__}[{item}]'

    def name(self):
        return self.__name__

    def addr(self):
        return self.name()


class Index(IR):
    nindices = 0
    def __init__(self, dobject, index=None, ind_arr=None):
        super(IR, self).__init__()
        self.dobject = dobject
        self.index = index
        self.ind_arr = ind_arr
        self.dtype = self.dobject.dtype
        if ind_arr == None:
            self.size = dobject.size[1:]
        elif type(ind_arr) == Ndarray:
            self.size = ind_arr.size + dobject.size[1:]
        elif type(ind_arr) == Slice:
            s = Expr(Expr(ind_arr.stop, ind_arr.start, '-'), ind_arr.step, '/')
            self.size = [s] + dobject.size[1:]
        self.index_id = Index.nindices
        Index.nindices += 1


    def name(self):
        return f'ref{self.index_id}_{self.dobject.name()}'

    def addr(self):
        if self.ind_arr:
            return f'{self.dobject}[{self.ind_arr[0]}]'
        else:
            return f'{self.dobject}[0]'




class Decl(IR):
    def __init__(self, dobject):
        super(IR, self).__init__()
        self.dobject = dobject