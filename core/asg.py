import copy
import core
import inspect

MIN_INT = -2147483648
MAX_INT = 2147483647

arith_op = {'add': '+', 'sub': '-', 'mul': '*', 'floordiv': '/', 'truediv': '/'}
math_op = ['round', 'abs', 'nbits']
cmp_op = ['bigger', 'smaller']
func_op = ['apply', 'reduce', 'aggr']
other_op = ['setval', 'einsum', 'index', 'inline', 'getmember']

binary_elw = list(arith_op.keys()) + cmp_op
unary_elw = math_op
elementwise_op = binary_elw + unary_elw


def is_int_var(v):
    return isinstance(v, Tensor) and v.dtype == 'int' and len(v.ref_size) == 0


def is_scalar(v):
    return isinstance(v, int | float) or (isinstance(v, Tensor) and len(v.ref_size) == 0)


def is_1d_tensor(v):
    return isinstance(v, Tensor) and len(v.ref_size) == 1


def is_1dint_tensor(v):
    return is_1d_tensor(v) and v.dtype == 'int'


def eval_const_expr(expr):
    def _eval_expr(e):
        if type(e) == TensorOp and (e.op_type in arith_op):
            lhs = eval_const_expr(e.operators[0])
            if lhs != None:
                rhs = eval_const_expr(e.operators[1])
                if rhs != None:
                    match e.op_type:
                        case 'add':
                            return lhs + rhs
                        case 'sub':
                            return lhs - rhs
                        case 'mul':
                            return lhs * rhs
                        case 'floordiv':
                            return lhs // rhs
                        case 'truediv':
                            return lhs / rhs
                        case _:
                            return None
                else:
                    return None
            else:
                return None
        elif type(e) == Const:
            return e.val
        else:
            return None

    v = _eval_expr(expr)
    if v != None:
        return v
    else:
        return expr

def has_same_value(e1, e2):
    if type(e1) != type(e2):
        return False
    elif type(e1) == Var or type(e1) == Tensor:
        return e1.id == e2.id
    elif type(e1) == Const:
        if e1.dtype == 'int' and e2.dtype == 'int':
            return e1.val == e2.val
        elif e1.dtype == 'slice' and e2.dtype == 'slice':
            return has_same_value(e1.val.start, e2.val.start) and has_same_value(e1.val.stop,
                                                                                 e2.val.stop) and has_same_value(
                e1.val.step, e2.val.step)
        else:
            return False
    elif type(e1) == TensorOp:
        if e1.op_type != e2.op_type:
            return False
        elif e1.op_type in arith_op:
            return has_same_value(e1.operators[0], e2.operators[0]) and has_same_value(e2.operators[1], e2.operators[1])
        else:
            if len(e1.operators) != len(e2.operators):
                return False
            else:
                for i in range(len(e1.operators)):
                    if not has_same_value(e1.operators[i], e2.operators[i]):
                        return False
    return True


def is_same_size(s1, s2):
    if (len(s1) != len(s2)):
        return False
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            if type(s1[i]) == type(s2[i]):
                return has_same_value(s1[i], s2[i])
            else:
                v1 = eval_const_expr(s1[i])
                v2 = eval_const_expr(s2[i])
                return v1 == v2
    return True


def prefix_match_size(s1, s2):
    length = min(len(s1), len(s2))
    for i in range(length):
        if s1[i] != s2[i]:
            if type(s1[i]) == type(s2[i]):
                return has_same_value(s1[i], s2[i])
            else:
                return False
    return True


def bigger(x, y):
    return TensorOp('bigger', x, y)


def smaller(x, y):
    return TensorOp('smaller', x, y)


# By default, the output of func should have the same size for any input, but they can have different sizes in the first dim if out_ofss is provided
def apply(func, data: (list, tuple), axes=None, out_ofs=None, cond=None):
    assert callable(func)
    nparam = len(inspect.signature(func).parameters)
    assert len(data) == nparam
    if axes == None:
        axes = []
    while (len(axes) < nparam):
        axes.append(Const(0, 'int'))
    return TensorOp('apply', func, *data, *axes, out_ofs, cond)


def setval(res, val):
    return TensorOp('setval', res, val)


def inline(src, output, *inputs):
    return TensorOp('inline', src, output, *inputs)


class ASTNode:
    nuniq = 0

    def __init__(self):
        self.decl = []
        self.compute = []
        self.output_order = []
        self.input_orders = []
        self.eval = None
        self.ref_by = []
        self.id = ASTNode.nuniq
        ASTNode.nuniq += 1
        self.attr = {}


class Tensor(ASTNode):
    def __init__(self, name, size: list | tuple, dtype='float', fix_size=[], is_arg=True):
        super().__init__()

        self.is_arg = is_arg
        self.name = name
        self.ref_size = []
        for s in size:
            if is_int_var(s):
                self.ref_size.append(s)
            elif type(s) == int:
                self.ref_size.append(Const(s, 'int'))
            else:
                raise TypeError('tensor dimensions must be int or a scalar int variable')
        self.fix_size = []
        for s in fix_size:
            if is_int_var(s):
                self.fix_size.append(s)
            elif type(s) == int:
                self.fix_size.append(Const(s, 'int'))
            else:
                raise TypeError('tensor dimensions must be int or a scalar int variable')
        self.dtype = dtype

    def __sub__(self, other):
        return TensorOp('sub', self, other)

    def __add__(self, other):
        return TensorOp('add', self, other)

    def __mul__(self, other):
        return TensorOp('mul', self, other)

    def __truediv__(self, other):
        return TensorOp('truediv', self, other)

    def __floordiv__(self, other):
        return TensorOp('floordiv', self, other)

    def __matmul__(self, other):
        return TensorOp('einsum', self, other, 'ij,jk->ik')

    def __getitem__(self, idx):
        assert isinstance(idx, (int, slice, Tensor, tuple))
        return TensorOp('index', self, idx)


    def get_member(self, name):
        assert hasattr(self, name)
        return TensorOp('getmember', self, name)


    def apply(self, func, axis=0, out_ofs=None, cond=None):
        assert callable(func)
        return TensorOp('apply', func, self, axis, out_ofs, cond)

    def reduce(self, func, init, axis=0):
        if callable(func) and callable(init):
            return TensorOp('reduce', self, func, init, axis)
        else:
            raise TypeError('reduce must use a callable function')

    def sum(self, axis=0):
        s1 = ''
        rs = ''
        for i in range(len(self._size())):
            s1 += chr(ord('i') + i)
            if i != axis:
                rs += chr(ord('i') + i)
        return einsum(f'{s1},->{rs}', self, None)

    def max(self, axis=0):
        func = lambda x, y: bigger(x, y)
        init = lambda x: setval(x, MIN_INT)
        return self.reduce(func, init, axis)

    def min(self, axis=0):
        func = lambda x, y: smaller(x, y)
        init = lambda x: setval(x, MAX_INT)
        return self.reduce(func, init, axis)

    def aggr(self, func, init, indices, axis=0, size=None):
        if callable(func) and callable(init):
            op = TensorOp('aggr', self, func, init, indices, axis, size)
            return op
        else:
            raise TypeError('aggr must use a callable function')

    def aggr_sum(self, indices, axis=0, size=None):
        func = lambda x, y: x + y
        init = lambda x: setval(x, 0)
        return self.aggr(func, init, indices, axis, size)

    def aggr_max(self, indices, axis=0, size=None):
        func = lambda x, y: bigger(x, y)
        init = lambda x: setval(x, MIN_INT)
        return self.aggr(func, init, indices, axis, size)

    def aggr_min(self, indices, axis=0, size=None):
        func = lambda x, y: smaller(x, y)
        init = lambda x: setval(x, MAX_INT)
        return self.aggr(func, init, indices, axis, size)

    def prefix_sum(self, axis=0, inclusive=True):
        assert type(axis) == int
        size = self._size()
        assert len(size) > 0
        data = self
        if not inclusive:
            size[axis] = size[axis] + 1
        out = res = Tensor(f'psum_{self.name[:3]}{self.id}', size, dtype=self.dtype)

        for i in range(axis):
            data = data[:]
            res = res[:]

        if inclusive:
            return setval(out, data[:] + res[-1:size[axis] - 1])
        else:
            return setval(out, data[-1:data._size()[axis]] + res[-1:size[axis] - 1])

    def _size(self):
        return self.ref_size

    def size(self):
        s = self._size()  # s is a list of int, Var, or Const
        if len(s) > 1:
            return Tensor(f'{self.name}_size', size=(len(s),), dtype='int', is_arg=False)
        elif len(s) == 1:
            if type(s[0]) == int:
                return Const(s, dtype='int')
            else:
                return s[0]
        else:
            return Const(0, dtype='int')

    def round(self):
        return TensorOp('round', self)

    def abs(self):
        return TensorOp('abs', self)

    def nbits(self):
        return TensorOp('nbits', self)

    def _gen_ir(self):
        return core.asg2ir.gen_ir(self)


class Var(Tensor):
    def __init__(self, name, dtype='int', is_arg=True):
        super().__init__(name, [], dtype, [], is_arg)


# const is var without name
class Const(Var):
    nconsts = 0

    def __init__(self, val, dtype):
        super().__init__(f'c{Const.nconsts}', dtype)
        Const.nconsts += 1
        # slice is considered constant because once the slice is created its start, stop, step cannot be reassigned
        # however, start, stop, step themselves can be variables
        if dtype == 'slice':
            if type(val.start) == int:
                start = Const(val.start, 'int')
            else:
                start = val.start
            if type(val.stop) == int:
                stop = Const(val.stop, 'int')
            else:
                stop = val.stop
            if type(val.step) == int:
                step = Const(val.step, 'int')
            else:
                step = val.step
            assert is_int_var(start)
            assert is_int_var(stop)
            assert is_int_var(step)
            self.val = slice(start, stop, step)
        else:
            self.val = val


def einsum(exp: str, tensor1, tensor2):
    return TensorOp('einsum', tensor1, tensor2, exp)


class TensorOp(Tensor):
    Types = func_op + list(arith_op.keys()) + math_op + cmp_op + other_op

    def __init__(self, op_type, *operators):
        assert op_type in TensorOp.Types

        # TODO: infer result data type
        self.operators = []
        for opr in operators:
            self.operators.append(opr)
            if isinstance(opr, ASTNode) and (not op_type in ('setval', 'inline')):
                opr.ref_by.append(self)

        if op_type in arith_op or op_type in cmp_op:
            dtype = operators[0].dtype
            if type(self.operators[0]) == int:
                self.operators[0] = Const(self.operators[0], 'int')
            elif type(operators[0]) == float:
                self.operators[0] = Const(self.operators[0], 'float')
            if type(self.operators[1]) == int:
                self.operators[1] = Const(self.operators[1], 'int')
            elif type(operators[1]) == float:
                self.operators[1] = Const(self.operators[1], 'float')
            assert prefix_match_size(self.operators[0]._size(), self.operators[1]._size())
            if (len(self.operators[0]._size()) > len(self.operators[1]._size())):
                ref_size = self.operators[0]._size()
            else:
                ref_size = self.operators[1]._size()
            fix_size = []

        elif op_type == 'einsum':
            dtype = operators[0].dtype
            exp = self.operators[2]
            inputs, output = exp.split('->')
            input1, input2 = inputs.split(',')
            op1_size = self.operators[0]._size()
            if self.operators[1] != None:
                op2_size = self.operators[1]._size()
            else:
                op2_size = []
            ref_size = []
            fix_size = []
            for i in output:
                pos1 = input1.find(i)
                if pos1 >= 0:
                    ref_size.append(op1_size[pos1])
                else:
                    pos2 = input2.find(i)
                    if pos2 >= 0:
                        ref_size.append(op2_size[pos2])
                    else:
                        raise IndexError('index not found!')

        elif op_type == 'index':
            dtype = operators[0].dtype
            if not type(operators[1]) in (list, tuple):
                self.operators[1] = [operators[1]]
            ref_size = self.operators[0]._size()[len(self.operators[1]):]
            fix_size = []

            new_size = []
            new_idx = []
            for i in range(len(self.operators[1])):
                idx = self.operators[1][i]
                if type(idx) == slice:
                    start = idx.start
                    if start == None:
                        start = Const(0, 'int')
                    elif type(start) == int:
                        start = Const(start, 'int')
                    stop = idx.stop
                    if stop == None:
                        stop = self.operators[0].ref_size[i]
                    elif type(stop) == int:
                        stop = Const(stop, 'int')
                    step = idx.step
                    if step == None:
                        step = Const(1, 'int')
                    elif type(step) == int:
                        step = Const(step, 'int')

                    idx = Const(slice(start, stop, step), 'slice')

                    if step.val == 1:
                        csize = [eval_const_expr(stop - start)]
                    else:
                        csize = [eval_const_expr(stop - start) // step]
                elif is_1d_tensor(idx):
                    csize = [idx.ref_size[0]]
                elif is_int_var(idx) or type(idx) == int:
                    csize = []
                    if type(idx) == int:
                        idx = Const(idx, 'int')
                else:
                    raise TypeError('index data type error!')

                new_size.extend(csize)
                new_idx.append(idx)

            ref_size[0:0] = new_size
            self.operators.pop()
            self.operators.extend(new_idx)

        elif op_type == 'apply':
            func = self.operators[0]
            self.nparams = len(inspect.signature(func).parameters)
            for i in range(self.nparams):
                axis = operators[1 + self.nparams + i]
                if type(axis) == int:
                    self.operators[1 + self.nparams + i] = Const(axis, 'int')

            data = []
            axis_size = self.operators[1]._size()[self.operators[1 + self.nparams].val]
            for i in range(1, 1 + self.nparams):
                data_size = self.operators[i]._size()
                axis = self.operators[self.nparams + i].val
                # every input item should have the same size as the primary axis size
                assert has_same_value(axis_size, data_size[axis])
                item_size = data_size[:axis] + data_size[axis + 1:]
                if (len(item_size) > 0):
                    item = Tensor(f'item_of_{self.operators[i].name}', item_size,
                                  self.operators[i].dtype, [], False)
                else:
                    item = Var(f'item_of_{self.operators[i].name}', self.operators[i].dtype, False)
                data.append(item)

            ret = self.operators[0](*data)
            dtype = ret.dtype
            out_ofs = self.operators[1 + 2 * self.nparams]
            if out_ofs == None:
                ref_size = [axis_size] + ret._size()
            else:
                ref_size = [out_ofs[axis_size]] + ret._size()[1:]
            fix_size = []

            self.operators.extend(data)
            self.operators.append(ret)

        elif op_type == 'reduce':
            assert type(self.operators[3]) == int
            axis = self.operators[3]
            self.operators[3] = Const(axis, 'int')
            ref_size = self.operators[0]._size()[:axis] + self.operators[0]._size()[axis + 1:]
            fix_size = []
            dtype = self.operators[0].dtype
            if (len(ref_size) > 0):
                item1 = Tensor(f'item1_of_{self.operators[0].name}', ref_size,
                               self.operators[0].dtype, [], False)
                item2 = Tensor(f'item2_of_{self.operators[0].name}', ref_size,
                               self.operators[0].dtype, [], False)
            else:
                item1 = Var(f'item1_of_{self.operators[0].name}', self.operators[0].dtype, False)
                item2 = Var(f'item2_of_{self.operators[0].name}', self.operators[0].dtype, False)

            self.operators.append(item1)
            self.operators.append(item2)
            self.operators.append(self.operators[1](item1, item2))

        elif op_type == 'aggr':
            dtype = operators[0].dtype
            assert is_1dint_tensor(self.operators[3])
            assert type(self.operators[4]) == int
            axis = self.operators[4]
            self.operators[4] = Const(axis, 'int')
            if self.operators[5] == None:
                self.operators[5] = self.operators[3].ref_size[0]
            else:
                assert is_int_var(self.operators[5])
                if type(self.operators[5]) == int:
                    self.operators[5] = Const(self.operators[5], 'int')
            ref_size = [self.operators[5]] + self.operators[0]._size()[:axis] + self.operators[0]._size()[axis + 1:]
            fix_size = []
            if (len(ref_size) > 1):
                item1 = Tensor(f'item1_of_{self.operators[0].name}', ref_size[1:],
                               self.operators[0].dtype, [], False)
                item2 = Tensor(f'item2_of_{self.operators[0].name}', ref_size[1:],
                               self.operators[0].dtype, [], False)
            else:
                item1 = Var(f'item1_of_{self.operators[0].name}', self.operators[0].dtype, False)
                item2 = Var(f'item2_of_{self.operators[0].name}', self.operators[0].dtype, False)
            self.operators.append(item1)
            self.operators.append(item2)
            self.operators.append(self.operators[1](item1, item2))

        elif op_type in math_op:
            dtype = self.operators[0].dtype
            ref_size = self.operators[0]._size()
            fix_size = []
            if op_type == 'round':
                dtype = 'int'
            elif op_type == 'abs':
                dtype = self.operators[0].dtype

        elif op_type == 'setval':
            dtype = self.operators[0].dtype
            ref_size = self.operators[0]._size()
            fix_size = []

            if (is_scalar(self.operators[1])):
                if type(self.operators[1]) == int:
                    self.operators[1] = Const(self.operators[1], 'int')
                elif type(self.operators[1]) == float:
                    self.operators[1] = Const(self.operators[1], 'float')

            else:
                assert self.operators[0].dtype == self.operators[1].dtype

        elif op_type == 'inline':
            dtype = self.operators[1][1].dtype
            ref_size = self.operators[1][1]._size()
            fix_size = []

            keyvalue = []
            src = self.operators[0]
            for op in self.operators[1:]:
                keyvalue.append(op[0])
                keyvalue.append(op[1])
            self.operators.clear()
            self.operators.append(src)
            self.operators.extend(keyvalue)

        elif op_type == 'getmember':
            member = getattr(self.operators[0], self.operators[1])
            dtype = member.dtype
            ref_size = member._size()
            fix_size = []





        name = f'{op_type}_' + '_'.join([op.name if hasattr(op, 'name') else '' for op in self.operators])

        super().__init__(name, ref_size, dtype, fix_size)

        self.op_type = op_type

        # call the init function for reduce and aggr
        if self.op_type in ('reduce', 'aggr'):
            self.operators[2] = self.operators[2](self)

        if self.op_type == 'apply':
            cond = self.operators[2 + 2 * self.nparams]
            if cond != None:
                assert is_1d_tensor(cond)
                assert has_same_value(axis_size, cond._size()[0])
                self.counter = setval(Var(f'counter{self.id}', dtype='int', is_arg=False), 0)
                self.operators.append(self.counter)
            else:
                self.operators.append(None)

        self.input_orders = [[] for o in self.operators]



