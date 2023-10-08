import core.ast
from core.ast import *
import batch

class Batch(ASTNode):
    def __init__(self, base):
        super().__init__()
        size = base._size()

        if type(base) == TensorOp and base.ref_count >= 1:
            self.base = copy.copy(base)
            self.base.eval = None
            self.base.decl.clear()
            self.base.compute.clear()
            self.base.ref_count = 1
        else:
            self.base = base
            self.base.ref_count += 1

        
        if (len(size) == 2):
            self.item_type = 'vec'
            self.dim = size[1]
            self.batch_size = size[0]
        elif len(size) == 3:
            self.item_type = 'mat'
            self.dim1 = size[1]
            self.dim2 = size[2]
            self.batch_size = size[0]
        elif len(size) == 1:
            self.item_type = 'scal'
            self.batch_size = size[0]
        elif len(size) == 0:
            self.item_type = 'const'
            self.batch_size = 0
        else:
            raise TypeError('Batch item type not supported')

        self.dtype = self.base.dtype
        self.name = self.base.name

    def _size(self):
        return self.base._size()


    def _gen_ir(self):
        return batch.ast2ir.gen_ir(self)

    def __sub__(self, other):
        return BatchOp('sub', self, other)

    def __add__(self, other):
        return BatchOp('add', self, other)

    def __mul__(self, other):
        return BatchOp('mul', self, other)

    def __floordiv__(self, other):
        return BatchOp('floordiv', self, other)

    def __truediv__(self, other):
        return BatchOp('truediv', self, other)

def bvv(v1: Batch, v2: Batch):
    assert v1.item_type == 'vec' and v2.item_type == 'vec'
    assert v1.dim == v2.dim
    return BatchOp('vec_mul_vec', v1, v2)

def bsv(v1: Batch, v2: Batch):
    assert v1.item_type == 'scal' and v2.item_type == 'vec'
    return BatchOp('scal_mul_vec', v1, v2)

def bvm(v1: Batch, v2: Batch):
    assert v1.item_type == 'vec' and v2.item_type == 'mat'
    assert v1.dim == v2.dim1
    return BatchOp('vec_mul_mat', v1, v2)

def bov(v1: Batch, v2: Batch):
    assert v1.item_type == 'vec' and v2.item_type == 'vec'
    return BatchOp('vec_outer_vec', v1, v2)

class BatchOp(Batch):
    Types = ['scal_mul_vec', 'vec_mul_vec', 'vec_mul_mat', 'vec_outer_vec'] + list(core.ast.op_mapping.keys())

    def __init__(self, op_type, *operators):
        assert op_type in BatchOp.Types
        # TODO: infer result data type
        dtype = operators[0].dtype

        self.operators = []
        for opr in operators:
            if isinstance(opr, Batch) and type(opr.base) == TensorOp and opr.base.op_type == 'index' and opr.ref_count >= 1:
                new_opr = copy.copy(opr)
                new_opr.ref_count = 1
                new_opr.eval = None
                new_opr.base = copy.copy(opr.base)
                new_opr.base.ref_count = 1
                new_opr.base.eval = None
                new_opr.base.decl.clear()
                new_opr.base.compute.clear()
                self.operators.append(new_opr)
            else:
                self.operators.append(opr)
                if isinstance(opr, ASTNode):
                    opr.ref_count += 1
                    opr.base.ref_count += 1

        name = f'{op_type}_' + '_'.join([op.name if hasattr(op, 'name') else '' for op in self.operators])

        if op_type in core.ast.op_mapping:
            match op_type:
                case 'add':
                    res = self.operators[0].base + self.operators[1].base
                case 'sub':
                    res = self.operators[0].base - self.operators[1].base
                case 'mul':
                    res = self.operators[0].base * self.operators[1].base
                case 'floordiv':
                    res = self.operators[0].base // self.operators[1].base
                case 'truediv':
                    res = self.operators[0].base / self.operators[1].base
            super().__init__(res)

        elif op_type == 'vec_mul_vec':
            bsize = self.operators[0].batch_size
            res = Tensor(name, (bsize, ), dtype)
            super().__init__(res)

        elif op_type == 'scal_mul_vec':
            bsize = self.operators[0].batch_size
            dim = self.operators[1].dim
            res = Tensor(name, (bsize, dim), dtype)
            super().__init__(res)

        elif op_type == 'vec_mul_mat':
            bsize = self.operators[0].batch_size
            dim = self.operators[1].dim2
            res = Tensor(name, (bsize, dim), dtype)
            super().__init__(res)

        elif op_type == 'vec_outer_vec':
            bsize = self.operators[0].batch_size
            res = Tensor(name, (bsize, self.operators[0].dim, self.operators[1].dim ), dtype)
            super().__init__(res)

        else: # TODO: complete other ops
            pass

        self.op_type = op_type

