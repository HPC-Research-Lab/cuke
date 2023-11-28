from core.asg import *
import cset
# from cset.ast2ir import *

class BinarySearch():
    def __init__(self, target_set, negative=False):
        self.target_set = target_set
        self.negative = negative
    def __call__(self, item):
        return SetOp('binary_search', item, self.target_set, self.negative)

class SmallerThan():
    def __init__(self, val):
        self.val = val
    def __call__(self, item):
        return SetOp('smaller', item, self.val)

class Edgefilter():
    def __init__(self, rowptr, threshold):
        self.rowptr = rowptr
        self.threshold = threshold
    def __call__(self, item):
        degree = self.rowptr[item[0]+1] - self.rowptr[item[0]]
        return SetOp('smaller', self.threshold, degree)

class InlineCode():
    def __init__(self, target_set):
        self.target_set = target_set
    def __call__(self, item):
        return SetOp('inline_code', item, self.target_set)

class FunctionCall():
    def __init__(self, target_set):
        self.target_set = target_set
    def __call__(self, item):
        return SetOp('function_call', item, self.target_set)

def PartialEdge(item):
    return SetOp('smaller', item[1], item[0])

class Set(ASTNode):
    capacity = 10000
    nelem_id = 0

    def __init__(self, storage):
        super().__init__()
        if isinstance(storage, Tensor):
            self.storage = storage
        else:
            raise TypeError("init val of set must be a tensor")

        self.name = f'set_{self.storage.name}'
        self.nelem = Var(f'nelem{Set.nelem_id}', dtype='int', is_arg=False)
        Set.nelem_id+=1
        
        self.dtype = storage.dtype
        self.is_arg = False
         
    def _tensor_size(self):
        return self.storage._size()
    def _gen_ir(self):
        return cset.ast2ir.gen_ir(self)

    def apply(self, func, init=None, k_capacity=10000):
        if callable(func):
            op = SetOp('apply', self, func, init, **{'capacity': k_capacity})
            return op
        else:
            raise TypeError('must apply a callable function')

    def filter(self, cond,  k_capacity=10000):
        if callable(cond):
            op = SetOp('filter', self, cond, **{'capacity': k_capacity})
            return op
        else:
            raise TypeError('must apply a callable condition')
    
    def intersection(self, other, cond, k_capacity=10000):
        if callable(cond):
            op = SetOp('intersection', self, cond(other), **{'capacity': k_capacity})
            return op
        else:
            raise TypeError('must apply a callable condition')
    
    def difference(self, other, cond,  k_capacity=10000):
        if callable(cond):
            op = SetOp('difference', self, cond(other), **{'capacity': k_capacity})
            return op
        else:
            raise TypeError('must apply a callable condition')
    
    def increment(self, val):
        return SetOp('increment', self, val)
    
    def retval(self, val):
        return SetOp('retval', self, val)
        
    
class SetOp(Set):
    ids = {'apply': 0, 'filter':0, 'intersection':0, 'difference':0, \
            'binary_search':0, 'merge_search': 0, 'smaller':0, \
            'add':0, 'setval':0, 'increment':0, 'retval':0, 'inline_code':0, 'function_call':0}

    def __init__(self, op_type, *operators, **config):
        
        self.operators = list(operators)
        self.config = config
        self.op_type = op_type

        tensor_name = f'{op_type}{SetOp.ids[op_type]}'
        SetOp.ids[op_type]+=1

        if op_type == 'apply' or op_type == 'filter' or op_type == 'intersection' or op_type == 'difference':
            input_set = self.operators[0]
            item_size = input_set._tensor_size()[1:]
            if (len(item_size) > 0):
                item = Tensor(f'item_{input_set.name}', item_size, input_set.dtype, is_arg=False)
            else:
                item = Var(f'item_{input_set.name}', input_set.dtype, False)
            self.operators.append(item)
            input_func = self.operators[1]
            func_ast =  input_func(item)
            self.operators[1] =func_ast
            
            if op_type == 'apply':
                input_init = self.operators[2]
                self.operators[2] = input_init() if input_init!=None else None
                super().__init__(Tensor(tensor_name, [config['capacity']] * (len(func_ast._tensor_size())+1), dtype = func_ast.dtype, is_arg=False))
            else:
                super().__init__(Tensor(tensor_name, [config['capacity']] + item_size, dtype = input_set.dtype, is_arg=False))
                        
        elif op_type == 'binary_search' or op_type == 'smaller':
            for i in range(0, len(self.operators)):
                if isinstance(self.operators[i], (int, float)):
                    self.operators[i] = Const(self.operators[i], type(self.operators[i]))
            super().__init__(Var(tensor_name, 'int', False))
        
        elif op_type == 'inline_code' or op_type == 'function_call':
            for i in range(0, len(self.operators)):
                if isinstance(self.operators[i], (int, float)):
                    self.operators[i] = Const(self.operators[i], type(self.operators[i]))
            super().__init__(Var(tensor_name, 'int', False))
                
        elif op_type == 'increment':
            self.operators[1] = Const(self.operators[1], 'int')
            super().__init__(Var(tensor_name, 'int', False))
        
        elif op_type == 'setval' or op_type == 'retval':
            super().__init__(Var(tensor_name, 'int', False))




        
