from asg import *
from cset.ast2ir import *


#self.nelem = Var(f'{self.name}_nelem', 'int', False)
# if type(self) == Set:
#     #helpers.get_ir_of_size(self.storage._size())
#     # self.nelem = Const(1, 'int')
#     # for s in self.storage._size():
#     #     self.nelem = self.nelem * s
# else:
#     self.nelem = Var(f'{self.name}_nelem', 'int', False)

# self.nelem = self.storage._size()

# def __getitem__(self, idx):
#     if isinstance(idx, (int, slice, Tensor, Set)):
#         return SetOp('index', self, idx)
#     else:
#         raise TypeError('invalid index type')

# def intersect(self, other):
#     assert type(other) == Set
#     return SetOp('intersect', self, other)

# class IsIn:
# 	def __init__(self, B:Set):
	
# 	#A中的Item
# 	def __call__(item):
# 		return SetOp('Search', item, B)

    # elif op_type == 'filter':
    #     input_storage = self.operators[0].storage
    #     input_storage_name = input_storage.name
    #     input_storage_size = input_storage._size()
    #     input_storage_dtype = input_storage.dtype
    #     assert len(input_storage_size)==1
    #     assert callable(self.operators[1])

    #     item = Var(f'item_of_{input_storage_name}', input_storage_dtype, False)
    #     self.operators.append(item)
    #     # cond_set_op = self.operators[1](item)
    #     # self.operators[1] =  cond_set_op   
        

    #     res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
    #     super().__init__(Tensor(res_storage_name, [self.capacity], dtype = 'int', is_arg=False))

class BinarySearch():
    def __init__(self, target_set):
        self.target_set = target_set
    def __call__(self, item):
        return SetOp('search', self.target_set, item)

class Difference():
    def __init__(self, target_set):
        self.target_set = target_set
    def __call__(self, item):
        return SetOp('difference', self.target_set, item)

class SmallerThan():
    def __init__(self, val):
        self.val = val
    def __call__(self, item):
        return SetOp('smaller', self.val, item)

class Set(ASTNode):
    capacity = 2000

    def __init__(self, storage):
        super().__init__()
        if isinstance(storage, Tensor):
            self.storage = storage
        else:
            raise TypeError("init val of set must be a tensor")

        self.name = f'set_{self.storage.name}'
        self.nelem = self.storage._size()
        self.dtype = storage.dtype
        self.is_arg = False
    
    def _size(self):
        return self.storage._size()
    def _gen_ir(self):
        return cset.ast2ir.gen_ir(self)

    def apply(self, func, cond, axis=0, decl_ret=True):
        op = SetOp('apply', self, func, cond, axis, decl_ret)
        return op

    def applyfunc(self, func, axis=0, decl_ret=False):
        if callable(func):
            op = self.apply(func, cond=None, axis=axis, decl_ret=decl_ret)
            return op
        else:
             raise TypeError('must apply a callable function')

    def filter(self, cond, axis=0):
        if callable(cond):
            op = self.apply(func=None, cond=cond, axis=axis,  decl_ret=True)
            return op
        else:
            raise TypeError('must apply a callable condition')
    
    def intersect(self, other):
        assert len(self._size())==1
        assert len(other._size())==1
        op = self.filter(BinarySearch(other))
        return op
    
    def difference(self, other):
        assert len(self._size())==1
        assert len(other._size())==1
        op = self.filter(Difference(other))
        return op
    
    def num_elem(self):
        op = SetOp('nelem', self)
        return op
    def sum(self):
        op = SetOp('sum', self)
        return op
    def addone(self):
        op = SetOp('addone', self)
        return op 
    def ret_val(self, val):
        op = SetOp('ret_val', self, val)
        return op 
    

class SetOp(Set):

    def __init__(self, op_type, *operators):
        
        self.operators = list(operators)
        self.op_type = op_type

        if op_type == 'apply':
            input_set = self.operators[0]
            input_func = self.operators[1]
            input_cond = self.operators[2]
            input_axis = self.operators[3]
            input_decl_ret = self.operators[4]

            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            input_storage_size = input_storage._size()
            input_storage_dtype = input_storage.dtype
            
            axis =  self.operators[3]
            self.operators[3] = Const(axis, 'int')
            data_size = input_storage_size
            item_size = data_size[:axis] + data_size[axis + 1:]
            if (len(item_size) > 0):
                item = Tensor(f'item_of_{input_storage_name}', item_size,
                              input_storage_dtype, [], False)
            else:
                item = Var(f'item_of_{input_storage_name}', input_storage_dtype, False)
            self.operators.append(item)

            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Tensor(res_storage_name, [self.capacity], dtype = 'int', is_arg=False))
        
        elif op_type == 'search':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Var(f'res_of_{res_storage_name}', 'int', False))
        
        elif op_type == 'difference':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Var(f'res_of_{res_storage_name}', 'int', False))
        
        elif op_type == 'smaller':
            val_name = self.operators[0].name
            res_name = f'{op_type}_than_' + '_'.join([val_name])
            super().__init__(Var(f'res_of_{res_name}', 'int', False))

        elif op_type == 'nelem':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Var(f'nelem_of_{res_storage_name}', 'int', False))
        
        elif op_type == 'sum':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Var(f'sum_of_{res_storage_name}', 'int', False))
        
        elif op_type == 'addone':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            super().__init__(Var(f'addone_of_{res_storage_name}', 'int', False))
       
        elif op_type == 'ret_val':
            input_storage = self.operators[0].storage
            input_storage_name = input_storage.name
            res_storage_name = f'{op_type}_' + '_'.join([input_storage_name])
            return_val = self.operators[1]
            super().__init__(Var(f'ret_val_of_{res_storage_name}', 'int', False))

        
