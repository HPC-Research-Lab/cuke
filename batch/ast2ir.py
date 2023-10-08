import sys
sys.path.append('/data/backed_up/lihhu/CUKE/cuke')
from batch.ast import *
from core.ast2ir import * 

def is_bvec(t):
    return isinstance(t, Batch) and t.item_type == 'vec'

def is_bmat(t):
    return isinstance(t, Batch) and t.item_type == 'mat'

def is_bscal(t):
    return isinstance(t, Batch) and t.item_type == 'scal'

def gen_ir(node):
    assert isinstance(node, Batch)
    if node.eval:
        return node

    if type(node) == Batch:
        node.base._gen_ir()
        node.eval = node.base.eval
    elif type(node) == BatchOp:
        if node.op_type in core.ast.op_mapping:
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            node.base._gen_ir()
            # print(node.operators[1].eval, node.op_type, node.base.compute)
            node.eval = node.base.eval
            node.decl = node.base.decl[:]
            node.compute = node.base.compute[:]
            node.base.decl.clear()
            node.base.compute.clear()

        elif node.op_type == 'vec_mul_vec':
            assert is_bvec(node.operators[0]) and is_bvec(node.operators[1])
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            size = helpers.get_ir_of_size(node._size())
            node.base.eval = node.eval = Ndarray(node.dtype, size)
            node.decl = [Decl(node.eval)]
            pre_loop = Loop(0, node.eval.size[0], 1, [])
            node.compute = [pre_loop]
            lhs = bind(node.operators[0].eval, pre_loop.iterate)
            rhs = bind(node.operators[1].eval, pre_loop.iterate)
            res = bind(node.eval, pre_loop.iterate)
            inner_loop =  Loop(0, node.operators[0].eval.size[1], 1, [])
            pre_loop.body.append(inner_loop)
            lhs = bind(lhs, inner_loop.iterate)
            rhs = bind(rhs, inner_loop.iterate)
            assign = Assignment(res, Expr(lhs, rhs, '*'), '+')
            inner_loop.body.append(assign)

        elif node.op_type == 'scal_mul_vec':
            assert is_bscal(node.operators[0]) and is_bvec(node.operators[1])
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            size = helpers.get_ir_of_size(node._size())
            node.base.eval = node.eval = Ndarray(node.dtype, size)
            node.decl = [Decl(node.eval)]
            pre_loop = Loop(0, node.eval.size[0], 1, [])
            node.compute = [pre_loop]
            lhs = bind(node.operators[0].eval, pre_loop.iterate)
            rhs = bind(node.operators[1].eval, pre_loop.iterate)
            res = bind(node.eval, pre_loop.iterate)
            inner_loop =  Loop(0, node.eval.size[1], 1, [])
            pre_loop.body.append(inner_loop)
            rhs = bind(rhs, inner_loop.iterate)
            res = bind(res, inner_loop.iterate)

            assign = Assignment(res, Expr(lhs, rhs, '*'))
            inner_loop.body.append(assign)

        elif node.op_type == 'vec_mul_mat':
            assert is_bvec(node.operators[0]) and is_bmat(node.operators[1])
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            size = helpers.get_ir_of_size(node._size())
            node.base.eval = node.eval = Ndarray(node.dtype, size)
            node.decl = [Decl(node.eval)]
            pre_loop = Loop(0, node.eval.size[0], 1, [])
            node.compute = [pre_loop]
            lhs = bind(node.operators[0].eval, pre_loop.iterate)
            rhs = bind(node.operators[1].eval, pre_loop.iterate)
            res = bind(node.eval, pre_loop.iterate)
            loop1 = Loop(0, node.eval.size[1], 1, [])
            pre_loop.body.append(loop1)
            res = bind(res, loop1.iterate)
            loop2 = Loop(0, node.operators[0].eval.size[1], 1, [])
            loop1.body.append(loop2)
            lhs = bind(lhs, loop2.iterate)
            rhs = bind(rhs, loop2.iterate)
            rhs = bind(rhs, loop1.iterate)

            assign = Assignment(res, Expr(lhs, rhs, '*'), '+')
            loop2.body.append(assign)
        
        elif node.op_type == 'vec_outer_vec':
            assert is_bvec(node.operators[0]) and is_bvec(node.operators[1])
            node.operators[0]._gen_ir()
            node.operators[1]._gen_ir()
            size = helpers.get_ir_of_size(node._size())
            node.base.eval = node.eval = Ndarray(node.dtype, size)
            node.decl = [Decl(node.eval)]
            pre_loop = Loop(0, node.eval.size[0], 1, [])
            node.compute = [pre_loop]
            lhs = bind(node.operators[0].eval, pre_loop.iterate)
            rhs = bind(node.operators[1].eval, pre_loop.iterate)
            res = bind(node.eval, pre_loop.iterate)
            loop1 = Loop(0, node.eval.size[1], 1, [])
            pre_loop.body.append(loop1)
            lhs = bind(lhs, loop1.iterate)
            res = bind(res, loop1.iterate)
            loop2 = Loop(0, node.eval.size[2], 1, [])
            loop1.body.append(loop2)
            
            rhs = bind(rhs, loop2.iterate)
            res = bind(res, loop2.iterate)

            assign = Assignment(res, Expr(lhs, rhs, '*'))
            loop2.body.append(assign)

    return node

