from codegen import *
import run
import torch
from helpers import new_op, ASGTraversal
from transform.fuse import merge_loops, fuser, basic_rule, fuse_operators
from core.asg import *
from transform import parallelize, split

@new_op
def bvv(a, b):
    return apply(lambda x, y: einsum('i,i->', x, y), (a, b))

@new_op
def bsv(a, b):
    return apply(lambda x, y: x * y, (a, b))

@new_op
def bvm(a, b):
    return apply(lambda x, y: einsum('i,ij->j', x, y), (a, b))

@new_op
def bov(a, b):
    return apply(lambda x, y: einsum('i,j->ij', x, y), (a, b))


def fuse_rule(node, res):
    if type(node) == TensorOp and 'bvv' in node.attr:
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if node.operators[1].op_type in elementwise_op or 'bvm' in node.operators[1].attr:
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op or 'bvm' in node.operators[2].attr:
                fuse_operators(node, node.input_orders[2], node.operators[2])

    if type(node) == TensorOp and 'bsv' in node.attr:
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if 'bvv' in node.operators[1].attr:
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[2], node.operators[2])

    if type(node) == TensorOp and 'bov' in node.attr:
        if type(node.operators[1]) == TensorOp and len(node.operators[1].ref_by) == 1:
            if node.operators[1].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[1], node.operators[1])

        if type(node.operators[2]) == TensorOp and len(node.operators[2].ref_by) == 1:
            if node.operators[2].op_type in elementwise_op:
                fuse_operators(node, node.input_orders[2], node.operators[2])


f = fuser()
f.register(basic_rule)
f.register(fuse_rule)
fuse = f.fuse

def tiling(asg, C, D):
    def action(node, res):
        if not 'bov' in node.attr and len(node.compute) > 0:
            split.split(node, C, 0)
            split.split(node, D, 1)

    t = ASGTraversal(action)
    t(asg)
    return asg


def transE():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    res = vh - vt + vr

    ir = fuse(res._gen_ir())
    tiling(ir, 16, 128)
    parallelize.parallelize(ir, [80, 8, 32])
    code = codegen.cpu.print_cpp(ir)
    print(code)


def transH():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    Pemb = Tensor('Pemb', (nedges, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]
    vp = Pemb[r]

    # TODO: if there are redundant computation, is fusion always beneficial
    res = vh - vt + vr - bsv(bvv(vp, vh - vt), vp)
    ir = fuse(res._gen_ir())
    tiling(ir, 16, 128)
    parallelize.parallelize(ir, [80, 8, 32])

    code = codegen.cpu.print_cpp(ir)
    print(code)


def transR():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    Proj = Tensor('Proj', (nedges, dim, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]
    vr = Remb[r]

    res = bvm(vh - vt, mr) + vr

    ir = fuse(res._gen_ir())
    tiling(ir, 16, 128)
    parallelize.parallelize(ir, [80, 8, 32])

    code = codegen.cpu.print_cpp(ir)
    
    print(code)


    

def transF():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]
    
    # alpha = Const(val=2, dtype='float')
    # alpha = Batch(alpha)
    # alpha = 2

    res = bvv(vh, vt) - bvv(vh - vt, vr)
    
    code = codegen.cpu.print_cpp(parallelize.parallelize(tiling(fuse(res._gen_ir()), 16, 128), [80, 8, 32]))
    # ast = res._gen_ir()
    # fuse_operators(ast)
    # tile_loop(ast)
    # parallel(ast)
    # add_smem(ast)
    # code = codegen.gpu.print_cuda(ast)
    print(code)
    # h = torch.randint(0, 9999, (4096, )).cuda(0)
    # r = torch.randint(0, 100, (4096, )).cuda(0)
    # t = torch.randint(0, 9999, (4096, )).cuda(0)
    # eemb = torch.rand((9999, 512)).cuda(0)
    # remb = torch.rand((100, 512)).cuda(0)

    # y = torch.einsum('ab,ab->a', eemb[h], eemb[t]) - torch.einsum('ab,ab->a',(eemb[h] - eemb[t]), remb[r])
    # print(y)
    
    # x = run.gpu.compile_and_run(code, 4096, 512, 0, eemb, h,t, 0, remb, r)
    # print(x)

def RESCAL():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Proj = Tensor('Proj', (nedges, dim, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    mr = Proj[r]

    res = bvv(bvm(vh, mr), vt)

    code = codegen.cpu.print_cpp(parallelize.parallelize(tiling(fuse(res._gen_ir()), 16, 128), [80, 8, 32]))
    
    # ast = res._gen_ir()
    
    # fuse_operators(ast)
    # tile_loop(ast)
    # parallel(ast)
    # add_smem(ast)
    # traversal call funcs to transform ir
    # code = codegen.cpu.print_cpp(ast)
    # code = codegen.gpu.print_cuda(ast)
    print(code)

    # h = torch.randint(0, 9999, (4096, )).cuda(0)
    # r = torch.randint(0, 100, (4096, )).cuda(0)
    # t = torch.randint(0, 9999, (4096, )).cuda(0)
    # eemb = torch.rand((9999, 512)).cuda(0)
    # remb = torch.rand((100, 512, 512)).cuda(0)

    # y = torch.einsum('ab,ab->a', torch.einsum('ab,abc->ac', eemb[h], remb[r]), eemb[t])
    # print(y, y.shape)
    
    # x = run.gpu.compile_and_run(code, 4096, 512, 0, eemb, h, 0, remb, r, t)
    # print(x)

def test():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    Proj = Tensor('Proj', (nedges, dim, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Eemb[h]
    vt = Eemb[t]
    vr = Remb[r]

    res = bov(vh+vr, vt-vr)

    ir = fuse(res._gen_ir())

    parallelize.t(ir)

    code = codegen.cpu.print_cpp(ir)
    print(code)
    # h = torch.randint(0, 9999, (4096, )).cuda(0)
    # r = torch.randint(0, 100, (4096, )).cuda(0)
    # t = torch.randint(0, 9999, (4096, )).cuda(0)
    # eemb = torch.rand((9999, 512)).cuda(0)
    # remb = torch.rand((100, 512)).cuda(0)
    #
    # y = torch.einsum('ab,ac->abc', eemb[h] + remb[r], eemb[t] - remb[r])
    # print(y)
    #
    # x = run.gpu.compile_and_run(code, 4096, 512, 0, eemb, h, 0, remb, r, t)
    # print(x)

if __name__ == "__main__":
    transE() # success
    transH() # success
    # transR() # success
    # transF() # success
    # RESCAL() # success
    # test()