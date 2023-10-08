import sys
sys.path.append('/data/backed_up/lihhu/CUKE/cuke')
from codegen import *
from batch.ast import *
from batch.opt.fusion_rules import *
from batch.opt.parallelism import *

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
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    vr = Batch(Remb[r])

    res = vh - vt + vr

    # code = codegen.cpu.print_cpp(res._gen_ir())

    ast = res._gen_ir()
    fuse_operators(ast)
    parallel(ast)
    # code = codegen.cpu.print_cpp(ast)
    code = codegen.gpu.print_cpp(ast)
    print(code)

def transH():
    nnodes = Var('nnodes')
    nedges = Var('nedges')
    dim = Var('dim')
    batch_size = Var('batch_size')
    Eemb = Tensor('Eemb', (nnodes, dim))
    Remb = Tensor('Remb', (nedges, dim))
    h = Tensor('h', (batch_size, ), dtype='int')
    t = Tensor('t', (batch_size, ), dtype='int')
    r = Tensor('r', (batch_size, ), dtype='int')
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    vr = Batch(Remb[r])

    res = vh - vt + vr - bsv(bvv(vr, vh - vt), vr)

    code = codegen.cpu.print_cpp(res._gen_ir())

    # print(code)
    ast = res._gen_ir()
    
    
    fuse_operators(ast)
    parallel(ast)
    # code = codegen.cpu.print_cpp(ast)
    code = codegen.gpu.print_cpp(ast)
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
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    mr = Batch(Proj[r])
    vr = Batch(Remb[r])

    res = bvm(vh -vt, mr) + vr

    code = codegen.cpu.print_cpp(res._gen_ir())
    ast = res._gen_ir()
    fuse_operators(ast)
    
    
    code = codegen.cpu.print_cpp(ast)

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
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    vr = Batch(Remb[r])
    
    alpha = Const(val=2, dtype='float')
    alpha = Batch(alpha)
    # alpha = 2

    res = bvv(vh, vt)  - bvv(vh - vt, vr)
    
    code = codegen.cpu.print_cpp(res._gen_ir())
    ast = res._gen_ir()
    fuse_operators(ast)
    
    
    code = codegen.cpu.print_cpp(ast)

    print(code)

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
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    mr = Batch(Proj[r])

    res = bvv(bvm(vh, mr), vt)

    code = codegen.cpu.print_cpp(res._gen_ir())
    
    ast = res._gen_ir()
    
    fuse_operators(ast)
    
    code = codegen.cpu.print_cpp(ast)

    print(code)

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
    vh = Batch(Eemb[h])
    vt = Batch(Eemb[t])
    vr = Batch(Remb[r])
    vrr = Batch(Remb[r])
    proj_m = Batch(Proj[r])
    proj_h = Batch(Proj[h])

    # res = vh - vt + vr - vrr
    # res = bov(vh+vr, vt-vr)
    res = vh+vt - (proj_m+ proj_h)

    # code = codegen.cpu.print_cpp(res._gen_ir())
    # print(code)
    ast = res._gen_ir()
    
    fuse_operators(ast)
    code = codegen.cpu.print_cpp(ast)
    print(code)

    # t = codegen.cpu.print_cpp(ast)
    # print(t)

if __name__ == "__main__":
    # test()
    # transE()
    transH()
    # transR()
    # transF()
    # RESCAL()
