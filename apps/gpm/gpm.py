import codegen.cpu
import transform.fuse
from core.asg import *
from helpers import new_op


class Set(Tensor):
    def __init__(self, storage):
        self.storage = storage
        if hasattr(storage, 'counter'):
            self.length = self.storage.get_member('counter')
        else:
            self.length = storage._size()[0]
        self.attr = self.storage.attr


    def _gen_ir(self):
        return self.length._gen_ir()


def is_in(x, li):
    src = ('''for (int i=0; i<LSIZE; i++) {
            F = true;
           }''')
    found = Var('found', dtype='int')
    return inline(src, ('F', found), ('X', x), ('LI', li), ('LSIZE', li._size()[0]))

def intersect(a: Set, b: Set):
    sa = a.to_tensor()
    sb = b.to_tensor()
    c = sa.apply(lambda x: is_in(x, sb))
    return Set(sa.apply(lambda x: x, cond=c))


class Graph:
    def __init__(self, rowptr, colidx):
        self.rowptr = rowptr
        self.colidx = colidx

    def get_neighbor(self, v):
        return Set(self.colidx[self.rowptr[v]:self.rowptr[v + 1]])



def test1():
    A = Set(Tensor('A', (10, )))
    B = Set(Tensor('B', (20, )))
    res = intersect(A, B)
    ir = res._gen_ir()
    f = transform.fuse.fuser()
    f.register(transform.fuse.basic_rule)
    ir = f.fuse(ir)
    code = codegen.cpu.print_cpp(ir)
    print(code)



def test2():
    A = Set(Tensor('A', (10, )))
    B = Set(Tensor('B', (20, )))
    C = Set(Tensor('C', (30, )))
    res = intersect(intersect(A, B), C)
    ir = res._gen_ir()
    # f = transform.fuse.fuser()
    # f.register(transform.fuse.basic_rule)
    # ir = f.fuse(ir)
    code = codegen.cpu.print_cpp(ir)
    print(code)



def test3():
    nnodes = 100
    nedges = 1000
    edges = Tensor('edges', (nedges, 2), dtype='int')
    rowptr = Tensor('rowptr', (nnodes + 1, ), dtype='int')
    colidx = Tensor('colidx', (nedges, ), dtype='int')
    g = Graph(rowptr, colidx)
    f = lambda e: intersect(g.get_neighbor(e[0]), g.get_neighbor(e[1])).length
    res = edges.apply(f)
    res = res.sum()
    ir = res._gen_ir()
    f = transform.fuse.fuser()
    f.register(transform.fuse.basic_rule)
    ir = f.fuse(ir)
    code = codegen.cpu.print_cpp(ir)
    print(code)



if __name__ == "__main__":
    # test1()
    # test2()
    test3()