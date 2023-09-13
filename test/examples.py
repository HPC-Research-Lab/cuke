from core.ast2ir import *
import run
import helpers
import codegen
import torch




def f18():
    A = Tensor('A', (100, 20))
    B = Tensor('B', (100, 20))
    b1 = Var('b1')
    b2 = Var('b2')

    return A[b1:b2][2:4] + B[b1:b2][2:4]

def test19():
    A = Tensor('A', (100, 20))
    B = Tensor('B', (100, 20))
    b1 = Var('b1')
    b2 = Var('b2')

    ast = A[:][b1:b2] + B[:][b1:b2]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    A = torch.rand(100, 20)
    B = torch.rand(100, 20)
    b1 = 4
    b2 = 5
    d = run.cpu.compile_and_run(code, b2, b1, A, B)

    print(A[:, b1:b2] + B[:, b1:b2])
    print(torch.equal(A[:, b1:b2] + B[:, b1:b2], d))



def test20():
    d = Var('d')
    A = Tensor('A', (100, 20, d))
    B = Tensor('B', (100, 20, d))
    b1 = Var('b1')
    b2 = Var('b2')
    idx = Tensor('idx', (5, ), dtype='int')

    ast = A[:][idx][b1:b2] + B[:][idx][b1:b2]
    print(helpers.get_input_nodes(ast))
    ir = gen_ir(ast)

    code = codegen.cpu.print_cpp(ir)
    d = 7
    A = torch.rand(100, 20, d)
    B = torch.rand(100, 20, d)
    b1 = 4
    b2 = 5
    idx = torch.IntTensor(5)

    d = run.cpu.compile_and_run(code, b2, b1, d, A, idx, B)

    print(A[:, idx][:, :, b1:b2] + B[:, idx][:, :, b1:b2])
    print(torch.equal(A[:, idx][:, :, b1:b2] + B[:, idx][:, :, b1:b2], d))



def f21():
    d = Var('d')
    A = Tensor('A', (100, 20, d))
    B = Tensor('B', (50, 100, d+d))
    b1 = Var('b1')
    b2 = Var('b2')
    idx = Tensor('idx', (5, ), dtype='int')

    return A[3:0:-1][idx][b1:b2] + B[3:0:-1][idx][b1:b2]



def f30():
    d1 = Var('d1')
    d2 = Var('d2')
    T = Tensor('T', size=(10, d1, d2))
    A = Set(T)
    B = Set([1,2,3,4])

    return A.num_elem() + B.num_elem()


def f31():
    d1 = Var('d1')
    d2 = Var('d2')
    T = Tensor('T', size=(10, d1, d2))
    A = Set(T)

    return A


def f32():
    T = Tensor('T', val=[1,2,3,4])
    A = Set(T)

    return A


def f33():
    x = Var('x')
    A = Set(x)

    return A.num_elem() + Set([1,2,3,4]).num_elem()

def f34():
    x = Var('x')
    A = Set(x)

    return A

def f35():
    T = Tensor('T', size=(100, ))
    A = Set(T[20:40])

    return A

def f36():
    T = Tensor('T', size=(100, ))
    A = Set(T[20:40])
    d = Var('d')

    return A.num_elem() + Set(T[40:]).num_elem() + Set(T[1:d]).num_elem()





def conv1d_v1():
    A = Tensor('a', (100, ))
    ast = A[0:97] + A[1:98] + A[2:99]
    ir = gen_ir(ast)
    print(helpers.get_input_nodes(ir))
    code = codegen.cpu.print_cpp(ir)
    print(code)

def conv1d_v2(width):
    A = Tensor('a', (100, ))
    res = Zeros(A[width:]._size())
    for i in range(width):
        res = res + A[i:i+97]
    ir = gen_ir(res)
    print(helpers.get_input_nodes(ir))
    code = codegen.cpu.print_cpp(ir)
    print(code)



