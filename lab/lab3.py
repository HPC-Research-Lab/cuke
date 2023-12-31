import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ir import *
from codegen.cpu import *

import codegen.cpu
from core.ir import *
from core.ast import *


def get_obj(ir: (Index, Scalar)):
    obj = ir
    while hasattr(obj, 'dobject'):
        obj = obj.dobject
    return obj


def replace_index_with_scalar(ir, old, new):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            replace_index_with_scalar(l, old, new)
    elif type(ir) == Loop:
        replace_index_with_scalar(ir.body, old, new)
    elif type(ir) == Expr:
        if type(ir.left) in (Index, Scalar):
            obj = get_obj(ir.left)
            if obj == old:
                ir.left = new
        else:
            replace_index_with_scalar(ir.left, old, new)
        
        if type(ir.right) in (Index, Scalar):
            obj = get_obj(ir.right)
            if obj == old:
                ir.right = new
        else:
            replace_index_with_scalar(ir.right, old, new)
    elif type(ir) == Assignment:
        if type(ir.lhs) in (Index, Scalar):
            obj = get_obj(ir.lhs)
            if obj == old:
                ir.lhs = new
        else:
            replace_index_with_scalar(ir.lhs, old, new)
        
        if type(ir.rhs) in (Index, Scalar):
            obj = get_obj(ir.rhs)
            if obj == old:
                ir.rhs = new
        else:
            replace_index_with_scalar(ir.rhs, old, new)

#Define it by yourself
def rebind_iterate(ir, old_idx, new_idx):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            rebind_iterate(l, old_idx, new_idx)
    elif type(ir) == Assignment:
        rebind_iterate(ir.left,  old_idx, new_idx)
        rebind_iterate(ir.right, old_idx, new_idx)
    elif type(ir) == Index:
        if ir.index == old_idx:
            ir.index = new_idx:
    # Some other types 

def fusable_level(node0, node1):
    def _fusable_level(loop0, loop1, level):
        if  type(loop0)!=Loop or type(loop1)!=Loop:
            return level
        if loop0.start==loop1.start and loop0.end==loop1.end and loop0.step==loop1.step:
            return _fusable_level(loop0.body[0], loop1.body[0], level+1)
        else:
            return level
    loop0 = node0.compute[0]
    loop1 = node1.compute[0]
    return _fusable_level(loop0, loop1, 0)

def move_ir(node0, node1, move_level):
    def _move_ir(loop0, loop1, cur_level):
        if cur_level==move_level-1:
            loop1.body = loop0.body + loop1.body 
        else:
            _move_ir(loop0.body[0], loop1.body[0], cur_level+1) 

    loop0 = node0.compute[0]
    loop1 = node1.compute[0]
    _move_ir(loop0, loop1, 0)

def get_rhs(node):
    def _get_rhs(ir):
        if type(ir)==Loop:
            return _get_rhs(ir.body[0])
        else:
            print(ir)
            assert type(ir)==Assignment
            return ir.rhs
    return _get_rhs(node.compute[0])

def fuse(ast_wit_ir):
	#ast_wit_irt =  #implement here
    elementwise_op = op_mapping
    node = ast_wit_ir
    def action(node, res):
        if type(node) == TensorOp and node.op_type in elementwise_op:
            if type(node.operators[0]) == TensorOp and node.operators[0].op_type in elementwise_op:      
                print("Find fusable pairs! Left")
                #Do something here
            if type(node.operators[1]) == TensorOp and node.operators[1].op_type in elementwise_op:
                print("Find fusable pairs! Right")
                #Do something here
            if type(node.operators[0]) == TensorOp and node.operators[0].op_type == 'einsum':
                #Do something here
            if type(node.operators[1]) == TensorOp and node.operators[1].op_type == 'einsum':
                #Do something here
           
    t = helpers.Traversal(action)
    t(node)
    return node


def test1():
    A = Tensor('a', (30, 30))
    B = Tensor('b', (30, 30))
    C = Tensor('c', (30, 30))
    D = Tensor('d', (30, 30))
    E = Tensor('e', (30, 30))

    res1 = A + B 
    res2 = C + D
    res = res1 + res2 + E
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print(code)
    new_res_with_ir = fuse(res_with_ir)

def test2():
    A = Tensor('a', (20, 30))
    B = Tensor('b', (30, 40))
    C = Tensor('c', (20, 40))
    D = Tensor('d', (40, 40))

    res1 = A @ B 
    res2 = C @ D
    res = res1 + res2
    res_with_ir = gen_ir(res)
    code = codegen.cpu.gen_cpp(res_with_ir)
    print(code)
    new_res_with_ir = fuse(res_with_ir)

if __name__ == "__main__":
    test1()


#     torch::Tensor add_add_a_b_c(torch::Tensor obj_a, torch::Tensor obj_b, torch::Tensor obj_c)
# {
    # auto a = obj_a.accessor<float, 2>();
    # auto b = obj_b.accessor<float, 2>();
    # torch::Tensor obj_arr2 = torch::empty({30,30}, at::kFloat);
    # auto arr2 = obj_arr2.accessor<float, 2>();
    # for (int _l0 = 0; _l0 < 30; _l0 += 1) {
    #   for (int _l1 = 0; _l1 < 30; _l1 += 1) {
    #       arr2[_l0][_l1] = a[_l0][_l1] + b[_l0][_l1];
    # } 
    # } 
    # auto c = obj_c.accessor<float, 2>();
    # torch::Tensor obj_arr6 = torch::empty({30,30}, at::kFloat);
    # auto arr6 = obj_arr6.accessor<float, 2>();
    # for (int _l2 = 0; _l2 < 30; _l2 += 1) {
    #   for (int _l3 = 0; _l3 < 30; _l3 += 1) {
    #       arr6[_l2][_l3] = arr2[_l2][_l3] + c[_l2][_l3];
    # } 
    # } 
    # return obj_arr6;

    # }

    # code = codegen.cpu.gen_cpp(new_res_with_ir)
    # print(code)



