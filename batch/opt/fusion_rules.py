from core.ir import *
from batch.ast import *
from batch.ast2ir import *
import codegen


def get_same_loop(outloop, fusedloop):
    pre_o = outloop
    pre_i = fusedloop
    oloop = outloop.body[0]
    iloop = fusedloop.body[0]
    
    pre_iter_i = []
    pre_iter_o = []

    pre_iter_i.append(pre_i.iterate)
    pre_iter_o.append(pre_o.iterate)
    while (isinstance(oloop, Loop) and isinstance(iloop, Loop)) and oloop.start == iloop.start and oloop.end.__name__ == iloop.end.__name__ and oloop.step == iloop.step:
        if len(pre_i.body)>1:
            break
        pre_o = oloop
        pre_i = iloop
        pre_iter_i.append(pre_i.iterate)
        pre_iter_o.append(pre_o.iterate)

        oloop = oloop.body[-1]
        iloop = iloop.body[0]
        
    return pre_o, pre_i, pre_iter_o, pre_iter_i

def loop_merge(o_loop, i_loop):
    # print(codegen.cpu.to_string(o_loop), codegen.cpu.to_string(i_loop))
    if (isinstance(o_loop, Loop) and isinstance(i_loop, Loop)) and o_loop.start == i_loop.start and o_loop.end.__name__ == i_loop.end.__name__ and o_loop.step == i_loop.step:

        for ii in range(len(i_loop.body)-1):
            
            change_index(i_loop.body[ii], [o_loop.iterate], [i_loop.iterate])
            o_loop.body.insert(ii, i_loop.body[ii])
        # print('last loop i', count, codegen.cpu.to_string(i_loop.body[-1]))
        change_index(i_loop.body[-1], [o_loop.iterate], [i_loop.iterate])
        # print("outer::",codegen.cpu.to_string(o_loop))
        # print("inner::",codegen.cpu.to_string(i_loop))
        if not isinstance(i_loop.body[-1], Loop):
            o_loop.body.insert(-1, i_loop.body[-1])
        elif not isinstance(o_loop.body[-1], Loop):
            o_loop.body.insert(0, i_loop.body[-1])
        else:
            loop_merge(o_loop.body[-1], i_loop.body[-1])
        
        # o_loop.body.insert(-1, i_loop.body[-1])
    # else:
    #     if isinstance(i_loop, Loop):
    #         for ii in range(len(i_loop.body)):
    #             o_loop.body.insert(ii, i_loop.body[ii])
        # else:
        #     o_loop.body.insert(0, i_loop)
        

def change_index(iassign, iter_o, iter_i):
    if isinstance(iassign, Index):
        for idx, item in enumerate(iter_i):
            if iassign.index == item:
                iassign.index = iter_o[idx]
        if isinstance(iassign.dobject, Index):
            change_index(iassign.dobject, iter_o, iter_i)
        
    elif isinstance(iassign, Expr):
        # both item.left and item.right
        change_index(iassign.left, iter_o, iter_i)
        change_index(iassign.right, iter_o, iter_i)
    elif isinstance(iassign, Assignment):
        # both item.lhs and item.rhs
        change_index(iassign.lhs, iter_o, iter_i)
        change_index(iassign.rhs, iter_o, iter_i)
    elif isinstance(iassign, Loop):
        for i in iassign.body:
            change_index(i, iter_o, iter_i)
    
def swap_arr_to_reg(ir, pre, cur):
    # print(ir, codegen.cpu.to_string(ir), codegen.cpu.to_string(pre), codegen.cpu.to_string(cur))
    if isinstance(ir, Index):
        # print(ir.dobject, pre)
        temp = ir
        while isinstance(temp, Index):
            # print(temp, codegen.cpu.to_string(temp))
            temp = temp.dobject
        # print('*************', temp, codegen.cpu.to_string(temp), pre)
        if temp == pre:
            return cur
        else:
            return ir
        
    elif isinstance(ir, Expr):
        ir.left = swap_arr_to_reg(ir.left, pre, cur)
        ir.right = swap_arr_to_reg(ir.right, pre, cur)
    elif isinstance(ir, Assignment):
        ir.lhs = swap_arr_to_reg(ir.lhs, pre, cur)
        ir.rhs = swap_arr_to_reg(ir.rhs, pre, cur)
    elif isinstance(ir, Loop):
        for i in range(len(ir.body)):
            ir.body[i] = swap_arr_to_reg(ir.body[i], pre, cur)
    return ir

def fuse_elementwise(ast):

    if type(ast) == BatchOp:
        if type(ast.operators[0]) == BatchOp:
            fuse_elementwise(ast.operators[0])
        if type(ast.operators[1]) == BatchOp:
            fuse_elementwise(ast.operators[1])
    else:
        return
    
    if type(ast.operators[1]) == BatchOp and ast.op_type in core.ast.op_mapping.keys():
        # fuse operators1 into elementwise
        if ast.item_type == 'vec' and ast.operators[1].item_type == ast.item_type:
            # check if type of operator1 is vector
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]

                oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                iloop.body.clear()
                ast.operators[1].compute.clear()
        elif ast.item_type == 'scal' and ast.operators[1].item_type == ast.item_type:
            # check if type of operator1 is scalar
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]

                oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                iloop.body.clear()
                ast.operators[1].compute.clear()
        elif ast.item_type not in ['vec', 'scal']:
            raise ValueError(f"Tensor type is wrong. Expect ast as 'vec' or 'scal' but found '{ast.item_type}'.")
        elif ast.operators[1].item_type not in ['vec', 'scal']:
            raise ValueError(f"Tensor type is wrong. Expect operators[1] as 'vec' or 'scal' but found '{ast.operators[1].item_type}'.")
        elif ast.operators[1].item_type != ast.item_type:
            raise ValueError(f"Tensor shape are not the same. Expect 'vec' or 'scal' but found ast: '{ast.item_type}' and operators[1]: '{ast.operators[1].item_type}'.")

    
    if type(ast.operators[0]) == BatchOp and ast.op_type in core.ast.op_mapping.keys():
        # fuse operators0 into elementwise
        if ast.item_type == 'vec' and ast.operators[0].item_type == ast.item_type:
            # check if type of operator1 is vector
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]
                
                oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                
                iloop.body.clear()
                ast.operators[0].compute.clear()
        if ast.item_type == 'scal' and ast.operators[0].item_type == ast.item_type:
            # check if type of operator1 is scalar
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]
                
                oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                
                iloop.body.clear()
                ast.operators[0].compute.clear()
        elif ast.item_type not in ['vec','scal']:
            raise ValueError(f"Tensor shape are not the same. Expect ast as 'vec' or 'scal' but found '{ast.item_type}'.")
        elif ast.operators[0].item_type not in ['vec','scal']:
            raise ValueError(f"Tensor shape are not the same. Expect operators[0] as 'vec' or 'scal' but found '{ast.operators[0].item_type}'.")
        elif ast.operators[0].item_type != ast.item_type:
            raise ValueError(f"Tensor shape are not the same. Expect 'vec' or 'scal' but found ast: '{ast.item_type}' and operators[0]: '{ast.operators[0].item_type}'.")

def fuse_bvv(ast):
    if type(ast) == BatchOp:
        if type(ast.operators[1]) == BatchOp:
            fuse_bvv(ast.operators[1])
        if type(ast.operators[0]) == BatchOp:
            fuse_bvv(ast.operators[0])
    else:
        return
    
    if type(ast.operators[1]) == BatchOp and ast.op_type == 'vec_mul_vec':
        # fuse operators1 into bvv
        if ast.item_type == 'scal' and ast.operators[1].item_type == 'vec':
            # check if type of operator1 is vector
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]

                oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                iloop.body.clear()
                ast.operators[1].compute.clear()
        elif ast.operators[1].item_type != 'vec':
            raise ValueError(f"Tensor type is wrong. Expect operators[1] as 'vec' but found '{ast.operators[1].item_type}'.")    
    
    if type(ast.operators[0]) == BatchOp and ast.op_type == 'vec_mul_vec':
        # fuse operators1 into bvv
        if ast.item_type == 'scal' and ast.operators[0].item_type == 'vec':
            # check if type of operator0 is vector
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]

                oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                iloop.body.clear()
                ast.operators[0].compute.clear()
        elif ast.operators[0].item_type != 'vec':
            raise ValueError(f"Tensor type is wrong. Expect operators[0] as 'vec' but found '{ast.operators[0].item_type}'.")    
    
def fuse_bsv(ast):
    if type(ast) == BatchOp:
        if type(ast.operators[1]) == BatchOp:
            fuse_bsv(ast.operators[1])
        if type(ast.operators[0]) == BatchOp:
            fuse_bsv(ast.operators[0])
    else:
        return
    
    if type(ast) == BatchOp and ast.op_type == 'scal_mul_vec':
        # fuse operators into bsv
        if ast.item_type == 'vec' and ast.operators[1].item_type == 'vec' and ast.operators[0].item_type == 'scal':
            # check if type of operators are scal and vector
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]

                oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                iloop.body.clear()
                ast.operators[1].compute.clear()
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]

                oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                iloop.body.clear()
                ast.operators[0].compute.clear()
        elif ast.operators[0].item_type != 'scal':
            raise ValueError(f"Tensor type is wrong. Expect operators[0] as 'scal' but found '{ast.operators[0].item_type}'.")   
        elif ast.operators[1].item_type != 'vec':
            raise ValueError(f"Tensor type is wrong. Expect operators[1] as 'vec' but found '{ast.operators[0].item_type}'.")
        elif ast.item_type != 'vec':
            raise ValueError(f"Tensor type is wrong. Expect ast node as 'vec' but found '{ast.item_type}'.")   

def fuse_bvm(ast):
    if type(ast) == BatchOp:
        if type(ast.operators[1]) == BatchOp:
            fuse_bvm(ast.operators[1])
        if type(ast.operators[0]) == BatchOp:
            fuse_bvm(ast.operators[0])
    else:
        return
    
    if type(ast) == BatchOp and ast.op_type == 'vec_mul_mat':
    # fuse operators1 into bov
        if ast.item_type == 'vec' and ast.operators[0].item_type == 'vec' and ast.operators[1].item_type == 'mat':
            # check if type of operator1 is vector
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]

                oloop = outer_loop
                iloop = loop
                iter_o = []
                iter_i = []
                if (isinstance(oloop, Loop) and isinstance(iloop, Loop)) and oloop.start == iloop.start and oloop.end.__name__ == iloop.end.__name__ and oloop.step == iloop.step:
                    iter_i.append(iloop.iterate)
                    iter_o.append(oloop.iterate)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                iloop.body.clear()
                ast.operators[1].compute.clear()
                
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]

                oloop = outer_loop
                iloop = loop
                iter_o = []
                iter_i = []
                if (isinstance(oloop, Loop) and isinstance(iloop, Loop)) and oloop.start == iloop.start and oloop.end.__name__ == iloop.end.__name__ and oloop.step == iloop.step:
                    iter_i.append(iloop.iterate)
                    iter_o.append(oloop.iterate)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                iloop.body.clear()
                ast.operators[0].compute.clear()
        elif ast.item_type != 'vec':
            raise ValueError(f"Tensor shape are not the same. Expect ast node type 'mat' but found '{ast.item_type}'.")
        elif ast.operators[1].item_type != 'mat':
            raise ValueError(f"Tensor shape are not the same. Expect ast.operators[1] node type 'mat' but found '{ast.operators[1].item_type}'.")
        elif ast.operators[0].item_type != 'vec':
            raise ValueError(f"Tensor shape are not the same. Expect ast.operators[0] node type 'vec' but found '{ast.operators[1].item_type}'.")

def fuse_bov(ast):
    if type(ast) == BatchOp:
        if type(ast.operators[1]) == BatchOp:
            fuse_bov(ast.operators[1])
        if type(ast.operators[0]) == BatchOp:
            fuse_bov(ast.operators[0])
    else:
        return
    
    if type(ast.operators[1]) == BatchOp and ast.op_type == 'vec_outer_vec':
        # fuse operators1 into bov
        if ast.item_type == 'mat' and ast.operators[1].item_type == 'vec':
            # check if type of operator1 is vector
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]

                oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                iloop.body.clear()
                ast.operators[1].compute.clear()
        elif ast.item_type != 'mat':
            raise ValueError(f"Tensor shape are not the same. Expect ast node type 'mat' but found '{ast.item_type}'.")
        elif ast.operators[0].item_type != 'vec':
            raise ValueError(f"Tensor shape are not the same. Expect ast.operators[1] node type 'vec' but found '{ast.operators[1].item_type}'.")
            
    
    if type(ast.operators[0]) == BatchOp and ast.op_type == 'vec_outer_vec':
        # fuse operators0 into bov
        if ast.item_type == 'mat' and ast.operators[0].item_type == 'vec':
            # check if type of operator0 is vector
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]

                oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                iloop.body.clear()
                ast.operators[0].compute.clear()
        elif ast.item_type != 'mat':
            raise ValueError(f"Tensor shape are not the same. Expect ast node type 'mat' but found '{ast.item_type}'.")
        elif ast.operators[0].item_type != 'vec':
            raise ValueError(f"Tensor shape are not the same. Expect ast.operators[0] node type 'vec' but found '{ast.operators[0].item_type}'.")
        


def fuse_operators(ast):

    if type(ast) == BatchOp:
        if type(ast.operators[1]) == BatchOp:
            # print('op1:', ast.operators[1].op_type)
            fuse_operators(ast.operators[1])
        if type(ast.operators[0]) == BatchOp:
            # print('op0:', ast.operators[0].op_type)
            fuse_operators(ast.operators[0])
    else:
        return
    
    if type(ast.operators[1]) == BatchOp and ast.op_type in core.ast.op_mapping.keys():
        # fuse operators1 into elementwise
        if ast.item_type == 'vec' and ast.operators[1].item_type == ast.item_type:
            # check if type of operator1 is vector
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]

                # oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                # for i in iloop.body:
                #     change_index(i, iter_o, iter_i)
                # for i in range(len(iloop.body)):
                #     oloop.body.insert(i, iloop.body[i])
                
                loop_merge(outer_loop, loop)
                if ast.operators[1].op_type in core.ast.op_mapping.keys() or ast.operators[1].op_type == "scal_mul_vec":
                    a = Scalar(ast.operators[1].eval.dtype)
                    pre_arr = ast.operators[1].eval
                    ast.operators[1].decl = [Decl(a)]
                    ast.operators[1].eval = a
                    outer_loop = swap_arr_to_reg(outer_loop, pre_arr, a)
                ast.operators[1].compute.clear()
        elif ast.item_type == 'scal' and ast.operators[1].item_type == ast.item_type:
            # check if type of operator1 is scalar
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]
                
                loop_merge(outer_loop, loop)
                if ast.operators[1].op_type in core.ast.op_mapping.keys() or ast.operators[1].op_type == "scal_mul_vec":
                    a = Scalar(ast.operators[1].eval.dtype)
                    pre_arr = ast.operators[1].eval
                    ast.operators[1].decl = [Decl(a)]
                    ast.operators[1].eval = a
                    outer_loop = swap_arr_to_reg(outer_loop, pre_arr, a)
                ast.operators[1].compute.clear()
        elif ast.item_type not in ['vec', 'scal']:
            raise ValueError(f"Tensor type is wrong. Expect ast as 'vec' or 'scal' but found '{ast.item_type}'.")
        elif ast.operators[1].item_type not in ['vec', 'scal']:
            raise ValueError(f"Tensor type is wrong. Expect operators[1] as 'vec' or 'scal' but found '{ast.operators[1].item_type}'.")
        elif ast.operators[1].item_type != ast.item_type:
            raise ValueError(f"Tensor shape are not the same. Expect 'vec' or 'scal' but found ast: '{ast.item_type}' and operators[1]: '{ast.operators[1].item_type}'.")

    
    if type(ast.operators[0]) == BatchOp and ast.op_type in core.ast.op_mapping.keys():
        # fuse operators0 into elementwise
        if ast.item_type == 'vec' and ast.operators[0].item_type == ast.item_type:
            # check if type of operator0 is vector
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]
                loop_merge(outer_loop, loop)
                if ast.operators[0].op_type in core.ast.op_mapping.keys() or ast.operators[0].op_type == "scal_mul_vec":
                    a = Scalar(ast.operators[0].eval.dtype)
                    pre_arr = ast.operators[0].eval
                    ast.operators[0].decl = [Decl(a)]
                    ast.operators[0].eval = a
                    outer_loop = swap_arr_to_reg(outer_loop, pre_arr, a)
                ast.operators[0].compute.clear()
        if ast.item_type == 'scal' and ast.operators[0].item_type == ast.item_type:
            # check if type of operator1 is scalar
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]
                
                loop_merge(outer_loop, loop)
                if ast.operators[0].op_type in core.ast.op_mapping.keys() or ast.operators[0].op_type == "scal_mul_vec":
                    a = Scalar(ast.operators[0].eval.dtype)
                    pre_arr = ast.operators[0].eval
                    ast.operators[0].decl = [Decl(a)]
                    ast.operators[0].eval = a
                    outer_loop = swap_arr_to_reg(outer_loop, pre_arr, a)
                ast.operators[0].compute.clear()
        elif ast.item_type not in ['vec','scal']:
            raise ValueError(f"Tensor shape are not the same. Expect ast as 'vec' or 'scal' but found '{ast.item_type}'.")
        elif ast.operators[0].item_type not in ['vec','scal']:
            raise ValueError(f"Tensor shape are not the same. Expect operators[0] as 'vec' or 'scal' but found '{ast.operators[0].item_type}'.")
        elif ast.operators[0].item_type != ast.item_type:
            raise ValueError(f"Tensor shape are not the same. Expect 'vec' or 'scal' but found ast: '{ast.item_type}' and operators[0]: '{ast.operators[0].item_type}'.")
    
    if type(ast.operators[1]) == BatchOp and ast.op_type == 'vec_mul_vec':
        # fuse operators1 into bvv
        if ast.item_type == 'scal' and ast.operators[1].item_type == 'vec':
            # check if type of operator1 is vector
            print(ast.operators[1].compute , ast.compute)
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]

                # oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                # for i in iloop.body:
                #     change_index(i, iter_o, iter_i)
                # for i in range(len(iloop.body)):
                #     oloop.body.insert(i, iloop.body[i])
                # iloop.body.clear()
                loop_merge(outer_loop, loop)
                if ast.operators[1].op_type in core.ast.op_mapping.keys() or ast.operators[1].op_type == "scal_mul_vec":
                    a = Scalar(ast.operators[1].eval.dtype)
                    pre_arr = ast.operators[1].eval
                    ast.operators[1].decl = [Decl(a)]
                    ast.operators[1].eval = a
                    outer_loop = swap_arr_to_reg(outer_loop, pre_arr, a)
                ast.operators[1].compute.clear()
        elif ast.operators[1].item_type != 'vec':
            raise ValueError(f"Tensor type is wrong. Expect operators[1] as 'vec' but found '{ast.operators[1].item_type}'.")    
    

    if type(ast.operators[0]) == BatchOp and ast.op_type == 'vec_mul_vec':
        # fuse operators1 into bvv
        if ast.item_type == 'scal' and ast.operators[0].item_type == 'vec':
            # check if type of operator0 is vector
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]

                # oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                # for i in iloop.body:
                #     change_index(i, iter_o, iter_i)
                # for i in range(len(iloop.body)):
                #     oloop.body.insert(i, iloop.body[i])
                # iloop.body.clear()
                
                loop_merge(outer_loop, loop)
                if ast.operators[0].op_type in core.ast.op_mapping.keys() or ast.operators[0].op_type == "scal_mul_vec":
                    a = Scalar(ast.operators[0].eval.dtype)
                    pre_arr = ast.operators[0].eval
                    ast.operators[0].decl = [Decl(a)]
                    ast.operators[0].eval = a
                    outer_loop = swap_arr_to_reg(outer_loop, pre_arr, a)
                ast.operators[0].compute.clear()
        elif ast.operators[0].item_type != 'vec':
            raise ValueError(f"Tensor type is wrong. Expect operators[0] as 'vec' but found '{ast.operators[0].item_type}'.")    


    if type(ast) == BatchOp and ast.op_type == 'scal_mul_vec':
        # fuse operators into bsv
        if ast.item_type == 'vec' and ast.operators[1].item_type == 'vec' and ast.operators[0].item_type == 'scal':
            # check if type of operators are scal and vector
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]

                # oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                # for i in iloop.body:
                #     change_index(i, iter_o, iter_i)
                # for i in range(len(iloop.body)):
                #     oloop.body.insert(i, iloop.body[i])
                # iloop.body.clear()
                loop_merge(outer_loop, loop)
                if ast.operators[1].op_type in core.ast.op_mapping.keys() or ast.operators[1].op_type == "scal_mul_vec":
                    a = Scalar(ast.operators[1].eval.dtype)
                    pre_arr = ast.operators[1].eval
                    ast.operators[1].decl = [Decl(a)]
                    ast.operators[1].eval = a
                    outer_loop = swap_arr_to_reg(outer_loop, pre_arr, a)
                ast.operators[1].compute.clear()
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]

                oloop = outer_loop
                iloop = loop
                iter_o = []
                iter_i = []
                if (isinstance(oloop, Loop) and isinstance(iloop, Loop)) and oloop.start == iloop.start and oloop.end.__name__ == iloop.end.__name__ and oloop.step == iloop.step:
                    iter_i.append(iloop.iterate)
                    iter_o.append(oloop.iterate)
                if isinstance(iloop, Loop):
                    for i in iloop.body:
                        change_index(i, iter_o, iter_i)
                
                    for i in range(len(iloop.body)):
                        oloop.body.insert(i, iloop.body[i])
                    if ast.operators[0].op_type in core.ast.op_mapping.keys() or ast.operators[0].op_type == "scal_mul_vec":
                        a = Scalar(ast.operators[0].eval.dtype)
                        pre_arr = ast.operators[0].eval
                        ast.operators[0].decl = [Decl(a)]
                        ast.operators[0].eval = a
                        oloop = swap_arr_to_reg(oloop, pre_arr, a)
                    iloop.body.clear()
                    ast.operators[0].compute.clear()
        elif ast.operators[0].item_type != 'scal':
            raise ValueError(f"Tensor type is wrong. Expect operators[0] as 'scal' but found '{ast.operators[0].item_type}'.")   
        elif ast.operators[1].item_type != 'vec':
            raise ValueError(f"Tensor type is wrong. Expect operators[1] as 'vec' but found '{ast.operators[0].item_type}'.")
        elif ast.item_type != 'vec':
            raise ValueError(f"Tensor type is wrong. Expect ast node as 'vec' but found '{ast.item_type}'.")   
        

    if type(ast) == BatchOp and ast.op_type == 'vec_mul_mat':
        # fuse operators1 into bov
        if ast.item_type == 'vec' and ast.operators[0].item_type == 'vec' and ast.operators[1].item_type == 'mat':
            # check if type of operator1 is vector
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]

                # oloop = outer_loop
                # iloop = loop
                # iter_o = []
                # iter_i = []
                # if (isinstance(oloop, Loop) and isinstance(iloop, Loop)) and oloop.start == iloop.start and oloop.end.__name__ == iloop.end.__name__ and oloop.step == iloop.step:
                #     iter_i.append(iloop.iterate)
                #     iter_o.append(oloop.iterate)
                # for i in iloop.body:
                #     change_index(i, iter_o, iter_i)
                # for i in range(len(iloop.body)):
                #     oloop.body.insert(i, iloop.body[i])
                # iloop.body.clear()
                loop_merge(outer_loop, loop)
                ast.operators[1].compute.clear()
                
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]
                oloop = outer_loop
                iloop = loop
                iter_o = []
                iter_i = []

                while (isinstance(oloop, Loop) and isinstance(iloop, Loop)) and oloop.start == iloop.start and oloop.end.__name__ == iloop.end.__name__ and oloop.step == iloop.step:
                    iter_i.append(iloop.iterate)
                    iter_o.append(oloop.iterate)
                    if isinstance(oloop.body[-1], Loop) and isinstance(iloop, Loop):
                        oloop = oloop.body[-1].body[-1]
                        iloop = iloop.body[-1]
                    else:
                        break
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                ast.operators[0].compute.clear()
        elif ast.item_type != 'vec':
            raise ValueError(f"Tensor shape are not the same. Expect ast node type 'mat' but found '{ast.item_type}'.")
        elif ast.operators[1].item_type != 'mat':
            raise ValueError(f"Tensor shape are not the same. Expect ast.operators[1] node type 'mat' but found '{ast.operators[1].item_type}'.")
        elif ast.operators[0].item_type != 'vec':
            raise ValueError(f"Tensor shape are not the same. Expect ast.operators[0] node type 'vec' but found '{ast.operators[1].item_type}'.")


    if type(ast.operators[1]) == BatchOp and ast.op_type == 'vec_outer_vec':
        # fuse operators1 into bov
        if ast.item_type == 'mat' and ast.operators[1].item_type == 'vec':
            # check if type of operator1 is vector
            if ast.operators[1].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[1].compute[0]
                oloop = outer_loop
                iloop = loop
                iter_i = []
                iter_o = []
                while (isinstance(oloop, Loop) and isinstance(iloop, Loop)) and oloop.start == iloop.start and oloop.end.__name__ == iloop.end.__name__ and oloop.step == iloop.step:
                    iter_i.append(iloop.iterate)
                    iter_o.append(oloop.iterate)
                    if isinstance(oloop.body[-1], Loop) and isinstance(iloop, Loop):
                        oloop = oloop.body[-1].body[-1]
                        iloop = iloop.body[-1]
                    else:
                        break
                for i in iloop.body:
                    change_index(i, iter_o, iter_i)
                for i in range(len(iloop.body)):
                    oloop.body.insert(i, iloop.body[i])
                ast.operators[1].compute.clear()
        elif ast.item_type != 'mat':
            raise ValueError(f"Tensor shape are not the same. Expect ast node type 'mat' but found '{ast.item_type}'.")
        elif ast.operators[0].item_type != 'vec':
            raise ValueError(f"Tensor shape are not the same. Expect ast.operators[1] node type 'vec' but found '{ast.operators[1].item_type}'.")
            
    
    if type(ast.operators[0]) == BatchOp and ast.op_type == 'vec_outer_vec':
        # fuse operators0 into bov
        if ast.item_type == 'mat' and ast.operators[0].item_type == 'vec':
            # check if type of operator0 is vector
            if ast.operators[0].compute and ast.compute:
                outer_loop = ast.compute[0]
                loop = ast.operators[0].compute[0]

                # oloop, iloop, iter_o, iter_i = get_same_loop(outer_loop, loop)
                # for i in iloop.body:
                #     change_index(i, iter_o, iter_i)
                # for i in range(len(iloop.body)):
                #     oloop.body.insert(i, iloop.body[i])
                # iloop.body.clear()
                loop_merge(outer_loop, loop)
                ast.operators[0].compute.clear()
        elif ast.item_type != 'mat':
            raise ValueError(f"Tensor shape are not the same. Expect ast node type 'mat' but found '{ast.item_type}'.")
        elif ast.operators[0].item_type != 'vec':
            raise ValueError(f"Tensor shape are not the same. Expect ast.operators[0] node type 'vec' but found '{ast.operators[0].item_type}'.")