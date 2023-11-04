from core.ir import *
from batch.ast import *
from batch.ast2ir import *
import codegen
from batch.opt.ir import *
# for better optimization on GPU

def swap_arr_to_reg(ir, pre, cur):
    if isinstance(ir, Indexing):
        temp = ir
        while isinstance(temp, Indexing):
            temp = temp.dobject
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

def find_arr_ind(ir, pre):
    if isinstance(ir, Indexing):
        temp = ir
        while isinstance(temp, Indexing):
            temp = temp.dobject
        if temp == pre:
            return ir
        else:
            return None
    elif isinstance(ir, Expr):
        return Expr(find_arr_ind(ir.left, pre), find_arr_ind(ir.right, pre), ir.op)
    elif isinstance(ir, Assignment):
        return find_arr_ind(ir.lhs, pre)
    elif isinstance(ir, Loop):
        for i in ir.body:
            t = find_arr_ind(i, pre)
            if t:
                return t

def if_contain(item, ir):
    if isinstance(item, Loop):
        flag = False
        for i in item.body:
            if isinstance(i, Loop):
                flag = if_contain(i, ir)
            elif i == ir:
                flag = True
            if flag:
                return True
            
    return False

def add_thready(ir, arr):
    if isinstance(ir, Indexing):
        temp = ir
        idx_list = []
        while isinstance(temp, Indexing):
            idx_list.append(temp.idx)
            temp = temp.dobject
        if temp == arr:
            idx_list = idx_list[::-1]
            temp = Indexing(temp, Literal(-1, 'int'))
            temp.idx = ThreadIdy()
            for i in idx_list:
                if isinstance(i, (Scalar, Literal, Indexing)):
                    temp = Indexing(temp, i)
                else:
                    temp = Indexing(temp, Literal(-1, 'int'))
                    temp.idx = i
            ir = temp
            # print('yes', codegen.gpu.to_string(temp))
    elif isinstance(ir, Assignment):
        ir.lhs = add_thready(ir.lhs, arr)
        ir.rhs = add_thready(ir.rhs, arr)
    elif isinstance(ir, Expr):
        ir.left = add_thready(ir.left, arr)
        ir.right = add_thready(ir.right, arr)
    elif isinstance(ir, Loop):
        for i in range(len(ir.body)):
            ir.body[i] = add_thready(ir.body[i], arr)
    return ir

def add_reduction(ast):
    if type(ast) == BatchOp:
        if type(ast.operators[1]) == BatchOp:
            add_reduction(ast.operators[1])
        if type(ast.operators[0]) == BatchOp:
            add_reduction(ast.operators[0])
    else:
        return
    
    # todo: add traverse action to add reduction
    if ast.op_type == 'vec_mul_vec':
        # this inner_prod node is fused with upper layer
        eval = ast.eval
        # print(codegen.cpu.to_string(eval), codegen.cpu.to_string(ast.operators[0].eval), codegen.cpu.to_string(ast.operators[1].eval))
        # print(codegen.gpu.to_string(eval), ast.eval, ast.operators[0].eval, ast.operators[1].eval)
        # iff eval is scalar, we need to add shfl_sync
        if isinstance(ast.eval, Ndarray):
            new_compute = []
            for idx, item in enumerate(ast.compute):
                new_compute.append(item)
                if isinstance(item, Loop):
                    a = Scalar(ast.eval.dtype)
                    pre_arr = ast.eval
                    ast.decl.append(Decl(a))
                    t = find_arr_ind(item, pre_arr)
                    
                    swap_arr_to_reg(item, pre_arr, a)
                    new_compute.append(ShuffleDown(a))
                    new_compute.append(SaveAtThread(a, t, 0))
            ast.compute = new_compute
        elif not ((isinstance(ast.operators[0].eval, Ndarray) or isinstance(ast.operators[1].eval, Ndarray))):
            for i in ast.compute:
                # search all compute stmts
                if isinstance(i, Loop):
                    for j in i.body:
                        # search loop body
                        if isinstance(j, Loop):
                            # find the stmt of ast node 
                            main_loop = j.astnode.compute
                            for idx, item in enumerate(main_loop):
                                
                                if isinstance(item, Loop) and item == j:
                                    # fused operators
                                    main_loop.insert(idx+1, SyncThreads())
                                    main_loop.insert(idx+1, BroadCast(eval))
                                    main_loop.insert(idx+1, ShuffleDown(eval))
                                elif isinstance(item, Loop):
                                    # before fuse operators
                                    if if_contain(item, j.body[0]):
                                        main_loop.insert(idx+1, SyncThreads())
                                        main_loop.insert(idx+1, BroadCast(eval))
                                        main_loop.insert(idx+1, ShuffleDown(eval))
    if ast.op_type == 'vec_mul_mat':
        # print(ast.eval, codegen.gpu.to_string(ast.eval), ast.eval.size)
        ast.eval.size.insert(0, Scalar('int', 'C'))
        # print(ast.compute[0].astnode.compute)
        for i in ast.compute[0].astnode.compute:
            # print(codegen.gpu.to_string(i))
            t = add_thready(i, ast.eval)
            # print(t, codegen.gpu.to_string(t))
        
        


def cuda_spec(ast):
    if ast.compute and ast.valid:
        compute_list = []
        for body in ast.compute:
            body_list = []
            if isinstance(body, Loop):
                # print(body.iterate, body.iterate.dobject)
                ast.decl.append(Decl(body.iterate))
                assign = Assignment(body.iterate, Expr(ThreadIdy(), Expr(BlockDimy(), BlockIdx(), '*'), '+'))
                body_list.append(assign)
                for item in body.body:
                    if isinstance(item, Loop) and item.step == 1:
                        item.start = ThreadIdx()
                        item.step = BlockDimx()
                    elif isinstance(item, Loop):
                        for j in item.body:
                            if isinstance(j, Loop) and j.step == 1:
                                j.start = ThreadIdx()
                                j.step = BlockDimx()
                            elif isinstance(j, Loop):
                                for k in j.body:
                                    if isinstance(k, Loop) and k.step == 1:
                                        k.start = ThreadIdx()
                                        k.step = BlockDimx()

                    body_list.append(item)
                compute_list.extend(body_list)
            ast.compute = compute_list

def add_cuda_spec(ast):
    if type(ast) == BatchOp:
        if type(ast.operators[1]) == BatchOp:
            add_cuda_spec(ast.operators[1])
        if type(ast.operators[0]) == BatchOp:
            add_cuda_spec(ast.operators[0])
    else:
        return

    cuda_spec(ast)
    

def parallel(ast):
    
    # print(ast, ast.op_type, ast.compute)
    # for i in ast.compute:
    #     print(i, i.ast_ref, i.ast_ref.compute)
    # print(ast.compute)
    # for i in ast.compute:
    #     print(codegen.gpu.to_string(i))
    
    add_cuda_spec(ast)
    add_reduction(ast)
    
        