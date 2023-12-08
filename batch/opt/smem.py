from batch.ast2ir import *
from batch.opt.ir import *

def find_reuse(ir, smem_list):
    if isinstance(ir, Indexing):
        # print(find_reuse(ir.dobject, smem_list), find_reuse(ir.idx, smem_list))
        temp = ir.dobject
        shape_size = 0
        idx = [ir.idx]
        while isinstance(temp, Indexing):
            shape_size += 1
            
            if isinstance(temp.idx, Indexing):
                if temp.idx.dobject.name() == 'r':
                    data = get_ori_var(ir)
                    if data in smem_list.keys():
                        smem_list[data].append([ir, shape_size, idx])
                    else:
                        smem_list[data] = [[ir, shape_size, idx]]
            idx.append(temp.idx)
            temp = temp.dobject
        
    elif isinstance(ir, Assignment):
        find_reuse(ir.lhs, smem_list)
        find_reuse(ir.rhs, smem_list)
    elif isinstance(ir, Expr):
        find_reuse(ir.left, smem_list)
        find_reuse(ir.right, smem_list)
    elif isinstance(ir, Loop):
        for i in ir.body:
            find_reuse(i, smem_list)

def get_ori_var(ir):
    if isinstance(ir, Indexing):
        return get_ori_var(ir.dobject)
    elif isinstance(ir, Ndarray):
        return ir

def change_expridx(ir, var):
    # if is expr, change right of expr as loop.iterate
    if isinstance(ir, Indexing):
        temp = ir
        temp.idx = change_expridx(temp.idx, var)
        temp = temp.dobject
    elif isinstance(ir, Expr):
        if ir.right != var and ir.op == '+':
            ir.right = var
    return ir

def if_exist(ir, arr):
    if isinstance(ir, Indexing):
        if ir == arr:
            return True
        else:
            return False
    elif isinstance(ir, Assignment):
        return if_exist(ir.lhs, arr) or if_exist(ir.rhs, arr)
    elif isinstance(ir, Expr):
        return if_exist(ir.left, arr) or if_exist(ir.right, arr)
    elif isinstance(ir, Loop):
        for i in ir.body:
            t = if_exist(i, arr)
            if t:
                return t
    return False

def change_access(ir, ori, cur):
    if isinstance(ir, Indexing):
        if ir == ori:
            return cur
    elif isinstance(ir, Assignment):
        ir.lhs = change_access(ir.lhs, ori, cur)
        ir.rhs = change_access(ir.rhs, ori, cur)
    elif isinstance(ir, Expr):
        ir.left = change_access(ir.left, ori, cur)
        ir.right = change_access(ir.right, ori, cur)
    elif isinstance(ir, Loop):
        for i in ir.body:
            i = change_access(i, ori, cur)
    return ir

def data_loading(compute_ir, smem_arr, ori_arr):
    decl = []
    if ori_arr.keys():
        for i in ori_arr[list(ori_arr.keys())[0]]:
            indirect_arr = get_ori_var(i[2][-1])
            decl.append(Decl(Buffer(indirect_arr)))
            decl.append(Decl(Uniq(indirect_arr)))
            break
        t = Scalar('int')
        decl.append(Decl(t))
        bufidx = Indexing(Ndarray('int', [Scalar('int', 'batch_size/16'), Scalar('int', 'BlockDim.y')]), Literal(-1, 'int'))
        bufidx.dobject = Buffer(indirect_arr)
        bufidx.idx = ThreadIdy()
        smtm1 = Assignment(t, bufidx)

        main_inequa = Expr(t, Scalar('int', 'C'), '<')
        idx_var = Scalar('int')
        decl.append(Decl(idx_var))
        idx_false = Indexing(indirect_arr, Literal(-1, 'int'))
        idx_false.idx = Expr(t, Scalar('int', 'C'), '-')
        idx = IF(idx_var, main_inequa, t, idx_false)
        ptr_access = Scalar('int')
        decl.append(Decl(ptr_access))
        set_ptr = IF(ptr_access, main_inequa, Scalar('int', 'D'), Scalar('int', 'dim'))

        compute_ir.insert(0, set_ptr)
        compute_ir.insert(0, idx)
        compute_ir.insert(0, smtm1)
        for stmt in compute_ir:
            if isinstance(stmt, Loop):
                # data loading here
                
                for i in range(len(smem_arr)):
                    cur_arr = ori_arr[list(ori_arr.keys())[i]]
                    if len(smem_arr[i].size) == 2:
                        temp_stmt = stmt

                        for iloop in stmt.body:
                            if isinstance(iloop, Loop) and isinstance(iloop.body[0], Loop):
                                if iloop.start == 0 and iloop.end.name() == 'dim' and isinstance(iloop.body[0].start, ThreadIdx) and iloop.body[0].end.name() == 'D':
                                    temp_stmt = iloop
                                    break
                        for shared_item in cur_arr:
                            flag = if_exist(temp_stmt, shared_item[0])
                            if flag:
                                store_loop = Loop(Expr(Expr(ThreadIdy(), BlockDimx(), '*'), ThreadIdx(), '+'), Expr(Scalar('float', 'D'), Uniq(shared_item[2][-1].dobject), '*'), Expr(BlockDimx(), BlockDimy(), '*'), [])
                                
                                left = smem_arr[i]
                                left = Indexing(left, Literal(-1, 'int'))
                                left.idx = Expr(store_loop.iterate, Scalar('int', 'D'), '/')
                                left = Indexing(left, Literal(-1, 'int'))
                                left.idx = Expr(store_loop.iterate, Scalar('int', 'D'), '%')
                                
                                main_arr = get_ori_var(shared_item[0])
                                indirect_arr = get_ori_var(shared_item[2][-1])
                                uniq = Ndarray('int', [Scalar('int', 'batch_size/16'), Scalar('int', 'dim')], f'{indirect_arr.__name__}_Uniq')
                                uniq = Indexing(uniq, Literal(-1, 'int'))
                                uniq.idx = BlockIdx()
                                uniq = Indexing(uniq, Literal(-1, 'int'))
                                uniq.idx = Expr(store_loop.iterate, Scalar('int', 'D'), '/')
                                main_arr = Indexing(main_arr, Literal(-1, 'int'))
                                main_arr.idx = uniq
                                main_arr = Indexing(main_arr, Literal(-1, 'int'))
                                # main_arr.idx = shared_item[2][0]
                                main_arr.idx = Expr(temp_stmt.iterate, Expr(store_loop.iterate, Scalar('float', 'D'), '%'), '+')
                                assign = Assignment(left, main_arr, '')
                                store_loop.body.append(assign)

                                
                                ptr = Pointer('float *', [Scalar('int', 'C'), ptr_access])
                                pmat = IF(ptr, main_inequa, smem_arr[i], get_ori_var(shared_item[0]))
                                decl.append(Decl(ptr))
                                
                                offset_1 = Scalar('int')
                                decl.append(Decl(offset_1))
                                ofs1 = IF(offset_1, main_inequa, 0, stmt.iterate)

                                access_var = Access_ptr(ptr, [])
                                access_var.idx.append(idx_var)
                                for kk in range(len(shared_item[2]) - 1):
                                    shared_item[2][kk].left = offset_1
                                    access_var.idx.append(shared_item[2][kk]) 

                                stmt = change_access(stmt, shared_item[0], access_var)
                                
                                temp_stmt.body.insert(0, ofs1)
                                temp_stmt.body.insert(0, pmat)

                                stmt.body.insert(0, SyncThreads())
                                stmt.body.insert(0, store_loop)
                    elif len(smem_arr[i].size) == 3:
                        temp_stmt = stmt
                        
                        for iloop in stmt.body:
                            if isinstance(iloop, Loop) and isinstance(iloop.body[0], Loop):
                                if iloop.start == 0 and iloop.end.name() == 'dim' and isinstance(iloop.body[0].start, ThreadIdx) and iloop.body[0].end.name() == 'D':
                                    temp_stmt = iloop
                                    break
                        for shared_item in cur_arr:
                            flag = if_exist(temp_stmt, shared_item[0])
                            # print(codegen.gpu.to_string(temp_stmt), codegen.gpu.to_string(shared_item[0]))
                            if flag:
                                oloop = Loop(0, 2, 1, [])
                                store_loop1 = Loop(ThreadIdy(), stmt.step, BlockDimy(), [])
                                store_loop2 = Loop(ThreadIdx(), stmt.step, BlockDimx(), [])
                                oloop.body.append(store_loop1)
                                store_loop1.body.append(store_loop2)

                                left = smem_arr[i]
                                left = Indexing(left, oloop.iterate)
                                # left.idx = ThreadIdy()
                                left = Indexing(left, store_loop1.iterate)
                                left = Indexing(left, store_loop2.iterate)

                                main_arr = get_ori_var(shared_item[0])
                                indirect_arr = get_ori_var(shared_item[2][-1])
                                uniq = Ndarray('int', [Scalar('int', 'batch_size/16'), Scalar('int', 'dim')], f'{indirect_arr.__name__}_Uniq')
                                uniq = Indexing(uniq, Literal(-1, 'int'))
                                uniq.idx = BlockIdx()
                                uniq = Indexing(uniq, oloop.iterate)
                                # uniq.idx = ThreadIdy()
                                main_arr = Indexing(main_arr, Literal(-1, 'int'))
                                main_arr.idx = uniq
                                main_arr = Indexing(main_arr, Literal(-1, 'int'))
                                main_arr.idx = Expr(shared_item[2][1].left, store_loop1.iterate, '+')
                                main_arr = Indexing(main_arr, Literal(-1, 'int'))
                                main_arr.idx = Expr(shared_item[2][0].left, store_loop2.iterate, '+')
                                assign = Assignment(left, main_arr, '')
                                store_loop2.body.append(assign)
                                
                                
                                ptr = Pointer('float *', [Scalar('int', 'C'), ptr_access, ptr_access])
                                decl.append(Decl(ptr))
                                pmat = IF(ptr, main_inequa, smem_arr[i], get_ori_var(shared_item[0]))
                                
                                offset_1 = Scalar('int')
                                decl.append(Decl(offset_1))
                                ofs1 = IF(offset_1, main_inequa, 0, stmt.iterate)
                                offset_2 = Scalar('int')
                                decl.append(Decl(offset_2))
                                ofs2 = IF(offset_2, main_inequa, 0, temp_stmt.iterate)

                                access_var = Access_ptr(ptr, [])
                                access_var.idx.append(idx_var)
                                shared_item[2][1].left = offset_2
                                access_var.idx.append(shared_item[2][1])
                                shared_item[2][0].left = offset_1
                                access_var.idx.append(shared_item[2][0])
                                
                                stmt = change_access(stmt, shared_item[0],access_var)

                                temp_stmt.body.insert(0, ofs2)
                                temp_stmt.body.insert(0, ofs1)
                                temp_stmt.body.insert(0, pmat)

                                temp_stmt.body.insert(0, SyncThreads())
                                temp_stmt.body.insert(0, oloop)
    return decl

def swap_arr_to_reg(ir, pre, cur):
    if isinstance(ir, Indexing):
        # todo: add index here
        if ir == pre[0]:
            temp = Indexing(cur, Literal(-1, 'int'))
            # temp.idx = Scalar('int', 'idx')
            for i in range(pre[1]):
                temp = Indexing(temp, Literal(-1, 'int'))
                if isinstance(pre[2][-2-i], Expr):
                    # pre[2][-2-i].left = f'ofs_{i+1}'
                    temp.idx = pre[2][-2-i].right
                else:
                    temp.idx = pre[2][-2-i]
            return temp
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

def if_in_ir(ir, arr, smem_dict):
    if isinstance(ir, Indexing):
        temp = ir
        while isinstance(temp, Indexing):
            if temp.dobject == arr:
                if temp.dobject in smem_dict.keys():
                    smem_dict[temp.dobject].append(ir)
                else:
                    smem_dict[temp.dobject] = [ir]
                return True
            temp = temp.dobject
    elif isinstance(ir, Assignment):
        return if_in_ir(ir.lhs, arr, smem_dict) or if_in_ir(ir.rhs, arr, smem_dict)
    elif isinstance(ir, Expr):
        return if_in_ir(ir.left, arr, smem_dict) or if_in_ir(ir.right, arr, smem_dict)
    elif isinstance(ir, Loop):
        bool_list = []
        for i in ir.body:
            t = if_in_ir(i, arr, smem_dict)
            bool_list.append(t)
        for i in bool_list:
            if i:
                return True
    return False



def add_smem(ast):

    if type(ast) == BatchOp:
        if type(ast.operators[1]) == BatchOp:
            add_smem(ast.operators[1])
        if type(ast.operators[0]) == BatchOp:
            add_smem(ast.operators[0])
    
    if type(ast) == BatchOp and ast.valid:
        
        var_list = {}
        smem_list = []
        for i in ast.compute:
            find_reuse(i, var_list)
        
        # for i in var_list.keys():
        #     t = get_ori_var(var_list[i][0][0])
        #     print('var_list::', codegen.gpu.to_string(var_list[i][0][0]), var_list[i][0][1], codegen.gpu.to_string(t), t, i)
        #     for j in var_list[i][0][2]:
        #         print(codegen.gpu.to_string(j))
        for i in var_list:
            size = [Scalar('int', 'C')]
            for s in range(var_list[i][0][1]):
                size.append(Scalar('int', 'D'))
            smema = Ndarray('float', size, f'smem_{i.dobject_id}')
            smem_list.append(smema)
            ast.decl.append(Decl(Shared(smema)))
        
        decl = data_loading(ast.compute, smem_list, var_list)
        ast.decl.extend(decl)

    if type(ast) == BatchOp and not ast.valid and isinstance(ast.eval, Ndarray):
        node = ast.compute[0].astnode
        ir_dict = {}
        smem_list = []
        for i in node.compute:
            if if_in_ir(i, ast.eval, ir_dict):
                smema = Ndarray('float', [Scalar('int', 'C'), Scalar('int', 'D')], f'smem_{ast.eval.dobject_id}')
                smem_list.append(smema)
                node.decl.append(Decl(Shared(smema)))
        for i in node.decl:
            if i.dobject == ast.eval:
                node.decl.remove(i)
                
        for i in range(len(list(ir_dict.keys()))):
            key = list(ir_dict.keys())[i]
            smem_idx_list = []
            # print(key, codegen.gpu.to_string(key), smem_list[i], codegen.gpu.to_string(smem_list[i]))
            for j in ir_dict[key]:
                # print('smemlist:::', codegen.gpu.to_string(j))
                idx_list = []
                temp = j
                while isinstance(temp, Indexing):
                    idx_list.append(temp.idx)
                    temp = temp.dobject
                idx_list = idx_list[::-1]
                newsmem = smem_list[i]
                for kk in range(len(idx_list)):
                    newsmem = Indexing(newsmem, Literal(-1, 'int'))
                    if isinstance(idx_list[kk], Scalar):
                        newsmem.idx = ThreadIdy()
                    elif isinstance(idx_list[kk], Expr):
                        newsmem.idx = idx_list[kk].right
                smem_idx_list.append(newsmem)
            for j in node.compute:
                for idx in range(len(smem_idx_list)):
                    j = change_access(j, ir_dict[key][idx], smem_idx_list[idx])
                