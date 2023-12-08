from batch.ast2ir import *
from batch.opt.ir import *

def swap_arr_to_reg(ir, pre, cur):
    # print(codegen.gpu.to_string(ir), codegen.gpu.to_string(pre), codegen.gpu.to_string(cur))
    if isinstance(ir, Indexing):
        temp = ir
        while isinstance(temp, Indexing):
            # print(codegen.gpu.to_string(temp), codegen.gpu.to_string(temp.idx))
            if temp.idx == pre:
                temp.idx = Expr(pre, cur, '+')
            temp = temp.dobject
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

def tile_wD(ir):
    if isinstance(ir, Loop):
        if ir.end.name() == 'dim':
            scalar_D = Scalar('int', 'D')
            tbody = ir.body
            ir.step = scalar_D
            new_loop = Loop(0, scalar_D, 1,[])
            for i in range(len(tbody)):
                # tile_wD(tbody[i])
                tbody[i] = swap_arr_to_reg(tbody[i], ir.iterate, new_loop.iterate)
            new_loop.body.extend(tbody)
            # iloops.append(new_loop)
            ir.body = [new_loop]
    return ir

def recursive_tile(ir):
    if isinstance(ir, Loop):
        ir = tile_wD(ir)
        for i in range(len(ir.body)):
            ir.body[i] = recursive_tile(ir.body[i])
         
    return ir

def tile_loops(ir, tile_list):
    if isinstance(ir, Loop) and ir.end.name() == 'dim':
        scalar_D = Scalar('int', 'D')
        ir.step = scalar_D
        new_loop = Loop(0, scalar_D, 1,[])
        tile_list.append(ir)
        for i in range(len(ir.body)):
            ir.body[i] = tile_loops(ir.body[i], tile_list)
    return ir

def swap_loops_tile(ir):
    if isinstance(ir, Loop):
        # print('oloop and iloop:', codegen.gpu.to_string(oloop), codegen.gpu.to_string(iloop))
        temp = ir
        
        # while isinstance(temp, Loop) and temp.end.name() != 'dim':
            
        #     temp = temp.body[0]
        for i in range(len(ir.body)):
            # print(ir.body[i], codegen.gpu.to_string(ir.body[i]))
            if isinstance(ir.body[i], Loop) and ir.body[i].end.name() != 'dim':
                # print(ir.body[i], codegen.gpu.to_string(ir.body[i]))
                ir.body[i] = swap_loops_tile(ir.body[i])
            
            elif isinstance(ir.body[i], Loop) and ir.body[i].end.name() == 'dim':
                # print(ir.body[i], codegen.gpu.to_string(ir.body[i]))
                tile_list = []
                temp = ir.body[i]
                tile_loops(temp, tile_list)
                # print(temp, tile_list, codegen.gpu.to_string(temp), codegen.gpu.to_string(tile_list[-1]))
                # print(codegen.gpu.to_string(tile_list[0]))
                # print(tile_list)
                multi_lv_loop = {}
                for lidx in range(len(tile_list)):
                    t = tile_list[lidx]
                    # add tiled loop here 
                    scalar_D = Scalar('int', 'D')
                    new_loop = Loop(0, scalar_D, 1,[])
                    temp_body = []
                    for k in range(len(t.body)):
                        item = t.body[k]
                        if isinstance(item, Loop) and item.end.name() == 'dim':
                            if new_loop.body != []:
                                # add new_tiled loop and create a new one
                                temp_body.append(new_loop)
                                new_loop = Loop(0, scalar_D, 1,[])
                            
                            if item in multi_lv_loop.keys():
                                multi_lv_loop[item].append([multi_lv_loop[item][-1][0]+1, t])
                            else:
                                multi_lv_loop[item] = [[1, t]]
                            temp_body.append(item)
                        elif t in multi_lv_loop.keys():
                            oloop = Loop(0, scalar_D, 1,[])
                            for jj in range(len(t.body)):
                                t.body[jj] = swap_arr_to_reg(t.body[jj], multi_lv_loop[t][0][1].iterate, oloop.iterate)
                            loop1 = oloop
                            for i in range(len(multi_lv_loop[t])):
                                tloop = Loop(0, scalar_D, 1,[])
                                loop1.body.append(tloop)
                                if i+1 < len(multi_lv_loop[t]):
                                    for jj in range(len(t.body)):
                                        t.body[jj] = swap_arr_to_reg(t.body[jj], multi_lv_loop[t][i+1][1].iterate, tloop.iterate)
                                    loop1 = tloop
                            for jj in range(len(t.body)):
                                t.body[jj] = swap_arr_to_reg(t.body[jj], t.iterate, tloop.iterate)
                            # print('...............', oloop, codegen.gpu.to_string(tloop), codegen.gpu.to_string(t))
                            tloop.body = t.body
                            temp_body.append(oloop)
                        else:
                            # add tiled loop here
                            item = swap_arr_to_reg(item, t.iterate, new_loop.iterate)
                            new_loop.body.append(item)
                    if new_loop.body != []:
                        temp_body.append(new_loop)
                    tile_list[lidx].body = temp_body
                # while isinstance(temp, Loop) and temp.end.name() == 'dim':
                #     scalar_D = Scalar('int', 'D')
                #     temp.step = scalar_D
                #     new_loop = Loop(0, scalar_D, 1,[])
                #     tile_list.append([new_loop, temp.iterate])
                #     if not isinstance(temp.body[0], Loop):
                #         oloop = temp
                    
                #     new_loop2 = Loop(0, scalar_D, 1,[])
                #     if len(temp.body) > 1:
                #         for oidx in range(1, len(temp.body)):
                #             swap_arr_to_reg(temp.body[oidx], temp.iterate, new_loop2.iterate)
                #             new_loop2.body.append(temp.body[oidx])
                #         tbody = [temp.body[0], new_loop2]
                #         temp.body = tbody
                #     temp = temp.body[0]
                # # temp is assign, oloop is outloop of it
                # print('tile list len:', tile_list)
                # for i in range(len(tile_list)):
                #     oloop.body = [tile_list[i][0]]
                #     temp = swap_arr_to_reg(temp, tile_list[i][1], tile_list[i][0].iterate)
                #     oloop = oloop.body[0]
                # oloop.body = [temp]
    return ir

def tile_loop(ast):
    if type(ast) == BatchOp:
        if type(ast.operators[1]) == BatchOp:
            tile_loop(ast.operators[1])
        if type(ast.operators[0]) == BatchOp:
            tile_loop(ast.operators[0])
    else:
        return 
    
    if ast.compute and ast.valid:
        # print(ast.compute[0], codegen.gpu.to_string(ast.compute[0]))
        for i in ast.compute:
            # recursive_tile(i)
            t = swap_loops_tile(i)
            # print(codegen.gpu.to_string(t))