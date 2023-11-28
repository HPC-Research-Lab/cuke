from cset.ast2ir import *
from codegen.cpu import *

# def move_ir(intermediate_sets, apply_nodes, ir_list, level, remove_record):
#     if level>=len(apply_nodes):
#         return
#     intermediate_ops = intermediate_sets[level-1] if level>0 else []
#     remove_record.append([])
#     ir_set = set()
#     for ir in ir_list:
#         if ir.astnode in intermediate_ops: 
#             for sub_ir in get_ir(ir.astnode):
#                 if sub_ir in ir_list and sub_ir not in ir_set:
#                     ir_set.add(sub_ir)
#                     remove_record[level].append(sub_ir)  
#         if type(ir)==Loop:            
#             assert ir.astnode==apply_nodes[level]
#             move_ir(intermediate_sets, apply_nodes, ir.body, level+1, remove_record)
            
# def move_ir2(ir_list, remove_record, level):
#     if level>=len(remove_record):
#         return

#     remove_record_this_level = remove_record[level]
#     for ir in remove_record_this_level:
#         ir_list.remove(ir)

#     loop_ir = None
#     for ir in ir_list:
#         if type(ir)==Loop:  
#             loop_ir = ir          
#             move_ir2(ir.body, remove_record, level+1)

#     remove_record_next_level = remove_record[level+1] if level < len(remove_record)-1 else []
#     for ir in remove_record_next_level:
#         ir_list.insert(ir_list.index(loop_ir), ir)

def PrintCCode(ir):
    code = ''
    for d in ir:
        if d:
            code += to_string(d)
    print(code)
    return ir

def revalid(ast):
    def action(node, res):
         node.valid = True
    t = helpers.Traversal(action)
    ir = t(ast)

def get_ir(ast):
    def action(node, res):
        # if node.valid == True:
        if type(node) == Var or type(node) == Tensor:
            res.extend(node.decl)
        elif type(node) == TensorOp:
            res.extend(node.decl)
            res.extend(node.compute)
        elif type(node) == Set:
            res.extend(node.decl)
        elif type(node) == SetOp:
            res.extend(node.decl)
            res.extend(node.compute)

    t = helpers.Traversal(action)
    ret_ir = t(ast)
    ret_decl = []
    ret_compute = []

    for ir in ret_ir:
        if type(ir) == Decl:                       
            ret_decl.extend([ir])
        else:
            ret_compute.extend([ir])
    return ret_decl, ret_compute

def get_ops(ast):
    if type(ast)==SetOp and ast.op_type=='apply':
        partial_res = ast.operators[0]
        return partial_res

def is_same_operation(ast1, ast2):
    if type(ast1)==Set and type(ast2)==Set and ast1==ast2:
        return True
    if type(ast1) != type(ast2):
        return False

    if ast1.operators[5].operators[0] == ast2.operators[5].operators[0]:
        if is_same_operation(ast1.operators[0], ast2.operators[0]):
            return True
    return False

def extract_intermediate_sets(ast, level, intermediate_sets, ast_loops):
    if type(ast) != SetOp or ast.op_type!='apply':
        return
    input_setop = ast.operators[0] # Last Set operation at top
    if type(input_setop)==SetOp and input_setop.op_type=='filter':
        input_setop = input_setop.operators[0]
    
    input_func = ast.operators[6]

    intermediate_sets.extend([[]])
    ast_loops.append(ast)
    extract_intermediate_sets(input_func, level+1, intermediate_sets, ast_loops)

    if level>=2:
        for setops in [input_setop] + intermediate_sets[level]:
            subexpr1 = setops.operators[0]
            find = False
            for subexpr2 in intermediate_sets[level-1]:
                if is_same_operation(subexpr1, subexpr2):
                    find = True
            if find == False:
                intermediate_sets[level-1].append(subexpr1)


def move_ir_on_ast(cur_node, pre_node, level, intermediate_sets, apply_nodes):
    if type(cur_node) != SetOp or cur_node.op_type!='apply':
        return    

    move_ir_on_ast(cur_node.operators[6], cur_node, level+1, intermediate_sets, apply_nodes)

    intermediate_ops = intermediate_sets[level] if level>0 else []
    intermediate_irs = []
    for item in intermediate_ops:
        ir_decl, ir_compute = get_ir(item)
        intermediate_irs.append(ir_compute)

    assert type(cur_node.compute[0])==Loop
    pre_body = pre_node.compute[0].body if pre_node else []
    cur_body = cur_node.compute[0].body

    pre_loop_ir = None
    for item in pre_body:
        if type(item)==Loop:
            pre_loop_ir = item

    loop_ir = None
    for intermediate_ir in intermediate_irs:
        remove_list = []
        for item in cur_body:
            if item in intermediate_ir:
                remove_list.append(item)
        for item in remove_list:
            pre_body.insert(pre_body.index(pre_loop_ir), item)
            cur_body.remove(item)



def has_item(node, item):
    assert type(node)==Set
    # assert type(node.storage)==Indexing
    colidx = node.storage.operators[0]
    colidx_slice = node.storage.operators[1]
    start = colidx_slice.val.start
    end = colidx_slice.val.stop
    assert(start.operators[0]==end.operators[0])
    rowptr = start.operators[0]
    v = start.operators[1]
    if item==v:
        return True
    else:
        return False

def extract_subtree(setops, item):
    if type(setops)==Set:
        if not has_item(setops, item):
            return setops
        else:
            return None

    left_tree = setops.operators[0]
    right_set = setops.operators[5].operators[0]
    res = extract_subtree(left_tree, item)
    if res==left_tree and not has_item(right_set, item):
        return setops
    else:
        return left_tree

def get_vertex_name(node):
    colidx = node.storage.operators[0]
    colidx_slice = node.storage.operators[1]
    start = colidx_slice.val.start
    end = colidx_slice.val.stop
    assert(start.operators[0]==end.operators[0])
    rowptr = start.operators[0]
    v = start.operators[1]
    return v.name

def is_same_operation2(ast1, ast2, level):
    if type(ast1)==Set and type(ast2)==Set and get_vertex_name(ast1)==get_vertex_name(ast2):
        return True
    if type(ast1) != type(ast2):
        return False

    if get_vertex_name(ast1.operators[5].operators[0]) == get_vertex_name(ast2.operators[5].operators[0]):
        if is_same_operation2(ast1.operators[0], ast2.operators[0], level):
            return True
    return False

def extract_intermediate_sets2(ast, level, intermediate_sets, ast_loops):
    if type(ast) != SetOp or ast.op_type!='apply':
        return
    input_setop = ast.operators[0] # Last Set operation at top
    if type(input_setop)==SetOp and input_setop.op_type=='filter':
        input_setop = input_setop.operators[0]
    
    input_func = ast.operators[6]

    intermediate_sets.extend([[]])
    ast_loops.append(ast)
    extract_intermediate_sets2(input_func, level+1, intermediate_sets, ast_loops)

    if level>=2:
        pre_apply_node = ast_loops[level-1]
        item = pre_apply_node.operators[5]

        for setops in [input_setop] + intermediate_sets[level]:
            subexpr1 = extract_subtree(setops, item)
            assert(subexpr1==setops.operators[0])
            find = False

            for subexpr2 in  intermediate_sets[level-1]:
                if is_same_operation2(subexpr1, subexpr2, level):
                    find = True
            if find == False:
                intermediate_sets[level-1].append(subexpr1)
    
def rebind(index: (Indexing, Ndarray, Slice), idx):
    return Indexing(index.dobject, idx)

def replace_input(node, input):
    assert(type(node)==SetOp)
    assert(node.op_type=='intersection' or node.op_type=='difference')

    node_loop_ir = None
    for ir in node.compute:
        if type(ir)==Loop:
            node_loop_ir = ir
    assert(type(node_loop_ir.body[0]==Code))
    node_code = node_loop_ir.body[0]
    keywords = node_code.keywords


    input_loop_ir = None
    for ir in input.compute:
        if type(ir)==Loop:
            input_loop_ir = ir
    assert(type(input_loop_ir.body[0]==Code))
    input_pos_ir = input_loop_ir.body[0].keywords['pos_increment'].left

    # print(to_string(node_loop_ir.end))
    # print(to_string(input_pos_ir))

    # print(to_string(keywords['first_smaller_second'].left))
    # print(to_string(input.eval))
    node_loop_ir.end = input_pos_ir
    keywords['first_smaller_second'].left = rebind(input.eval, node_loop_ir.iterate)
    keywords['first_larger_second'].left = rebind(input.eval, node_loop_ir.iterate)
    keywords['first_equal_second'].left = rebind(input.eval, node_loop_ir.iterate)
    keywords['assignment'].rhs = rebind(input.eval, node_loop_ir.iterate)


def move_ir_on_ast2(cur_node, pre_node, level, intermediate_sets, apply_nodes):
    if type(cur_node) != SetOp or cur_node.op_type!='apply':
        return    

    move_ir_on_ast2(cur_node.operators[6], cur_node, level+1, intermediate_sets, apply_nodes)

    intermediate_ops = intermediate_sets[level] if level>0 else []
    intermediate_irs = []
    delete_irs = []
    for item in intermediate_ops:
        delete_irs_next_level = []
        for tree_next_level in [cur_node.operators[6].operators[0].operators[0]] + intermediate_sets[level+1]:
            left_subtree = tree_next_level.operators[0]
            if left_subtree!=item:
                replace_input(tree_next_level, item)
            if is_same_operation2(item, left_subtree, level):
                delete_ir_decl, delete_ir_compute = get_ir(left_subtree)
                delete_irs_next_level.extend(delete_ir_compute)
        
        ir_decl, ir_compute = get_ir(item)
        intermediate_irs.append(ir_compute)
        delete_irs.append(delete_irs_next_level)

    def _get_loop_body(node):
        if not node:
            return []
        for i in node.compute:
            if type(i)==Loop:
                return i.body
        return []

    pre_body = _get_loop_body(pre_node)#pre_node.compute[0].body if pre_node else []
    cur_body = _get_loop_body(cur_node)#cur_node.compute[0].body

    pre_loop_ir = None
    for item in pre_body:
        if type(item)==Loop:
            pre_loop_ir = item

    loop_ir = None
    for delete_ir in delete_irs:
        delete_list = []
        for item in cur_body:
            if item in delete_ir:
                delete_list.append(item)
        for item in delete_list:
            # pre_body.insert(pre_body.index(pre_loop_ir), item)
            cur_body.remove(item)
    
    for intermediate_ir in intermediate_irs:
        for item in intermediate_ir:
            pre_body.insert(pre_body.index(pre_loop_ir), item)

def code_motion(ast):
    ast = ast.operators[0]
    assert(type(ast)==SetOp and ast.op_type=='apply')

    intermediate_sets = []
    apply_nodes = []
    remove_record = []
    extract_intermediate_sets2(ast, 0, intermediate_sets, apply_nodes)

    move_ir_on_ast2(ast, None, 0, intermediate_sets, apply_nodes)

    # move_ir(intermediate_sets, apply_nodes, ir, 0, remove_record)
    # move_ir2(ir, remove_record, 0)
    # for k in remove_record:
    #     print(len(k))







