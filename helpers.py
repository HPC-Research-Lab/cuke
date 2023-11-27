from apps import compression
from core.asg import *
from core.ir import *

class ASGTraversal:

    def __init__(self, action):
        self.action = action

    def _post_traverse(self, node, visited, res):
        import batch
        if not isinstance(node, ASTNode):
            return
        if node in visited:
            return
        else:
            visited.add(node)

        if type(node) == Var:
            self.action(node, res)
        elif type(node) == Const:
            if node.dtype == 'slice':
                self._post_traverse(node.val.start, visited, res)
                self._post_traverse(node.val.stop, visited, res)
                self._post_traverse(node.val.step, visited, res)
        elif type(node) == Tensor:
            for s in node.fix_size:
                self._post_traverse(s, visited, res)
            for s in node.ref_size:
                self._post_traverse(s, visited, res)
            self.action(node, res)
        elif type(node) == TensorOp:
            for s in node.fix_size:
                self._post_traverse(s, visited, res)
            for s in node.ref_size:
                self._post_traverse(s, visited, res)
            for c in node.operators:
                self._post_traverse(c, visited, res)
            self.action(node, res)
        # elif type(node) == batch.ast.Batch:
        #     self._post_traverse(node.base, visited, res)
        # elif type(node) == batch.ast.BatchOp:
        #     for c in node.operators:
        #         self._post_traverse(c, visited, res)
        #     self.action(node, res)
        # elif type(node) == compression.asg.Encoder:
        #     for s in node.fix_size:
        #         self._post_traverse(s, visited, res)
        #     for s in node.ref_size:
        #         self._post_traverse(s, visited, res)
        #     for c in node.operators:
        #         self._post_traverse(c, visited, res)
        #     self.action(node, res)

    def __call__(self, ast):
        visited = set()
        res = []
        self._post_traverse(ast, visited, res)
        return res

def get_input_nodes(ast):
    def action(node, res):
        if type(node) == Var or type(node) == Tensor:
            if node.is_arg:
                res.append([node.name, node])

    t = ASGTraversal(action)
    return dict(t(ast))

def get_ir_of_size(size):
    ir_size = []
    for s in size:
        assert isinstance(s, ASTNode)
        s._gen_ir()
        ir_size.append(s.eval)
    return ir_size

def collect_ir(ast, ir):
    import batch
    def action(node, res):
        if isinstance(node, Tensor):
            res.extend(node.decl)
            res.extend(node.compute)

    t = ASGTraversal(action)
    ir.extend(t(ast))


def new_op(func):
    def wrapper_func(*args, **kwargs):
        _res = func(*args, **kwargs)
        _res.attr[func.__name__] = True
        return _res
    return wrapper_func


def get_obj(ir: (Indexing, Scalar)):
    obj = ir
    while hasattr(obj, 'dobject'):
        obj = obj.dobject
    return obj

def get_val(ir):
    if type(ir) == Literal:
        return ir.val
    elif type(ir) in (int, float):
        return ir
    else:
        return ir


class IRTraversal:

    def __init__(self, action):
        self.action = action

    def _preorder_traverse(self, stmt, res):
        if type(stmt) == list or type(stmt) == tuple:
            cond = self.action(stmt, res)
            if cond[0]:
                for l in stmt:
                    self._preorder_traverse(l, res)
        elif isinstance(stmt, Loop):
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.start, res)
            if cond[1]:
                self._preorder_traverse(stmt.end, res)
            if cond[2]:
                self._preorder_traverse(stmt.step, res)
            if cond[3]:
                self._preorder_traverse(stmt.body, res)
            if type(stmt) == FilterLoop and cond[4]:
                self._preorder_traverse(stmt.cond, res)
        elif type(stmt) == Expr:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.left, res)
            if cond[1]:
                self._preorder_traverse(stmt.right, res)
        elif type(stmt) == Assignment:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.lhs, res)
            if cond[1]:
                self._preorder_traverse(stmt.rhs, res)
        elif type(stmt) == Ndarray:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.size, res)
        elif type(stmt) == Scalar:
            self.action(stmt, res)
        elif type(stmt) == Indexing:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.dobject, res)
            if cond[1]:
                self._preorder_traverse(stmt.idx, res)
        elif type(stmt) == Slice:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.start, res)
            if cond[1]:
                self._preorder_traverse(stmt.stop, res)
            if cond[2]:
                self._preorder_traverse(stmt.step, res)
        elif type(stmt) == Math:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.val, res)
        elif type(stmt) == Code:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.output[1], res)
            if cond[1]:
                for k in stmt.inputs:
                    self._preorder_traverse(stmt.inputs[k], res)

    def __call__(self, ir):
        res = []
        self._preorder_traverse(ir, res)
        return res


def rebind_iterate(ir, old, new):
    def action(stmt, res):
        if type(stmt) == Indexing and type(stmt.idx) in (Scalar, Literal):
            if stmt.idx.dobject_id == old.dobject_id:
                stmt.idx = new
        return [True, True, True, True]

    t = IRTraversal(action)
    t(ir)

def replace_all_ref(ir, old, new):
    def action(stmt, res):
        match stmt.__class__.__name__:
            case 'Loop':
                if stmt.start == old:
                    stmt.start = new
                if stmt.end == old:
                    stmt.end = new
                if stmt.step == old:
                    stmt.step = new
            case 'FilterLoop':
                if stmt.cond == old:
                    stmt.cond = new
                if stmt.start == old:
                    stmt.start = new
                if stmt.end == old:
                    stmt.end = new
                if stmt.step == old:
                    stmt.step = new
            case 'Expr':
                if stmt.left == old:
                    stmt.left = new
                if stmt.right == old:
                    stmt.right = new
            case 'Assignment':
                if stmt.lhs == old:
                    stmt.lhs = new
                if stmt.rhs == old:
                    stmt.rhs = new
            case 'Indexing':
                if stmt.dobject == old:
                    stmt.dobject = new
                if stmt.idx == old:
                    stmt.idx = new
            case 'Slice':
                if stmt.start == old:
                    stmt.start = new
                if stmt.stop == old:
                    stmt.stop = new
                if stmt.step == old:
                    stmt.step = new
            case 'Math':
                if stmt.val == old:
                    stmt.val = new
            case 'Code':
                if stmt.output[1] == old:
                    stmt.output = (stmt.output[0], new)
                for k in stmt.inputs:
                    if stmt.inputs[k] == old:
                        stmt.inputs[k] = new
        return [True, True, True, True, True]

    t = IRTraversal(action)
    t(ir)

def ir_uses(ir, data):
    def action(stmt, res):
        if stmt == data or (isinstance(stmt, DObject) and stmt.dobject_id == data.dobject_id):
            if len(res) == 0:
                res.append(True)
            else:
                res[0] = True
        if type(stmt) == Assignment:
            if stmt.op != None:
                return [True, True]
            else:
                return [False, True]
        else:
            return [True, True, True, True, True]

    t = IRTraversal(action)
    r = t(ir)
    if len(r) > 0 and r[0] == True:
        return True
    else:
        return False


def ir_defs(ir, data):
    def action(stmt, res):
        if stmt == data or (isinstance(stmt, DObject) and stmt.dobject_id == data.dobject_id):
            if len(res) == 0:
                res.append(True)
            else:
                res[0] = True
        if type(stmt) == Assignment:
            return [True, False]
        else:
            return [True, True, True, True, True]

    t = IRTraversal(action)
    r = t(ir)
    if len(r) > 0 and r[0] == True:
        return True
    else:
        return False


def remove_decl(node, item):
    def action(n, res):
        n.decl = [d for d in n.decl if d.dobject != item]
    t = ASGTraversal(action)
    t(node)


def _remove_compute_of_node(loop, node):
    body = []
    for stmt in loop.body:
        if not ('asgnode' in stmt.attr and stmt.attr['asgnode'] == node):
            if type(stmt) == Loop:
                _remove_compute_of_node(stmt, node)
            body.append(stmt)
    loop.body = body



def clear_compute(node):
    node.compute.clear()
    if 'scope' in node.attr:
        scope = node.attr['scope']
        compute = []
        for stmt in scope.compute:
            if not ('asgnode' in stmt.attr and stmt.attr['asgnode'] == node):
                if type(stmt) == Loop:
                    _remove_compute_of_node(stmt, node)
                compute.append(stmt)
        scope.compute = compute


def ir_find_defs(ir, data):
    def action(stmt, res):
        if type(stmt) == Assignment and get_obj(stmt.lhs) == data:
            res.append(stmt)
        elif type(stmt) == Code and get_obj(stmt.output[1]) == data:
            res.append(stmt)

        return [True, True, True, True, True]

    t = IRTraversal(action)
    return t(ir)


def asg_find_defs(asg, data):
    def action(node, res):
        if type(node) == TensorOp:
            res.extend(ir_find_defs(node.compute, data))

    t = ASGTraversal(action)
    return t(asg)