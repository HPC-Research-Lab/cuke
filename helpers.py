from apps import compression
from core.asg import *
from core.ir import *
from cset.ast import *

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
        elif type(node) == batch.ast.Batch:
            self._post_traverse(node.base, visited, res)
        elif type(node) == batch.ast.BatchOp:
            for c in node.operators:
                self._post_traverse(c, visited, res)
            self.action(node, res)
        # elif type(node) == compression.asg.Encoder:
        #     for s in node.fix_size:
        #         self._post_traverse(s, visited, res)
        #     for s in node.ref_size:
        #         self._post_traverse(s, visited, res)
        #     for c in node.operators:
        #         self._post_traverse(c, visited, res)
        #     self.action(node, res)
        #cset extension
        elif type(node) == Set:
            self._post_traverse(node.nelem, visited, res)
            self._post_traverse(node.storage, visited, res)
            self.action(node, res)
        elif type(node) == SetOp:
            for c in node.operators:
                self._post_traverse(c, visited, res)
            self._post_traverse(node.nelem, visited, res)
            self._post_traverse(node.storage, visited, res)
            self.action(node, res)

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
        if isinstance(node, (Tensor, Set)):
            res.extend(node.decl)
            res.extend(node.compute)

    t = ASGTraversal(action)
    ir.extend(t(ast))


def new_op(func):
    def wrapper_func(*args, **kwargs):
        res = func(*args, **kwargs)
        res.attr[func.__name__] = True
        return res
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
        elif type(stmt) == Loop:
            cond = self.action(stmt, res)
            if cond[0]:
                self._preorder_traverse(stmt.start, res)
            if cond[1]:
                self._preorder_traverse(stmt.end, res)
            if cond[2]:
                self._preorder_traverse(stmt.step, res)
            if cond[3]:
                self._preorder_traverse(stmt.body, res)
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
        return [True, True, True, True]

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
            return [True, True, True, True]

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
            return [True, True, True, True]

    t = IRTraversal(action)
    r = t(ir)
    if len(r) > 0 and r[0] == True:
        return True
    else:
        return False