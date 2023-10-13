from core.ir import *

# class Intersect(IR):
#     def __init__(self, first, first_size, second, second_size, res):
#         self.first = first
#         self.first_size = first_size
#         self.second = second
#         self.second_size = second_size
#         self.res = res

# class Filter(IR):
#     filter_id = 0
#     def __init__(self, input, output, condition, condition_body):
#         self.input = input
#         self.output = output
#         self.condition = condition
#         self.condition_body = condition_body
#         self.res_size = Scalar('int', f'_f{self.fid}')
#         self.fid = Filter.filter_id
#         Filter.filter_id += 1

class Condition(IR):
    condition_id = 0
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body
        self.cid = Condition.condition_id
        Condition.condition_id+=1

class Search(IR):
    search_id = 0
    def __init__(self, dobject, start, end, item):
        self.dobject = dobject
        self.start = start
        self.end = end
        self.item = item
        # self.res = res
        Search.search_id += 1

class Ref(IR):
    nrefs = 0
    def __init__(self, dobject):
        self.dobject = dobject
        self.ref_id = Ref.nrefs
        Ref.nrefs += 1
        self.dtype = self.dobject.dtype
        self.size = dobject.size[:]

    # def name(self):
    #     return f'ref{self.ref_id}_{self.dobject.name()}'

    # def addr(self):
    #     return self.name()

# class RefIndex(Index):
#     nindices = 0
#     def __init__(self, dobject, index=None, ind_arr=None):
#         super.__init__(dobject, index, ind_arr)
#         if ind_arr == None:
#             if type(dobject)==Ref:
#                 self.size = []