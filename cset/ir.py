from core.ir import *

class FilterLoop(IR):
    filter_id = 0
    def __init__(self, start, end, step,  
                body = [], 
                cond = None, 
                cond_body = []):
        self.fid = FilterLoop.filter_id
        FilterLoop.filter_id += 1
        
        self.start = start
        self.end = end
        self.step = step
        self.iterate = Scalar('int', f'_fl{self.fid}')

        self.body = body
        self.cond = cond
        self.cond_body = cond_body

class BinarySearch(IR):
    search_id = 0
    def __init__(self, dobject, start, end, item):
        self.dobject = dobject
        self.start = start
        self.end = end
        self.item = item
        # self.res = res
        BinarySearch.search_id += 1

class Not(IR):
    def __init__(self, dobject):
        self.dobject = dobject

# class 