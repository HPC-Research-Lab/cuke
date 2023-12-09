import transform.fuse
import transform.interchange
import transform.parallelize


passes = []

fu = fuse.fuser()
fu.register(fuse.basic_rule)
passes.append(fu)