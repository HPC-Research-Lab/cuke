import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test import examples
from cset.app import subgraph_matching

if __name__ == "__main__":
    subgraph_matching.merge_search(sys.argv[1])