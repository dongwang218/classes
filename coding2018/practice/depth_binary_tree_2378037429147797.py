from __future__ import annotations

"""
None
depth: 0

1
depth: 1

    1
   /
  2
 /
3
depth: 3

  1
 / \
 2  3
depth: 2

    1
  /    \
 2      3
  \    /
   4  5
       \
        6 
depth 4
"""

from typing import Optional

class Node:
    def __init__(self, value: int, left: Optional[Node] = None, right: Optional[Node] = None):
        self.value = value
        self.left = left
        self.right = right

def depth_tree(node: Optional[Node]) -> int:
    if node is None:
        return 0

    return 1 + max(depth_tree(node.left), depth_tree(node.right))

if __name__ == '__main__':

    assert depth_tree(Node(1, Node(2, Node(3)))) == 3
    assert depth_tree(Node(1, Node(2), Node(3))) == 2
    assert depth_tree(Node(1, Node(2, None, Node(4)), Node(3, Node(5, None, Node(6))))) == 4
