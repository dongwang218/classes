
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

diameter = None        
def get_diameter(node):
    global diameter
    if node is None:
        return 0
    d_left = get_diameter(node.left)
    d_right = get_diameter(node.right)
    d2 = d_left+d_right+1
    if diameter is None or diameter < d2:
        diameter = d2
        
    return max(d_left, d_right) + 1

def get_dia(node):
    def helper(node):
        if node is None:
            return (0, 0)
        depth1, dia1 = helper(node.left)
        depth2, dia2 = helper(node.right)

        return max([depth1, depth2])+1, max([dia1, dia2, depth1+depth2+1])

    return helper(node)[1]
"""
            1 
          /   \
         2     3 
        / \     
       4   5     
      /     \
     6       7
    /         \
   8           9
"""
t = Node(1, Node(2, Node(4, Node(6, Node(8))), Node(5, None, Node(7, None, Node(9)))), Node(3))
get_diameter(t)
assert diameter  == 7, "depth is {}".format(diameter)

assert get_dia(t) == 7
