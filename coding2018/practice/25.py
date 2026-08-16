from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def print(self, lst):
        print("*")
        n = lst
        while n:
            print(n.val)
            n = n.next
            
    def reverse(self, tail, head):
        prev_node = None
        node = head
        while node:
            next = node.next
            node.next = prev_node
            prev_node = node
            node = next
        return tail
    
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # we need three pointers, prev_tail, tail, head
        if k <= 1: return head
        if not head: return head

        prev_tail = tail = None
        n = 0
        node = head
        prev_node = None
        while node:
            # print(node.val)
            n += 1
            if n == 1:
                tail = node
            if n == k:
                n = 0
                if prev_tail:
                    prev_tail.next = node
                else:
                    head = node
                prev_tail = tail

            next = node.next
            node.next = prev_node
            prev_node = node
            node = next

        tail.next = None
        if n != 0:
            lst = self.reverse(tail, prev_node)
            if prev_tail:
                prev_tail.next = lst
            else:
                head = lst
        return head

    def reverseKGroup_lazy(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if k <= 1: return head
        if not head: return head

        prev_head = None
        node = head
        n = 0
        new_head = head
        while node:
            n += 1
            next = node.next
            if n == k:
                n = 0
                node.next = None
                part = self.reverse(node, head)
                if prev_head:
                    prev_head.next = part
                else:
                    new_head = node
                prev_head = head
                head = next

            node = next
        if n != 0 and prev_head:
            prev_head.next = head
        return new_head
    
def mk_list(lst):
    node = None
    for num in lst[::-1]:
        node = ListNode(num, node)
    return node

if __name__ == '__main__':
    """
    mistake:
    1. infinite loop due to not reverse the list correctly, fix is to introudce prev_node.
    2. tail.next is not None.
    """

    sol = Solution()
    # sol.print(sol.reverseKGroup(mk_list([1,2,3,4,5]), 1))
    # sol.print(sol.reverseKGroup(mk_list([1,2,3,4,5]), 2))
    # sol.print(sol.reverseKGroup(mk_list([1,2,3,4,5]), 3))
    # sol.print(sol.reverseKGroup(mk_list([1,2,3,4,5]), 4))
    # sol.print(sol.reverseKGroup(mk_list([1,2,3,4,5]), 5))
    # sol.print(sol.reverseKGroup(mk_list([1,2,3,4,5]), 6))
    
    # sol.print(sol.reverseKGroup(mk_list([1,2,3,4,5,6]), 3))


    sol.print(sol.reverseKGroup_lazy(mk_list([1,2,3,4]), 2))
    sol.print(sol.reverseKGroup_lazy(mk_list([1,2,3,4,5]), 1))
    sol.print(sol.reverseKGroup_lazy(mk_list([1,2,3,4,5]), 2))
    sol.print(sol.reverseKGroup_lazy(mk_list([1,2,3,4,5]), 3))
    sol.print(sol.reverseKGroup_lazy(mk_list([1,2,3,4,5]), 4))
    sol.print(sol.reverseKGroup_lazy(mk_list([1,2,3,4,5]), 5))
    sol.print(sol.reverseKGroup_lazy(mk_list([1,2,3,4,5]), 6))
    
    sol.print(sol.reverseKGroup_lazy(mk_list([1,2,3,4,5,6]), 3))
