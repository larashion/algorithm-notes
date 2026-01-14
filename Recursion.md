#### 21. Merge Two Sorted Lists

go

```go
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
    if list1 == nil {return list2}
    if list2 == nil {return list1}
    if list1.Val < list2.Val {
        list1.Next = mergeTwoLists(list2,list1.Next)
        return list1
    }
    list2.Next = mergeTwoLists(list1,list2.Next)
    return list2
}
```

python

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        if list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        list2.next = self.mergeTwoLists(list2.next, list1)
        return list2
```

#### 24. Swap Nodes in Pairs

java

```java
class Solution {
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode tmp = head.next;
        head.next = swapPairs(head.next.next);
        tmp.next = head;
        return tmp;
    }
}
```

#### 50. Pow(x, n)

```go
func myPow(x float64, n int) float64 {
    if n < 0 {
        n = -n
        x = 1 / x
    }
    if n == 0 {
        return 1
    }
    if n&1 == 1 {
        return x * myPow(x*x, n>>1)
    }
    return myPow(x*x, n>>1)
}
```

#### 203. Remove Linked List Elements

go

```go
func removeElements(head *ListNode, val int) *ListNode {
	if head == nil {
		return nil
	}
	head.Next = removeElements(head.Next, val)
	if head.Val == val {
		return head.Next
	}
	return head
}
```

#### 258. Add Digits

rust

```rust
impl Solution {
    pub fn add_digits(num: i32) -> i32 {
        if num / 10 == 0 {
            return num;
        }
        Solution::add_digits(num / 10 + num % 10)
    }
}
```

