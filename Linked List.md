#### 725. Split Linked List in Parts

java

```java
class Solution {
    public ListNode[] splitListToParts(ListNode head, int k) {
        int len = length(head);
        int partSize = len / k, remain = len % k;
        ListNode[] parts = new ListNode[k];
        ListNode curr = head, prev = null;
        for (int i = 0; i < k && curr != null; i++) {
            parts[i] = curr;
            int size = partSize;
            if (remain > 0) {
                remain--;
                size++;
            }
            for (int j = 0; j < size; j++) {
                prev = curr;
                curr = curr.next;
            }
            prev.next = null;
        }
        return parts;
    }

    private int length(ListNode head) {
        int res = 0;
        for (ListNode curr = head; curr != null; curr = curr.next) {
            res++;
        }
        return res;
    }
}
```

go

```go
func splitListToParts(head *ListNode, k int) []*ListNode {
	listLength := length(head)
	partSize, remain := listLength/k, listLength%k
	parts := make([]*ListNode, k)
	curr := head
	var prev *ListNode
	for i := 0; i < k && curr != nil; i++ {
		parts[i] = curr
		size := partSize
		if remain > 0 {
			size++
			remain--
		}
		for j := 0; j < size; j++ {
			prev = curr
			curr = curr.Next
		}
		prev.Next = nil
	}
	return parts
}

func length(head *ListNode) int {
	res := 0
	for curr := head; curr != nil; curr = curr.Next {
		res++
	}
	return res
}
```

#### 382. Linked List Random Node

java

```java
class Solution {
    Random random;
    ListNode head;

    public Solution(ListNode head) {
        this.head = head;
        random = new Random();
    }

    public int getRandom() {
        ListNode res = head;
        int i = 0;
        for (ListNode curr = head; curr != null; curr = curr.next) {
            if (random.nextInt(i + 1) == i) {
                res = curr;
            }
            i++;
        }
        return res.val;
    }
}
```

#### 19. Remove Nth Node From End of List

go

```go
func removeNthFromEnd(head *ListNode, n int) *ListNode {  
   dummyHead := &ListNode{Next: head}  
   preSlow, slow, fast := dummyHead, head, head  
   for ; fast != nil; fast = fast.Next {  
      if n <= 0 {  
         preSlow, slow = slow, slow.Next  
      }  
      n--  
   }  
   preSlow.Next = slow.Next  
   return dummyHead.Next  
}
```

python

```python
class Solution:  
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:  
        dummy = ListNode(next=head)  
        prev, slow, fast = dummy, head, head  
        while fast:  
            if n <= 0:  
                prev, slow = slow, slow.next  
            fast = fast.next  
            n -= 1  
        prev.next = slow.next  
        return dummy.next
```

java

```java
class Solution {  
    public ListNode removeNthFromEnd(ListNode head, int n) {  
        ListNode dummy = new ListNode(-1, head);  
        ListNode slow = head, fast = head, prev = dummy;  
        while (fast != null) {  
            if (n <= 0) {  
                prev = slow;  
                slow = slow.next;  
            }  
            fast = fast.next;  
            n--;  
        }  
        prev.next = slow.next;  
        return dummy.next;  
    }  
}
```

#### 24. Swap Nodes in Pairs

Go

```go
func swapPairs(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	prev := dummy
	for prev.Next != nil && prev.Next.Next != nil {
		fir := prev.Next
		sec := fir.Next
		prev.Next, sec.Next, fir.Next = sec, fir, sec.Next
		prev = fir
	}
	return dummy.Next
}
```

java

```java
class Solution {
    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(-1, head);
        ListNode prev = dummy;
        while (prev.next != null && prev.next.next != null) {
            ListNode fir = prev.next;
            ListNode sec = fir.next;

            fir.next = sec.next;
            sec.next = fir;
            prev.next = sec;
            prev = fir;
        }
        return dummy.next;
    }
}
```

#### 25. Reverse Nodes in k-Group

https://leetcode.com/problems/reverse-nodes-in-k-group/discuss/11491/Succinct-iterative-Python-O(n)-time-O(1)-space

go

```go
func reverseKGroup(head *ListNode, k int) *ListNode {
	dummy := &ListNode{Next: head}
	left, right, tail := head, head, dummy
	for {
		count := 0
		for ; right != nil && count < k; right = right.Next {
			count++
		}
		if count == k {
			prev := reverse(left, right)
			tail.Next, tail, left = prev, left, right
		} else {
			return dummy.Next
		}
	}
}

// reverse nodes between [left, right)
func reverse(left, right *ListNode) *ListNode {
	// right is the start in next k-group
	prev := right
	for curr := left; curr != right; {
		curr.Next, curr, prev = prev, curr.Next, curr
	}
	return prev
}
```

java

```java
class Solution {

    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode left = head, right = head, dummy = new ListNode(-1, head), tail = dummy;
        while (true) {
            int count = 0;
            for (; count < k && right != null; right = right.next) {
                count++;
            }
            if (count == k) {
                tail.next = reverseList(left, right);
                tail = left;
                left = right;
            } else return dummy.next;
        }
    }

    ListNode reverseList(ListNode left, ListNode right) {
        ListNode prev = right;
        for (ListNode curr = left; curr != right; ) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }
}
```

python

```python
class Solution:  
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:  
        dummy = ListNode(next=head)  
        left, right, jump = head, head, dummy  
        while True:  
            count = 0  
            while right and count < k:  
                right = right.next  
                count += 1  
            if count == k:  
                prev, curr = right, left  
                for _ in range(k):  
                    curr.next, curr, prev = prev, curr.next, curr  
                jump.next, jump, left = prev, left, right  
            else:  
                return dummy.next
```

#### 61. Rotate List

dummy

```go
func rotateRight(head *ListNode, k int) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    length := 2
    dummy := &ListNode{Next: head}
    fast, slow := head.Next, dummy
    for ; fast.Next != nil; fast = fast.Next {
        length++
    }
    if k%length == 0 {
        return head
    }
    for m := length - k%length; m > 0; m-- {
        slow = slow.Next
    }
    slow.Next, fast.Next, dummy.Next = nil, dummy.Next, slow.Next
    return dummy.Next
}
```

#### 82. Remove Duplicates from Sorted List II

```go
func deleteDuplicates(head *ListNode) *ListNode {
   if head == nil || head.Next == nil {
      return head
   }
   dummy := &ListNode{Next: head, Val: -200}
   curr, prev := head, dummy
   for curr != nil {
      for curr.Next != nil && curr.Val == curr.Next.Val {
         curr = curr.Next
      }
      if prev.Next == curr {
         prev = prev.Next
      } else {
         prev.Next = curr.Next
      }
      curr = curr.Next
   }
   return dummy.Next
}
```

#### 83. Remove Duplicates from Sorted List

```go
func deleteDuplicates(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    var prev *ListNode
    for curr := head; curr != nil; curr = curr.Next {
        if prev != nil && curr.Val == prev.Val {
            prev.Next = curr.Next
        } else {
            prev = curr
        }
    }
    return head
}
```

#### 86. Partition List

java

```java
class Solution {
    public ListNode partition(ListNode head, int x) {
        ListNode dummy1 = new ListNode(), pt1 = dummy1;
        ListNode dummy2 = new ListNode(), pt2 = dummy2;
        while (head != null) {
            if (head.val < x) {
                pt1.next = head;
                pt1 = pt1.next;
            } else {
                pt2.next = head;
                pt2 = pt2.next;
            }
            head = head.next;
        }
        pt1.next = dummy2.next;
        pt2.next = null;
        return dummy1.next;
    }
}
```

Go

```go
func partition(head *ListNode, x int) *ListNode {
	dummy1, dummy2 := &ListNode{}, &ListNode{}
	pt1, pt2 := dummy1, dummy2
	for ; head != nil; head = head.Next {
		if head.Val < x {
			pt1.Next = head
			pt1 = pt1.Next
		} else {
			pt2.Next = head
			pt2 = pt2.Next
		}
	}
	pt1.Next = dummy2.Next
	pt2.Next = nil
	return dummy1.Next
}
```

#### 92. Reverse Linked List II

go

```go
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	dummy := &ListNode{Next: head}
	curr, prev := dummy, dummy
	for i := 0; i < right; i++ {
		if i == left-1 {
			prev = curr
		}
		curr = curr.Next
	}
	prev.Next = reverseList(prev.Next, curr.Next)
	return dummy.Next
}
func reverseList(head, tail *ListNode) *ListNode {
	var prev *ListNode
	for curr := head; curr != tail; {
		prev, curr, curr.Next = curr, curr.Next, prev
	}
	head.Next = tail
	return prev
}
```

java

```java
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode(-501, head);
        ListNode curr = dummy, prev = dummy;
        for (int i = 0; i < right; i++) {
            if (i == left - 1) {
                prev = curr;
            }
            curr = curr.next;
        }
        prev.next = reverse(prev.next, curr.next);
        return dummy.next;
    }

    private ListNode reverse(ListNode head, ListNode tail) {
        ListNode prev = null;
        for (ListNode curr = head; curr != tail; ) {
            ListNode tmp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = tmp;
        }
        head.next = tail;
        return prev;
    }
}
```

#### 138. Copy List with Random Pointer

Go

```go
func copyRandomList(head *Node) *Node {
	hashmap := make(map[*Node]*Node)
	for curr := head; curr != nil; curr = curr.Next {
		hashmap[curr] = &Node{Val: curr.Val}
	}
	for src, dst := range hashmap {
		dst.Next = hashmap[src.Next]
		dst.Random = hashmap[src.Random]
	}
	return hashmap[head]
}
```

java

```java
class Solution {
    public Node copyRandomList(Node head) {
        HashMap<Node, Node> map = new HashMap<>();
        for (Node curr = head; curr != null; curr = curr.next) {
            map.put(curr, new Node(curr.val));
        }
        for (Node node : map.keySet()) {
            Node dup = map.get(node);
            dup.next = map.get(node.next);
            dup.random = map.get(node.random);
        }
        return map.get(head);
    }
}
```

#### 141. Linked List Cycle

判断环形链表

快慢指针相遇的话，说明有环

如果快指针走完了，都没碰到一起，说明没有环

不用判断慢指针走不走到头

```go
func hasCycle(head *ListNode) bool {
    fast, slow := head, head
    for fast != nil && fast.Next != nil {
       slow, fast = slow.Next, fast.Next.Next
        if fast == slow {
            return true
        }
    }
    return false
}
```

#### 142. Linked List Cycle II

go

```go
func detectCycle(head *ListNode) *ListNode {
   if head == nil || head.Next == nil {
      return nil
   }
   fast, slow := head, head
   for fast != nil && fast.Next != nil {
      fast = fast.Next.Next
      slow = slow.Next
      if fast == slow {
         for head != fast {
            fast = fast.Next
            head = head.Next
         }
         return head
      }
   }
   return nil
}
```

#### 143. Reorder List

java

```java
class Solution {
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) return;
        ListNode mid = findMid(head);

        ListNode p2 = reverse(mid);
        ListNode p1 = head;
        while (p1 != mid) {
            ListNode tmp = p1.next;
            p1.next = p2;
            p1 = p2;
            p2 = tmp;
        }
    }

    ListNode reverse(ListNode head) {
        ListNode prev = null;
        for (ListNode curr = head; curr != null; ) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }

    ListNode findMid(ListNode head) {
        ListNode slow = head;
        for (ListNode fast = head; fast != null && fast.next != null; ) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
}
```

go

```go
func reorderList(head *ListNode) {
	if head == nil || head.Next == nil {
		return
	}
	slow := findMid(head)
	p1, p2 := head, reverse(slow)
	for p1 != slow {
		p1.Next, p1, p2 = p2, p2, p1.Next
	}
}
func findMid(head *ListNode) *ListNode {
	slow := head
	for fast := head; fast != nil && fast.Next != nil; {
		fast = fast.Next.Next
		slow = slow.Next
	}
	return slow
}
func reverse(head *ListNode) *ListNode {
	var prev *ListNode
	for curr := head; curr != nil; {
		curr, curr.Next, prev = curr.Next, prev, curr
	}
	return prev
}
```

python

一行交换的写法有顺序要求，比如必须先设置curr.next才能变动curr

```python
class Solution:
    def find_mid(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast, slow = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def reverse(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
            curr.next, curr, prev = prev, curr.next, curr
        return prev

    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return
        mid = self.find_mid(head)
        p1 = head
        p2 = self.reverse(mid)
        while p1 != mid:
            p1.next, p2, p1 = p2, p1.next, p2
```

#### 147. Insertion Sort List

go

```go
func insertionSortList(head *ListNode) *ListNode {
	dummy := &ListNode{}
	prev := dummy
	for head != nil {
		if prev.Val > head.Val {
			prev = dummy
		}
		for prev.Next != nil && prev.Next.Val < head.Val {
			prev = prev.Next
		}
		prev.Next, head, head.Next = head, head.Next, prev.Next
	}
	return dummy.Next
}
```

java

```java
class Solution {
    public ListNode insertionSortList(ListNode head) {
        ListNode dummy = new ListNode(), prev = dummy;
        while (head != null) {
            if (prev.val > head.val) {
                prev = dummy;
            }
            while (prev.next != null && prev.next.val < head.val) {
                prev = prev.next;
            }
            ListNode next = head.next;
            head.next = prev.next;
            prev.next = head;
            head = next;
        }
        return dummy.next;
    }
}
```

Python

```Python
class Solution:  
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:  
        dummy = ListNode()  
        prev = dummy  
        while head:  
            if prev.val > head.val:  
                prev = dummy  
            while prev.next and prev.next.val < head.val:  
                prev = prev.next  
            prev.next, head.next, head = head, prev.next, head.next  
        return dummy.next
```

#### 146. LRU Cache

java LinkedHashMap

```java
class LRUCache {
    int cap;
    LinkedHashMap<Integer, Integer> cache;

    public LRUCache(int capacity) {
        cap = capacity;
        cache = new LinkedHashMap<>(capacity, 0.75f, true);
    }

    public int get(int key) {
        return cache.getOrDefault(key, -1);
    }

    public void put(int key, int val) {
        cache.put(key, val);
        if (cache.size() > cap) {
            Iterator<Integer> iterator = cache.keySet().iterator();
            iterator.next();
            iterator.remove();
        }
    }
}
```

java 

```java
class Node {
    Node prev;
    Node next;
    Integer key;
    Integer val;

    public Node(Integer key, Integer val) {
        this.key = key;
        this.val = val;
    }
}

class LRUCache {
    int capacity;
    HashMap<Integer, Node> map;
    Node head;
    Node tail;

    public LRUCache(int capacity) {
        map = new HashMap<>();
        head = new Node(-1, -1);
        tail = new Node(-1, -1);
        head.next = tail;
        tail.prev = head;
        this.capacity = capacity;
    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) return -1;
        makeRecently(node);
        return node.val;
    }

    public void put(int key, int val) {
        if (map.containsKey(key)) {
            Node node = map.get(key);
            node.val = val;
            makeRecently(node);
            return;
        }
        Node node = new Node(key, val);
        map.put(key, node);
        toTail(node);
        if (map.size() > capacity) {
            remove(head.next);
        }
    }

    private void makeRecently(Node node) {
        unlink(node);
        toTail(node);
    }

    private void unlink(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void toTail(Node node) {
        node.prev = tail.prev;
        node.next = tail;
        tail.prev.next = node;
        tail.prev = node;
    }

    private void remove(Node node) {
        unlink(map.remove(node.key));
    }
}
```

Go

```go
type Node struct {
	prev *Node
	next *Node
	key  int
	val  int
}

type LRUCache struct {
	head     *Node
	tail     *Node
	capacity int
	Nodes    map[int]*Node
}

func Constructor(capacity int) LRUCache {
	head := &Node{}
	tail := &Node{}
	head.next = tail
	tail.prev = head
	return LRUCache{
		head:     head,
		tail:     tail,
		capacity: capacity,
		Nodes:    make(map[int]*Node),
	}
}

func (lru *LRUCache) Get(key int) int {
	if node, ok := lru.Nodes[key]; ok {
		lru.makeRecently(node)
		return node.val
	}
	return -1
}

func (lru *LRUCache) Put(key int, value int) {
	if node, ok := lru.Nodes[key]; ok {
		node.val = value
		lru.makeRecently(node)
		return
	}
	node := &Node{key: key, val: value}
	lru.Nodes[key] = node
	lru.toTail(node)
	if len(lru.Nodes) > lru.capacity {
		lru.remove(lru.head.next)
	}
}
func (lru *LRUCache) unlink(node *Node) {
	node.prev.next = node.next
	node.next.prev = node.prev
}
func (lru *LRUCache) toTail(node *Node) {
	node.prev = lru.tail.prev
	node.next = lru.tail
	lru.tail.prev.next = node
	lru.tail.prev = node
}
func (lru *LRUCache) makeRecently(node *Node) {
	lru.unlink(node)
	lru.toTail(node)
}
func (lru *LRUCache) remove(node *Node) {
	lru.unlink(node)
	delete(lru.Nodes, node.key)
}
```

#### 148. Sort List

```go
func findMid(head *ListNode) (prev, slow *ListNode) {
	slow = head
	for fast := head; fast != nil && fast.Next != nil; {
		fast = fast.Next.Next
		prev = slow
		slow = slow.Next
	}
	return prev, slow
}
func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	prev, slow := findMid(head)
	prev.Next = nil
	left := sortList(head)
	right := sortList(slow)
	return merge(left, right)
}
func merge(l1, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	pt := dummy
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
			pt.Next = l1
			l1 = l1.Next
		} else {
			pt.Next = l2
			l2 = l2.Next
		}
		pt = pt.Next
	}
	for l1 != nil {
		pt.Next = l1
		pt = pt.Next
		l1 = l1.Next
	}
	for l2 != nil {
		pt.Next = l2
		pt = pt.Next
		l2 = l2.Next
	}
	return dummy.Next
}
```

#### 155. Min Stack

go

```go
type stackNode struct {
	val      int
	minValue int
	next     *stackNode
}

type MinStack struct {
	head *stackNode
}

func Constructor() MinStack {
	return MinStack{}
}

func (stack *MinStack) Push(val int) {
	if stack.head == nil {
		stack.head = &stackNode{val, val, nil}
	} else {
		stack.head = &stackNode{val, min(stack.head.minValue, val), stack.head}
	}
}

func (stack *MinStack) Pop() {
	stack.head = stack.head.next
}

func (stack *MinStack) Top() int {
	return stack.head.val
}

func (stack *MinStack) GetMin() int {
	return stack.head.minValue
}
func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
```

java

```java
class MinStack {
    record Node(int val, int minValue, Node next) {
    }

    Node head;

    public void push(int val) {
        if (head == null) {
            head = new Node(val, val, null);
        } else {
            head = new Node(val, Math.min(val, head.minValue), head);
        }
    }

    public void pop() {
        head = head.next;
    }

    public int top() {
        return head.val;
    }

    public int getMin() {
        return head.minValue;
    }
}
```

#### 160. Intersection of Two Linked Lists

go

```go
func getIntersectionNode(headA, headB *ListNode) *ListNode {
   ptr1, ptr2 := headA, headB
   for ptr1 != ptr2 {
      if ptr1 == nil {
         ptr1 = headB
      } else {
         ptr1 = ptr1.Next
      }
      if ptr2 == nil {
         ptr2 = headA
      } else {
         ptr2 = ptr2.Next
      }
   }
   return ptr1
}
```

python

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        ptr1,ptr2 = headA,headB
        while ptr1 != ptr2:
            if not ptr1:
                ptr1 = headB
            else:
                ptr1 = ptr1.next
            if not ptr2:
                ptr2 = headA
            else:
                ptr2 = ptr2.next
        return ptr1
```

#### 203. Remove Linked List Elements

go

```go
func removeElements(head *ListNode, val int) *ListNode {
    dummy := &ListNode{Next: head}
    prev := dummy
    for curr:=head; curr != nil; curr = curr.Next {
        if curr.Val == val {
            prev.Next = curr.Next
        } else {
            prev = prev.Next
        }
    }
    return dummy.Next
}
```

python

```python
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        ptr = dummy = ListNode(next=head)
        while ptr and ptr.next:
            if ptr.next.val == val:
                ptr.next = ptr.next.next
            else:
                ptr = ptr.next
        return dummy.next
```

#### 206. Reverse Linked List

反转链表

迭代 go

```go
func reverseList(head *ListNode) *ListNode {
    var prev *ListNode
    for head != nil {
        prev, head, head.Next = head, head.Next, prev
    }
    return prev
}
```

迭代 python

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head
        while curr:
             curr.next, prev, curr = prev, curr, curr.next 
        return prev
```

递归 python

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        res = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return res
```

递归 go

```go
func reverseList(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
        return head
    }
    reversed := reverseList(head.Next)
    head.Next.Next = head
    head.Next = nil
    return reversed
}
```

#### 234. Palindrome Linked List

判断回文链表

反转后半部分然后比较

判断完需要恢复原样

O(1) space

```go
func isPalindrome(head *ListNode) bool {
	tail := halfEnd(head)
	rev := reverseList(tail.Next)
	defer func() { tail.Next = reverseList(rev) }()
	p1, p2 := head, rev
	for p1 != nil && p2 != nil && p1.Val == p2.Val {
		p1 = p1.Next
		p2 = p2.Next
	}
	res := p2 == nil
	return res
}
func reverseList(head *ListNode) *ListNode {
	var prev *ListNode
	for head != nil {
		prev, head, head.Next = head, head.Next, prev
	}
	return prev
}
func halfEnd(head *ListNode) *ListNode {
	slow := head
	for fast := head; fast.Next != nil && fast.Next.Next != nil; {
		fast = fast.Next.Next
		slow = slow.Next
	}
	return slow
}
```

#### 237. Delete Node in a Linked List

go

```go
func deleteNode(node *ListNode) {
    node.Val, node.Next = node.Next.Val, node.Next.Next
}
```

#### 287. Find the Duplicate Number

长度为n+1的数组，每个数都在range [1, n]范围中

（其中必有重复数字）

假设其中仅仅有一个数重复（重复两次甚至更多），如何找出这个数？

限制：不可修改原本数组，不可超过常数级别空间复杂度

空间的限制意味着不能用hashset保存

提示用线性时间复杂度解决

- 将数组看作链表，根据数组的值当做索引、寻址，可以找下一个节点

- 由于数组存在重复值，所以链表必定成环

- 找到环的入口即可

- 假设数组为 [1,3,4,2,2]

  [0|1] -> [1|3]-> [3|2]-> [2|4]-> [4|2]-> [2|4]

```go
func findDuplicate(nums []int) int {
	fast, slow := 0, 0
	for {
		fast = nums[nums[fast]]
		slow = nums[slow]
		if fast == slow {
			break
		}
	}
	fast = 0
	for slow != fast {
		slow = nums[slow]
		fast = nums[fast]
	}
	return slow
}
```

#### 328. Odd Even Linked List

java

```java
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode odd = head, even = head.next, evenHead = even;
        while (even != null && even.next != null) {
            odd.next = odd.next.next;
            even.next = even.next.next;
            even = even.next;
            odd = odd.next;
        }
        odd.next = evenHead;
        return head;
    }
}
```

#### 445. Add Two Numbers II

Go

```go
type stack struct {
	arr [100]int
	top int
}

func (s *stack) push(v int) {
	s.arr[s.top] = v
	s.top++
}
func (s *stack) pop() int {
	res := s.arr[s.top-1]
	s.top--
	return res
}
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	s1, s2 := stack{arr: [100]int{}}, stack{arr: [100]int{}}
	for ; l1 != nil; l1 = l1.Next {
		s1.push(l1.Val)
	}
	for ; l2 != nil; l2 = l2.Next {
		s2.push(l2.Val)
	}
	carry := 0
	dummy := &ListNode{}
	for curr := dummy; s1.top > 0 || s2.top > 0 || carry > 0; {
		if s1.top > 0 {
			carry += s1.pop()
		}
		if s2.top > 0 {
			carry += s2.pop()
		}
		curr.Next = &ListNode{Val: carry % 10, Next: curr.Next}
		carry /= 10
	}
	return dummy.Next
}
```

#### 707. Design Linked List

go

```go
type MyLinkedList struct {
	dummy *Node
	size  int
}
type Node struct {
	val  int
	next *Node
}

func Constructor() MyLinkedList {
	return MyLinkedList{&Node{val: -1}, 0}
}

func (list *MyLinkedList) Get(index int) int {
	if index > list.size-1 || index < 0 {
		return -1
	}
	curr := list.dummy.next
	for i := 0; i < index; i++ {
		curr = curr.next
	}
	return curr.val
}

func (list *MyLinkedList) AddAtHead(val int) {
	list.AddAtIndex(0, val)
}

func (list *MyLinkedList) AddAtTail(val int) {
	list.AddAtIndex(list.size, val)
}

func (list *MyLinkedList) AddAtIndex(index int, val int) {
	if index > list.size {
		return
	}
	if index < 0 {
		index = 0
	}
	list.size++
	prev := list.dummy
	for i := 0; i < index; i++ {
		prev = prev.next
	}
	prev.next = &Node{val, prev.next}
}

func (list *MyLinkedList) DeleteAtIndex(index int) {
	if index > list.size-1 || index < 0 {
		return
	}
	list.size--
	prev := list.dummy
	for i := 0; i < index; i++ {
		prev = prev.next
	}
	prev.next = prev.next.next
}
```

#### 876. Middle of the Linked List

go

```go
func middleNode(head *ListNode) *ListNode {
    fast, slow := head, head
    for fast != nil && fast.Next != nil {
        fast = fast.Next.Next
        slow = slow.Next
    }
    return slow
}
```

#### 1206. Design Skiplist

删除数据的时候要注意把索引也删了

Go

```go
type Node struct {
	val  int
	next *Node
	down *Node
}

type Skiplist struct {
	head *Node
}

func Constructor() Skiplist {
	return Skiplist{
		head: &Node{-1, nil, nil},
	}
}

func (sl *Skiplist) Search(target int) bool {
	for curr := sl.head; curr != nil; curr = curr.down {
		for curr.next != nil && curr.next.val < target {
			curr = curr.next
		}
		if curr.next != nil && curr.next.val == target {
			return true
		}
	}
	return false
}

func (sl *Skiplist) Add(num int) {
	var stack []*Node
	for curr := sl.head; curr != nil; curr = curr.down {
		for curr.next != nil && curr.next.val < num {
			curr = curr.next
		}
		stack = append(stack, curr)
	}
	insert := true
	var down *Node
	for insert && len(stack) > 0 {
		curr := stack[len(stack)-1]
		curr.next = &Node{val: num, next: curr.next, down: down}
		down = curr.next
		stack = stack[:len(stack)-1]
		insert = rand.Float64() < 0.5
	}
	if insert {
		sl.head = &Node{val: -1, next: nil, down: sl.head}
	}
}

func (sl *Skiplist) Erase(num int) bool {
	found := false
	for curr := sl.head; curr != nil; curr = curr.down {
		for curr.next != nil && curr.next.val < num {
			curr = curr.next
		}
		if curr.next != nil && curr.next.val == num {
			found = true
			curr.next = curr.next.next
		}
	}
	return found
}
```

java

```java
class Skiplist {
    static class Node {
        private final int val;
        private Node next;
        private final Node down;

        public Node(int val, Node next, Node down) {
            this.val = val;
            this.next = next;
            this.down = down;
        }
    }

    private Node head;
    private final Random random;

    public Skiplist() {
        this.head = new Node(-1, null, null);
        this.random = new Random();
    }

    public boolean search(int target) {
        for (Node curr = head; curr != null; curr = curr.down) {
            while (curr.next != null && curr.next.val < target) {
                curr = curr.next;
            }
            if (curr.next != null && curr.next.val == target) {
                return true;
            }
        }
        return false;
    }

    public void add(int num) {
        ArrayDeque<Node> stack = new ArrayDeque<>();
        for (Node curr = head; curr != null; curr = curr.down) {
            while (curr.next != null && curr.next.val < num) {
                curr = curr.next;
            }
            stack.push(curr);
        }
        boolean insert = true;
        Node down = null;
        while (insert && !stack.isEmpty()) {
            Node curr = stack.pop();
            curr.next = new Node(num, curr.next, down);
            down = curr.next;
            insert = random.nextDouble() < 0.5;
        }
        // Finally, update 'head' according to random value
        if (insert) {
            // use the original head as 'down'
            head = new Node(-1, null, head);
        }
    }

    public boolean erase(int num) {
        boolean found = false;
        for (Node curr = head; curr != null; curr = curr.down) {
            while (curr.next != null && curr.next.val < num) {
                curr = curr.next;
            }
            if (curr.next != null && curr.next.val == num) {
                found = true;
                curr.next = curr.next.next;
            }
        }
        return found;
    }
}
```

#### 2130. Maximum Twin Sum of a Linked List

java

```java
class Solution {
    public int pairSum(ListNode head) {
        ListNode mid = findMid(head);
        ListNode reverse = reverse(mid);
        int res = 0;
        ListNode fir = head;
        for (ListNode sec = reverse; sec != null; ) {
            res = Math.max(res, fir.val + sec.val);
            fir = fir.next;
            sec = sec.next;
        }
        return res;
    }

    ListNode findMid(ListNode head) {
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    ListNode reverse(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode tmp = head.next;
            head.next = prev;
            prev = head;
            head = tmp;
        }
        return prev;
    }
}
```

