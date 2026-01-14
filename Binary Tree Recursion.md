#### 94. Binary Tree Inorder Traversal

python

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        self.helper(root, res)
        return res

    def helper(self, root: Optional[TreeNode], res: list) -> None:
        if not root:
            return
        self.helper(root.left, res)
        res += root.val,
        self.helper(root.right, res)
```

go

```go
func inorderTraversal(root *TreeNode) []int {
	var res []int
	dfs(root, &res)
	return res
}
func dfs(root *TreeNode, res *[]int) {
	if root == nil {
		return
	}
	dfs(root.Left, res)
	*res = append(*res, root.Val)
	dfs(root.Right, res)
}
```

no recursion

Go

```go
func inorderTraversal(root *TreeNode) []int {
	var res []int
	var stack []*TreeNode
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, root.Val)
		root = root.Right
	}
	return res
}
```

#### 144. Binary Tree Preorder Traversal

go

```go
func preorderTraversal(root *TreeNode) []int {
   res := make([]int, 0)
   pre(root, &res)
   return res
}
func pre(root *TreeNode, res *[]int) {
   if root == nil {
      return
   }
   *res = append(*res, root.Val)
   pre(root.Left, res)
   pre(root.Right, res)
}
```

no recursion

Go

```go
func preorderTraversal(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}

	for stack := []*TreeNode{root}; len(stack) > 0; {
		curr := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		res = append(res, curr.Val)
		if curr.Right != nil {
			stack = append(stack, curr.Right)
		}
		if curr.Left != nil {
			stack = append(stack, curr.Left)
		}
	}
	return res
}
```

#### 145. Binary Tree Postorder Traversal

go

```go
func postorderTraversal(root *TreeNode) []int {
   res := make([]int, 0)
   post(root, &res)
   return res
}
func post(root *TreeNode, res *[]int) {
   if root == nil {
      return
   }
   post(root.Left, res)
   post(root.Right, res)
   *res = append(*res, root.Val)
}
```

#### 173. Binary Search Tree Iterator

手动操作栈实现中序

注意栈里面装的是节点而不是数值

无脑将左子树压栈，知道curr指针为空

然后弹栈，让curr指针挪到栈顶元素的右子树

为什么这是对的呢？ 因为整棵树可以看做根节点+左子树

弹栈顺序是先左再头，递归序也对

Go

```go
type BSTIterator struct {
	curr  *TreeNode
	stack []*TreeNode
}

func Constructor(root *TreeNode) BSTIterator {
	return BSTIterator{curr: root}
}

func (it *BSTIterator) Next() int {
	for it.curr != nil {
		it.stack = append(it.stack, it.curr)
		it.curr = it.curr.Left
	}
	res := it.stack[len(it.stack)-1]
	it.stack = it.stack[:len(it.stack)-1]
	it.curr = res.Right
	return res.Val
}

func (it *BSTIterator) HasNext() bool {
	return it.curr != nil || len(it.stack) > 0
}
```

java

```java
class BSTIterator {
    ArrayDeque<TreeNode> stack;
    TreeNode curr;

    public BSTIterator(TreeNode root) {
        stack = new ArrayDeque<>();
        curr = root;
    }

    public int next() {
        while (curr != null) {
            stack.push(curr);
            curr = curr.left;
        }
        TreeNode res = stack.pop();
        curr = res.right;
        return res.val;
    }

    public boolean hasNext() {
        return curr != null || !stack.isEmpty();
    }
}
```

python

```python
class BSTIterator:    
    def __init__(self, root: Optional[TreeNode]):  
        self.stack = []  
        self.curr = root  
  
    def next(self) -> int:  
        while self.curr:  
            self.stack.append(self.curr)  
            self.curr = self.curr.left  
        top = self.stack.pop()  
        self.curr = top.right  
        return top.val  
  
    def hasNext(self) -> bool:  
        if self.curr or self.stack:  
            return True  
        return False
```

二叉树非常适合用分治和回溯

二叉树有n个节点，那么子树的数量是n，子树从该节点出发一直探到底

本文收录二叉树题目但不包括BFS解法，纯纯深搜

二叉树中特定的某个节点X，在二叉树中做前序遍历得到的在X之前的节点集合，与后续遍历得到的在X之后的节点集合，求交集，发现交集都是X的祖先节点，为什么？

由于先序会先访问头结点，所以祖先一定出现在X左边，后序会最后访问头结点，所以X的位置一定在其祖先的左边

先序遍历中X的子节点都会出现在X的右边

#### [687. Longest Univalue Path](https://leetcode.com/problems/longest-univalue-path/)

java

```java
class Solution {
    int res;

    public int longestUnivaluePath(TreeNode root) {
        res = 0;
        dfs(root, -1);
        return res;
    }

    private int dfs(TreeNode root, int val) {
        if (root == null) {
            return 0;
        }
        int left = dfs(root.left, root.val);
        int right = dfs(root.right, root.val);
        res = Math.max(res, left + right);
        if (root.val == val) {
            return Math.max(left, right) + 1;
        }
        return 0;
    }
}
```

#### [652. Find Duplicate Subtrees](https://leetcode.com/problems/find-duplicate-subtrees/)

```java
class Solution {  
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {  
        HashMap<String, List<TreeNode>> map = new HashMap<>();  
        dfs(root, map);  
        return map.values().stream().filter(l -> l.size() > 1).map(l -> l.get(0)).collect(Collectors.toList());  
    }  
  
    private String dfs(TreeNode root, HashMap<String, List<TreeNode>> map) {  
        if (root == null) {  
            return "null";  
        }  
  
        String text = root.val +  
                "," +  
                dfs(root.left, map) +  
                "," +  
                dfs(root.right, map);  
        map.putIfAbsent(text, new ArrayList<>());  
        map.get(text).add(root);  
        return text;  
    }  
}
```

#### [1080. Insufficient Nodes in Root to Leaf Paths](https://leetcode.com/problems/insufficient-nodes-in-root-to-leaf-paths/)

```java
class Solution {
    public TreeNode sufficientSubset(TreeNode root, int limit) {
        if (root == null) {
            return null;
        }
        if (isLeaf(root)) {
            return root.val < limit ? null : root;
        }
        root.left = sufficientSubset(root.left, limit - root.val);
        root.right = sufficientSubset(root.right, limit - root.val);
        return isLeaf(root) ? null : root;
    }

    private boolean isLeaf(TreeNode root) {
        return root.left == null && root.right == null;
    }
}
```

#### [951. Flip Equivalent Binary Trees](https://leetcode.com/problems/flip-equivalent-binary-trees/)

```java
class Solution {  
    public boolean flipEquiv(TreeNode root1, TreeNode root2) {  
        if (root1 == null || root2 == null) {  
            return root1 == root2;  
        }  
        if (root1.val != root2.val) {  
            return false;  
        }  
        // A binary tree X is flip equivalent to a binary tree Y  
        // if and only if we can make X equal to Y after some number of flip operations.        // 'number' could be 0        return flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left)  
                || flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right);  
    }  
}
```

#### [1367. Linked List in Binary Tree](https://leetcode.com/problems/linked-list-in-binary-tree/)

```java
class Solution {  
    public boolean isSubPath(ListNode head, TreeNode root) {  
        if (head == null) return true;  
        if (root == null) return false;  
        return dfs(head, root) || isSubPath(head, root.left) || isSubPath(head, root.right);  
    }  
  
    // compare the value of head and root  
    private boolean dfs(ListNode head, TreeNode root) {  
        if (head == null) return true;  
        if (root == null) return false;  
        if (head.val != root.val) {  
            return false;  
        }  
        return dfs(head.next, root.left) || dfs(head.next, root.right);  
    }  
}
```

#### [1110. Delete Nodes And Return Forest](https://leetcode.com/problems/delete-nodes-and-return-forest/)



```java
class Solution {
    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        HashSet<Integer> set = new HashSet<>();
        Arrays.stream(to_delete).forEach(set::add);
        ArrayList<TreeNode> res = new ArrayList<>();
        dfs(root, set, res, true);
        return res;
    }

    private TreeNode dfs(TreeNode root, HashSet<Integer> set, ArrayList<TreeNode> res, boolean isRoot) {
        if (root == null) {
            return null;
        }
        boolean isDeleted = set.contains(root.val);
        if (isRoot && !isDeleted) {
            res.add(root);
        }
        root.left = dfs(root.left, set, res, isDeleted);
        root.right = dfs(root.right, set, res, isDeleted);
        return isDeleted ? null : root;
    }
}
```

#### [1161. Maximum Level Sum of a Binary Tree](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/)

bfs

```java
class Solution {  
    public int maxLevelSum(TreeNode root) {  
        int max = Integer.MIN_VALUE, maxLevel = 1;  
        ArrayDeque<TreeNode> queue = new ArrayDeque<>();  
        queue.offer(root);  
        for (int level = 1; !queue.isEmpty(); level++) {  
            int sum = 0;  
            for (int i = queue.size() - 1; i >= 0; i--) {  
                TreeNode poll = queue.poll();  
                assert poll != null;  
                sum += poll.val;  
                if (poll.left != null) {  
                    queue.offer(poll.left);  
                }  
                if (poll.right != null) {  
                    queue.offer(poll.right);  
                }  
            }  
            if (max < sum) {  
                max =sum;  
                maxLevel = level;  
            }  
        }  
       return maxLevel;  
    }  
}
```

dfs

```java
class Solution {
    public int maxLevelSum(TreeNode root) {
        ArrayList<Integer> list = new ArrayList<>();
        dfs(root, list, 0);
        int maxIdx = getMaxIdx(list);
        System.out.println(list);
        return maxIdx + 1;
    }

    private int getMaxIdx(ArrayList<Integer> list) {
        int maxIdx = 0, max = Integer.MIN_VALUE;
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) > max) {
                max = list.get(i);
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private void dfs(TreeNode root, ArrayList<Integer> list, int i) {
        if (root == null) {
            return;
        }
        if (list.size() == i) {
            list.add(root.val);
        } else {
            list.set(i, list.get(i) + root.val);
        }
        dfs(root.left, list, i + 1);
        dfs(root.right, list, i + 1);
    }
}
```

#### [1325. Delete Leaves With a Given Value](https://leetcode.com/problems/delete-leaves-with-a-given-value/)

```java
class Solution {
    public TreeNode removeLeafNodes(TreeNode root, int target) {
        if (root == null) {
            return null;
        }
        root.left = removeLeafNodes(root.left, target);
        root.right = removeLeafNodes(root.right, target);
        if (root.left == null && root.right == null && root.val == target) {
            return null;
        }
        return root;
    }
}
```

#### 606. Construct String from Binary Tree

java

```java
class Solution {
    public String tree2str(TreeNode root) {
        StringBuilder path = new StringBuilder();
        dfs(root, path);
        return path.toString();
    }

    void dfs(TreeNode root, StringBuilder path) {
        if (root == null) return;
        path.append(root.val);
        if (root.left == null && root.right == null) return;
        path.append("(");
        dfs(root.left, path);
        path.append(")");
        if (root.right != null) {
            path.append("(");
            dfs(root.right, path);
            path.append(")");
        }
    }
}
```


#### 95. Unique Binary Search Trees II

go

https://leetcode.com/problems/unique-binary-search-trees-ii/discuss/31508/Divide-and-conquer.-F(i)-G(i-1)-

```go
func generateTrees(n int) []*TreeNode {
   return dfs(1, n)
}
func dfs(start, end int) []*TreeNode {
   var res []*TreeNode
   if start > end {
      res = append(res, nil)
      return res
   }
   for i := start; i < end+1; i++ {
      leftSub := dfs(start, i-1)
      rightSub := dfs(i+1, end)
      for _, left := range leftSub {
         for _, right := range rightSub {
            root := &TreeNode{Val: i}
            root.Left = left
            root.Right = right
            res = append(res, root)
         }
      }
   }
   return res
}
```

#### 96. Unique Binary Search Trees

go

```go
func numTrees(n int) int {
   dp := make([]int, n+1)
   dp[0], dp[1] = 1, 1
   for i := 2; i < n+1; i++ {
      for j := 1; j < i+1; j++ {
         dp[i] += dp[j-1] * dp[i-j]
      }
   }
   return dp[n]
}
```

#### 98. Validate Binary Search Tree

go

```go
func isValidBST(root *TreeNode) bool {  
   return dfs(root, nil, nil)  
}  
func dfs(root, min, max *TreeNode) bool {  
   if root == nil {  
      return true  
   }  
   if min != nil && min.Val >= root.Val {  
      return false  
   }  
   if max != nil && max.Val <= root.Val {  
      return false  
   }  
   return dfs(root.Left, min, root) && dfs(root.Right, root, max)  
}
```

python 

https://leetcode.com/problems/validate-binary-search-tree/discuss/1904588/Simple-Python-Recursive-Solution-oror-Faster-than-90.23

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        return self.validate(root, None, None)
    def validate(self,root: [TreeNode], min_node: [TreeNode], max_node: [TreeNode]):
        if not root:
            return True
        if min_node and min_node.val >= root.val:
            return False
        if max_node and max_node.val <= root.val:
            return False
        return self.validate(root.left, min_node, root) and self.validate(root.right,root, max_node)
```

#### 99. Recover Binary Search Tree

Go

```go
func recoverTree(root *TreeNode) {
	var (
		first *TreeNode
		last  *TreeNode
	)
	prev := &TreeNode{Val: math.MinInt}

	var dfs func(root *TreeNode)
	dfs = func(root *TreeNode) {
		if root == nil {
			return
		}
		dfs(root.Left)
		if first == nil && prev.Val > root.Val {
			first = prev
		}
		if first != nil && prev.Val > root.Val {
			last = root
		}
		prev = root
		dfs(root.Right)
	}

	dfs(root)
	first.Val, last.Val = last.Val, first.Val
}
```

java

```java
class Solution {
    TreeNode first = null;
    TreeNode last = null;
    TreeNode prev = new TreeNode(Integer.MIN_VALUE);

    public void recoverTree(TreeNode root) {
        dfs(root);
        first.val = first.val ^ last.val;
        last.val = first.val ^ last.val;
        first.val = first.val ^ last.val;
    }

    void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        dfs(root.left);
        if (first == null && prev.val > root.val) {
            first = prev;
        }
        if (first != null && prev.val > root.val) {
            last = root;
        }
        prev = root;
        dfs(root.right);
    }
}
```

#### 108. Convert Sorted Array to Binary Search Tree

https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/discuss/35220/My-Accepted-Java-Solution

```go
func sortedArrayToBST(nums []int) *TreeNode {  
   return dfs(0, len(nums), nums)  
}  
func dfs(low, high int, nums []int) *TreeNode {  
   for low < high {  
      mid := (low + high) / 2  
      root := &TreeNode{nums[mid], nil, nil}  
      root.Left = dfs(low, mid, nums)  
      root.Right = dfs(mid+1, high, nums)  
      return root  
   }  
   return nil  
}
```

#### 100. Same Tree

```go
func isSameTree(p *TreeNode, q *TreeNode) bool {
   if p != nil && q != nil {
      return p.Val == q.Val && isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
   }
   return p == q
}
```

#### 101. Symmetric Tree

```go
func isSymmetric(root *TreeNode) bool {
    return dfs(root.Left, root.Right)
}
func dfs(left, right *TreeNode) bool {
    if left != nil && right != nil {
        return left.Val == right.Val && dfs(left.Left, right.Right) && dfs(left.Right, right.Left)
    }
    return left == right
}
```

#### 104. Maximum Depth of Binary Tree

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left),self.maxDepth(root.right))
```

#### 105. Construct Binary Tree from Preorder and Inorder Traversal

Go

```go
func buildTree(preorder []int, inorder []int) *TreeNode {  
   mapInorder := make(map[int]int)  
   for i, v := range inorder {  
      mapInorder[v] = i  
   }  
   n := len(preorder)  
   pos := 0  
   return dfs(0, n, preorder, mapInorder, &pos)  
}  
func dfs(low int, high int, preorder []int, mapInorder map[int]int, pos *int) *TreeNode {  
   if low >= high {  
      return nil  
   }  
   root := &TreeNode{Val: preorder[*pos]}  
   *pos++  
   mid := mapInorder[root.Val]  
   root.Left = dfs(low, mid, preorder, mapInorder, pos)  
   root.Right = dfs(mid+1, high, preorder, mapInorder, pos)  
   return root  
}
```

java

```java
class Solution {  
    int pos;  
  
    public TreeNode buildTree(int[] preorder, int[] inorder) {  
        HashMap<Integer, Integer> map = new HashMap<>();  
        int n = inorder.length;  
        pos = 0;  
        for (int i = 0; i < n; i++) {  
            map.put(inorder[i], i);  
        }  
        return dfs(preorder, map, 0, n);  
    }  
  
    TreeNode dfs(int[] preorder, HashMap<Integer, Integer> map, int low, int high) {  
        if (low >= high) return null;  
        int x = preorder[pos];  
        pos++;  
        TreeNode root = new TreeNode(x);  
        Integer inIdx = map.get(x);  
        root.left = dfs(preorder, map, low, inIdx);  
        root.right = dfs(preorder, map, inIdx + 1, high);  
        return root;  
    }  
}
```

#### 106. Construct Binary Tree from Inorder and Postorder Traversal

用类似中序遍历的方式构建二叉树，每层递归中只处理一个节点

按照中-->右-->左的顺序

Go

```go
func buildTree(inorder, postorder []int) *TreeNode {
	mapInorder := make(map[int]int)
	for i, v := range inorder {
		mapInorder[v] = i
	}
	n := len(postorder)
	pos := n - 1
	return dfs(0, n, postorder, mapInorder, &pos)
}
func dfs(low int, high int, postorder []int, mapInorder map[int]int, pos *int) *TreeNode {
	if low >= high {
		return nil
	}
	x := &TreeNode{Val: postorder[*pos]}
	*pos--
	mid := mapInorder[x.Val]
	x.Right = dfs(mid+1, high, postorder, mapInorder, pos)
	x.Left = dfs(low, mid, postorder, mapInorder, pos)
	return x
}
```

java

```java
class Solution {  
    int pos;  
  
    public TreeNode buildTree(int[] inorder, int[] postorder) {  
        HashMap<Integer, Integer> map = new HashMap<>();  
        int n = postorder.length;  
        pos = n - 1;  
        for (int i = 0; i < n; i++) {  
            map.put(inorder[i], i);  
        }  
        return dfs(postorder, map, 0, n);  
    }  
  
    TreeNode dfs(int[] postorder, HashMap<Integer, Integer> map, int low, int high) {  
        if (low >= high) return null;  
        int x = postorder[pos];  
        pos--;  
        TreeNode root = new TreeNode(x);  
        Integer inIdx = map.get(x);  
        root.right = dfs(postorder, map, inIdx + 1, high);  
        root.left = dfs(postorder, map, low, inIdx);  
        return root;  
    }  
}
```

#### 109. Convert Sorted List to Binary Search Tree

https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/solutions/35476/Share-my-JAVA-solution-1ms-very-short-and-concise./

Go

```go
func sortedListToBST(head *ListNode) *TreeNode {
   if head == nil {
      return nil
   }
   return dfs(head, nil)
}
func dfs(head, tail *ListNode) *TreeNode {
   if head == tail {
      return nil
   }
   fast, slow := head, head
   for fast != tail && fast.Next != tail {
      fast = fast.Next.Next
      slow = slow.Next
   }
   root := &TreeNode{Val: slow.Val}
   root.Left = dfs(head, slow)
   root.Right = dfs(slow.Next, tail)
   return root
}
```

java

```java
class Solution {
    public TreeNode sortedListToBST(ListNode head) {
        return dfs(head, null);
    }

    TreeNode dfs(ListNode head, ListNode tail) {
        if (head == tail) {
            return null;
        }
        ListNode mid = findMid(head, tail);
        TreeNode res = new TreeNode(mid.val);
        res.left = dfs(head, mid);
        res.right = dfs(mid.next, tail);
        return res;
    }

    ListNode findMid(ListNode head, ListNode tail) {
        ListNode fast = head, slow = head;
        while (fast != tail && fast.next != tail) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
}
```

#### 110. Balanced Binary Tree

java

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return dfs(root) > -1;
    }

    int dfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = dfs(root.left);
        int right = dfs(root.right);
        if (Math.abs(left - right) > 1 || left < 0 || right < 0) {
            return -1;
        }
        return Math.max(left, right) + 1;
    }
}
```

Go

```go
func isBalanced(root *TreeNode) bool {
	return dfs(root) > -1
}
func dfs(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left := dfs(root.Left)
	right := dfs(root.Right)
	if left < 0 || right < 0 || abs(right-left) > 1 {
		return -1
	}
	return max(left, right) + 1
}
func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

#### 111. Minimum Depth of Binary Tree

```python
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root: return 0
        if not root.left:
            return self.minDepth(root.right)+1
        if not root.right:
            return self.minDepth(root.left)+1
        return min(self.minDepth(root.left), self.minDepth(root.right))+1
```

#### 112. Path Sum

```go
func hasPathSum(root *TreeNode, targetSum int) bool {
    if root == nil {
        return false
    }
    if root.Left == nil && root.Right == nil && targetSum == root.Val {
        return true
    }
    return hasPathSum(root.Left, targetSum-root.Val) || hasPathSum(root.Right, targetSum-root.Val)
}
```

#### 113. Path Sum II

go

```go
func pathSum(root *TreeNode, targetSum int) [][]int {
   var res [][]int
   dfs(root, targetSum, &res, []int{})
   return res
}
func dfs(root *TreeNode, targetSum int, res *[][]int, path []int) {
   if root == nil {
      return
   }
   if root.Left == nil && root.Right == nil && targetSum == root.Val {
      path = append(path, root.Val)
      *res = append(*res, append([]int{}, path...))
      return
   }
   dfs(root.Left, targetSum-root.Val, res, append(path, root.Val))
   dfs(root.Right, targetSum-root.Val, res, append(path, root.Val))
}
```

java

```java
public class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<>();
        dfs(root, sum, new ArrayList<>(), res);
        return res;
    }

    void dfs(TreeNode root, int sum, List<Integer> path, List<List<Integer>> res) {
        if (root == null) return;
        path.add(root.val);
        if (root.left == null && root.right == null && sum == root.val)
            res.add(new ArrayList<>(path)); // Don' t return here
        dfs(root.left, sum - root.val, path, res);
        dfs(root.right, sum - root.val, path, res);
        path.remove(path.size() - 1);
    }
}
```

#### 114. Flatten Binary Tree to Linked List

```go
func flatten(root *TreeNode) {
	dfs(root, nil)
}
func dfs(root, prev *TreeNode) *TreeNode {
	if root == nil {
		return prev
	}
	prev = dfs(root.Right, prev)
	prev = dfs(root.Left, prev)
	root.Right = prev
	root.Left = nil
	return root
}
```

#### 116. Populating Next Right Pointers in Each Node

```go
func connect(root *Node) *Node {
    if root == nil {
        return nil
    }
    var curr *Node
    prev := root
    for prev.Left != nil {
        curr = prev
        for curr != nil {
            curr.Left.Next = curr.Right
            if curr.Next != nil {
                curr.Right.Next = curr.Next.Left
            }
            curr = curr.Next
        }
        prev = prev.Left
    }
    return root
}
```

#### 124. Binary Tree Maximum Path Sum

数据范围里有负数，而且题目说不一定需要经过根节点

然而既然是path就一定会经过节点，依然可以对每个节点计算和为最大的path

DFS返回单路最大路径和

go

```go
func maxPathSum(root *TreeNode) int {  
   res := math.MinInt  
   dfs(root, &res)  
   return res  
}  
func dfs(root *TreeNode, res *int) int {  
   if root == nil {  
      return 0  
   }  
   left := max(0, dfs(root.Left, res))  
   right := max(0, dfs(root.Right, res))  
   *res = max(*res, left+right+root.Val)  
   return max(left, right) + root.Val  
}  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}  
```

java

```java
class Solution {  
    int res;  
  
    public int maxPathSum(TreeNode root) {  
        res = Integer.MIN_VALUE;  
        dfs(root);  
        return res;  
    }  
  
    int dfs(TreeNode root) {  
        if (root == null) return 0;  
        int left = Math.max(0, dfs(root.left));  
        int right = Math.max(0, dfs(root.right));  
        res = Math.max(res, left + right + root.val);  
        return Math.max(left, right) + root.val;  
    }  
}
```

#### 129. Sum Root to Leaf Numbers

java

```java
class Solution {
    public int sumNumbers(TreeNode root) {
        return dfs(root, 0);
    }

    int dfs(TreeNode root, int res) {
        if (root == null) return 0;
        res = 10 * res + root.val;
        if (root.left == null && root.right == null) return res;
        return dfs(root.left, res) + dfs(root.right, res);
    }
}
```


#### 199. Binary Tree Right Side View

Go

```go
func rightSideView(root *TreeNode) []int {
   res := make([]int, 0)
   dfs(root, 0, &res)
   return res
}
func dfs(root *TreeNode, level int, res *[]int) {
   if root == nil {
      return
   }
   if len(*res) == level {
      *res = append(*res, root.Val)
   }
   dfs(root.Right, level+1, res)
   dfs(root.Left, level+1, res)
}
```

#### 222. Count Complete Tree Nodes

go

```go
func countNodes(root *TreeNode) int {
    if root == nil {
        return 0
    }
    return countNodes(root.Left) + countNodes(root.Right) + 1
}
```

#### 226. Invert Binary Tree

二叉树镜像翻转

java

```java
class Solution {  
    public TreeNode invertTree(TreeNode root) {  
        if (root != null) {  
            TreeNode tmp = root.right;  
            root.right = invertTree(root.left);  
            root.left = invertTree(tmp);  
        }  
        return root;  
    }  
}
```

python

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root:
            root.left, root.right = self.invertTree(root.right),self.invertTree(root.left)
        return root
```

go

```go
func invertTree(root *TreeNode) *TreeNode {
    if root != nil {
        root.Left, root.Right = invertTree(root.Right), invertTree(root.Left)
    }
    return root
}
```

#### 230. Kth Smallest Element in a BST

Go

```go
func kthSmallest(root *TreeNode, k int) int {
	res, count := 0, 0
	dfs(root, &res, &count, k)
	return res
}
func dfs(root *TreeNode, res *int, count *int, k int) {
	if root == nil {
		return
	}
	dfs(root.Left, res, count, k)
	*count++
	if *count == k {
		*res = root.Val
		return
	}
	dfs(root.Right, res, count, k)
}
```

#### 235. Lowest Common Ancestor of a Binary Search Tree

go

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if p.Val < root.Val && q.Val < root.Val {
        return lowestCommonAncestor(root.Left,p,q)
    }
    if p.Val > root.Val && q.Val > root.Val {
        return lowestCommonAncestor(root.Right,p,q)
    }
    return root
}
```

#### 236. Lowest Common Ancestor of a Binary Tree

寻找最低的公共父节点

go

```go
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if root == nil || root == p || root == q {
        return root
    }
    lc := lowestCommonAncestor(root.Left, p, q)
    rc := lowestCommonAncestor(root.Right, p, q)
    if lc == nil {
        return rc
    }
    if rc == nil {
        return lc
    }
    return root
}
```

python

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root in (None, p, q):
            return root
        lc = self.lowestCommonAncestor(root.left, p, q)
        rc = self.lowestCommonAncestor(root.right, p, q)
        if not lc:
            return rc
        if not rc:
            return lc
        return root
```

#### 257. Binary Tree Paths

```go
func binaryTreePaths(root *TreeNode) []string {
	var res []string
	dfs(root, &res, []string{})
	return res
}
func dfs(root *TreeNode, res *[]string, path []string) {
	if root == nil {
		return
	}
	path = append(path, strconv.Itoa(root.Val))
	if root.Left == nil && root.Right == nil {
		*res = append(*res, strings.Join(path, "->"))
	}
	dfs(root.Left, res, path)
	dfs(root.Right, res, path)
}

```

java

```java
class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        ArrayList<String> res = new ArrayList<>();
        dfs(root, res, new ArrayList<>());
        return res;
    }

    void dfs(TreeNode root, ArrayList<String> res, ArrayList<String> path) {
        if (root == null) return;
        path.add(String.valueOf(root.val));
        if (root.left == null && root.right == null) res.add(String.join("->", path));
        dfs(root.left, res, path);
        dfs(root.right, res, path);
        path.remove(path.size() - 1);
    }
}
```

python

```python
class Solution:  
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:  
        res = []  
        self.dfs(root, [], res)  
        return res  
  
    def dfs(self, root: Optional[TreeNode], path: List[str], res: List[str]):  
        if not root:  
            return  
        v = str(root.val)  
        path.append(v)  
        if not root.left and not root.right:  
            res.append("->".join(path))  
        self.dfs(root.left, path, res)  
        self.dfs(root.right, path, res)  
        path.pop()
```

#### 297. Serialize and Deserialize Binary Tree

中序方式会有歧义，以下展示先序方式

Go

```go
type Codec struct{}

func Constructor() (_ Codec) {
	return
}

// Serializes a tree to a single string.
func (*Codec) serialize(root *TreeNode) string {
	var res []string
	var dfs func(root *TreeNode)
	dfs = func(root *TreeNode) {
		if root == nil {
			res = append(res, "#")
			return
		}
		res = append(res, strconv.Itoa(root.Val))
		dfs(root.Left)
		dfs(root.Right)
	}
	dfs(root)
	return strings.Join(res, ",")
}

// Deserializes your encoded data to tree.
func (*Codec) deserialize(data string) *TreeNode {
	arr := strings.Split(data, ",")
	var dfs func() *TreeNode
	dfs = func() *TreeNode {
		token := arr[0]
		arr = arr[1:]
		if token == "#" {
			return nil
		}
		v, _ := strconv.Atoi(token)
		return &TreeNode{v, dfs(), dfs()}
	}
	return dfs()
}
```

java

```java
class Codec {
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        List<String> res = new ArrayList<>();
        buildString(root, res);
        return String.join(",", res);
    }

    private void buildString(TreeNode node, List<String> res) {
        if (node == null) {
            res.add("#");
            return;
        }
        res.add(String.valueOf(node.val));
        buildString(node.left, res);
        buildString(node.right, res);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        ArrayDeque<String> nodes = new ArrayDeque<>(List.of(data.split(",")));
        return buildTree(nodes);
    }

    private TreeNode buildTree(ArrayDeque<String> nodes) {
        String val = nodes.poll();
        if (val.equals("#")) return null;
        return new TreeNode(Integer.parseInt(val), buildTree(nodes), buildTree(nodes));
    }
}
```

#### 331. Verify Preorder Serialization of a Binary Tree

go

https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/discuss/78551/7-lines-Easy-Java-Solution

```go
func isValidSerialization(preorder string) bool {  
   nodes := strings.Split(preorder, ",")  
   diff := 1  
   for _, node := range nodes {  
      diff--  
      if diff < 0 {  
         return false  
      }  
      if node != "#" {  
         diff += 2  
      }  
   }  
   return diff == 0  
}
```

go

https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/discuss/78722/Straight-forward-C%2B%2B-solution-with-explanation

```go
func isValidSerialization(s string) bool {  
   nodes := strings.Split(s, ",")  
   nodeCnt, nilCnt := 0, 0  
   for i, node := range nodes {  
      if node == "#" {  
         nilCnt++  
      } else {  
         nodeCnt++  
      }  
      if nilCnt >= nodeCnt+1 && i < len(nodes)-1 {  
         return false  
      }  
   }  
   return nilCnt == nodeCnt+1  
}
```


#### 404. Sum of Left Leaves

Go

```go
func sumOfLeftLeaves(root *TreeNode) int {
   if root == nil {
      return 0
   }
   left := sumOfLeftLeaves(root.Left)
   right := sumOfLeftLeaves(root.Right)
   res := 0
   if root.Left != nil && root.Left.Left == nil && root.Left.Right == nil {
      res = root.Left.Val
   }
   return res + left + right
}
```
#### 430. Flatten a Multilevel Doubly Linked List

java

```java
class Solution {
    public Node flatten(Node head) {
        return flatten(head, null);
    }

    public Node flatten(Node head, Node rest) {
        if (head == null) {
            return rest;
        }
        head.next = flatten(head.child, flatten(head.next, rest));
        if (head.next != null) {
            head.next.prev = head;
        }
        head.child = null;
        return head;
    }
}
```
#### 437. Path Sum III

go

```go
func pathSum(root *TreeNode, target int) int {  
   hashMap := map[int]int{0: 1}  
   return dfs(root, target, 0, hashMap)  
}  
func dfs(root *TreeNode, target, sum int, hashMap map[int]int) int {  
   if root == nil {  
      return 0  
   }  
   sum += root.Val  
   count := 0  
   if _, ok := hashMap[sum-target]; ok {  
      count = hashMap[sum-target]  
   }  
   hashMap[sum]++  
   count += dfs(root.Left, target, sum, hashMap)  
   count += dfs(root.Right, target, sum, hashMap)  
   hashMap[sum]--  
   return count  
}
```

java

```java
class Solution {  
    public int pathSum(TreeNode root, int target) {  
        HashMap<Long, Integer> hashMap = new HashMap<>();  
        hashMap.put(0L, 1);  
        return dfs(root, target, 0, hashMap);  
    }  
  
    private int dfs(TreeNode root, int target, long sum, HashMap<Long, Integer> hashMap) {  
        if (root == null) {  
            return 0;  
        }  
        sum += root.val;  
        int count = hashMap.getOrDefault(sum - target, 0);  
        hashMap.put(sum, hashMap.getOrDefault(sum, 0) + 1);  
        count += dfs(root.left, target, sum, hashMap);  
        count += dfs(root.right, target, sum, hashMap);  
        hashMap.put(sum, hashMap.get(sum) - 1);  
        return count;  
    }  
}
```

#### 450. Delete Node in a BST

https://leetcode.com/problems/delete-node-in-a-bst/discuss/821420/Python-O(h)-solution-explained

Go

```go
func deleteNode(root *TreeNode, key int) *TreeNode {
   if root == nil {
      return nil
   }
   switch {
   case root.Val < key:
      root.Right = deleteNode(root.Right, key)
   case root.Val > key:
      root.Left = deleteNode(root.Left, key)
   default:
      if root.Left == nil {
         return root.Right
      }
      if root.Right == nil {
         return root.Left
      }
      tmp := root.Right
      for tmp.Left != nil {
         tmp = tmp.Left
      }
      root.Val = tmp.Val
      root.Right = deleteNode(root.Right, root.Val)
   }
   return root
}
```


#### 501. Find Mode in Binary Search Tree

```go
func findMode(root *TreeNode) []int {
   count, maxCount := 0, 0
   var (
      prev *TreeNode
      res  []int
   )
   
   var dfs func(root *TreeNode)
   dfs = func(root *TreeNode) {
      if root == nil {
         return
      }
      dfs(root.Left)
      if prev == nil || root.Val != prev.Val {
         count = 1
      } else {
         count++
      }
      if count == maxCount {
         res = append(res, root.Val)
      }
      if count > maxCount {
         maxCount = count
         res = append([]int{}, root.Val)
      }
      prev = root
      dfs(root.Right)
   }
   
   dfs(root)
   return res
}
```


#### 530. Minimum Absolute Difference in BST

https://leetcode.com/problems/minimum-absolute-difference-in-bst/solutions/99905/two-solutions-in-order-traversal-and-a-more-general-way-using-treeset/

```go
func getMinimumDifference(root *TreeNode) int {
   res := math.MaxInt32
   var prev *TreeNode
   var dfs func(root *TreeNode)
   dfs = func(root *TreeNode) {
      if root == nil {
         return
      }
      dfs(root.Left)
      if prev != nil && root.Val-prev.Val < res {
         res = root.Val - prev.Val
      }
      prev = root
      dfs(root.Right)
   }
   dfs(root)
   return res
}
```

#### 538. Convert BST to Greater Tree

```go
func convertBST(root *TreeNode) *TreeNode {
   var (
      dfs  func(root *TreeNode)
      prev *TreeNode
   )
   dfs = func(root *TreeNode) {
      if root == nil {
         return
      }
      dfs(root.Right)
      if prev != nil {
         root.Val += prev.Val
      }
      prev = root
      dfs(root.Left)
   }
   dfs(root)
   return root
}
```

#### 543. Diameter of Binary Tree

java

```java
public class Solution {  
    int res;  
  
    public int diameterOfBinaryTree(TreeNode root) {  
        res = 0;  
        dfs(root);  
        return res;  
    }  
  
    int dfs(TreeNode root) {  
        if (root == null) return 0;  
        int left = dfs(root.left);  
        int right = dfs(root.right);  
        res = Math.max(res, left + right);  
        return 1 + Math.max(left, right);  
    }  
}
```

go

```go
func diameterOfBinaryTree(root *TreeNode) int {
	res := 0
	dfs(root, &res)
	return res
}
func dfs(root *TreeNode, res *int) int {
	if root == nil {
		return 0
	}
	left := dfs(root.Left, res)
	right := dfs(root.Right, res)
	*res = max(*res, left+right)
	return max(left, right) + 1
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

#### 559. Maximum Depth of N-ary Tree

```go
func maxDepth(root *Node) int {
    if root == nil {
        return 0
    }
    res := 0
    for _, v := range root.Children {
        res = max(res, maxDepth(v))
    }
    return 1 + res
}
func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}
```

#### 572. Subtree of Another Tree

Go

```go
func isSubtree(root *TreeNode, subRoot *TreeNode) bool {
   if root == nil || subRoot == nil {
      return root == subRoot
   }
   if same(root, subRoot) {
      return true
   }
   return isSubtree(root.Left, subRoot) || isSubtree(root.Right, subRoot)
}

func same(root *TreeNode, subRoot *TreeNode) bool {
   if root == nil || subRoot == nil {
      return root == subRoot
   }
   if root.Val != subRoot.Val {
      return false
   }
   return same(root.Left, subRoot.Left) && same(root.Right, subRoot.Right)
}
```

#### 589. N-ary Tree Preorder Traversal

go

```go
func preorder(root *Node) []int {
   res := make([]int, 0)
   dfs(root, 0, &res)
   return res
}
func dfs(root *Node, level int, res *[]int) {
   if root == nil {
      return
   }
   *res = append(*res, root.Val)
   for _, c := range root.Children {
      dfs(c, level+1, res)
   }
}
```

#### 590. N-ary Tree Postorder Traversal

go

```go
func postorder(root *Node) []int {
   res := make([]int, 0)
   dfs(root, &res)
   return res
}
func dfs(root *Node, res *[]int) {
   if root == nil {
      return
   }
   for _, node := range root.Children {
      dfs(node, res)
   }
   *res = append(*res, root.Val)
}
```

#### 617. Merge Two Binary Trees

go

```go
func mergeTrees(root1 *TreeNode, root2 *TreeNode) *TreeNode {
   if root1 == nil {
      return root2
   }
   if root2 == nil {
      return root1
   }
   return &TreeNode{
      root1.Val + root2.Val,
      mergeTrees(root1.Left, root2.Left),
      mergeTrees(root1.Right, root2.Right),
   }
}
```

#### 653. Two Sum IV - Input is a BST

```go
func findTarget(root *TreeNode, k int) bool {
   m := make(map[int]bool)
   return dfs(root, k, m)
}
func dfs(root *TreeNode, k int, m map[int]bool) bool {
   if root == nil {
      return false
   }
   if m[k-root.Val] {
      return true
   }
   m[root.Val] = true
   return dfs(root.Left, k, m) || dfs(root.Right, k, m)
}
```

#### 654. Maximum Binary Tree

```go
func constructMaximumBinaryTree(nums []int) *TreeNode {
   if len(nums) < 1 {
      return nil
   }
   index := 0
   for i := 1; i < len(nums); i++ {
      if nums[i] > nums[index] {
         index = i
      }
   }
   root := &TreeNode{nums[index], nil, nil}

   root.Left = constructMaximumBinaryTree(nums[:index])
   root.Right = constructMaximumBinaryTree(nums[index+1:])
   return root
}
```

#### 669. Trim a Binary Search Tree

go

```go
func trimBST(root *TreeNode, low int, high int) *TreeNode {
   if root == nil {
      return root
   }
   lt, rt := trimBST(root.Left, low, high), trimBST(root.Right, low, high)
   if root.Val < low {
      return rt
   }
   if root.Val > high {
      return lt
   }
   root.Left = lt
   root.Right = rt
   return root
}
```

#### 700. Search in a Binary Search Tree

go

```go
func searchBST(root *TreeNode, val int) *TreeNode {
   if root == nil {
      return nil
   }
   switch {
   case root.Val < val:
      return searchBST(root.Right, val)
   case root.Val > val:
      return searchBST(root.Left, val)
   default:
      return root
   }
}
```

#### 701. Insert into a Binary Search Tree

go

```go
func insertIntoBST(root *TreeNode, val int) *TreeNode {
   if root == nil {
      return &TreeNode{Val: val}
   }
   if root.Val < val {
      root.Right = insertIntoBST(root.Right, val)
   }
   if root.Val > val {
      root.Left = insertIntoBST(root.Left, val)
   }
   return root
}
```

#### 938. Range Sum of BST

java

```java
public class Solution {  
    public int rangeSumBST(TreeNode root, int l, int r) {  
        if (root == null) return 0;  
        int val = root.val;  
        return (val <= r && val >= l ? val : 0) + rangeSumBST(root.left, l, r) + rangeSumBST(root.right, l, r);  
    }  
}
```

#### 958. Check Completeness of a Binary Tree

go

```go
func isCompleteTree(root *TreeNode) bool {
	return dfs(root, 1, countNodes(root))
}
func dfs(root *TreeNode, idx int, total int) bool {
	if root == nil {
		return true
	}
	if idx > total {
		return false
	}
	return dfs(root.Left, 2*idx, total) && dfs(root.Right, 2*idx+1, total)
}
func countNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	return countNodes(root.Left) + 1 + countNodes(root.Right)
}
```

java

```java
class Solution {
    public boolean isCompleteTree(TreeNode root) {
        return dfs(root, 1, count(root));
    }

    boolean dfs(TreeNode root, int idx, int total) {
        if (root == null) {
            return true;
        }
        if (idx > total) {
            return false;
        }
        return dfs(root.left, idx * 2, total) && dfs(root.right, idx * 2 + 1, total);
    }

    int count(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + count(root.left) + count(root.right);
    }
}
```

#### 968. Binary Tree Cameras

二叉树遍历方式也就三种，试也该试出来，一个后序遍历不就解决了吗，不要搞那么复杂

Go

```go
const (
	Leaf int = iota
	Camera
	covered
)

func minCameraCover(root *TreeNode) int {
	res := 0
	if dfs(root, &res) == Leaf {
		return res + 1
	}
	return res
}

func dfs(root *TreeNode, res *int) int {
	if root == nil {
		return covered
	}
	left, right := dfs(root.Left, res), dfs(root.Right, res)
	if left == Leaf || right == Leaf {
		*res++
		return Camera
	}
	if left == Camera || right == Camera {
		return covered
	}
	return Leaf
}
```

java

```java
interface state {
    int leaf = 0;
    int camera = 1;
    int covered = 2;
}

class Solution implements state {
    int res;

    public int minCameraCover(TreeNode root) {
        res = 0;
        if (dfs(root) == leaf) {
            res++;
        }
        return res;
    }

    int dfs(TreeNode root) {
        if (root == null) {
            return covered;
        }
        int left = dfs(root.left);
        int right = dfs(root.right);
        if (left == leaf || right == leaf) {
            res++;
            return camera;
        }
        if (left == camera || right == camera) {
            return covered;
        }
        return leaf;
    }
}
```
