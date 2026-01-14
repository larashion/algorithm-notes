#### 2. Add Two Numbers

Merge Sort, Linked List

We need to use the two **reverse ordered** linked lists, representing two integers, to return the sum as a linked list.

The given linked lists is reversed, we read the digit of the integer from the right to left, then add digit by digit. The length of the integer might be different, and the digit sum could be larger than 10 , so we need a variable 'carry' , so wirte a while loop until both of lists end and carry ends up with 0.

python

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        curr, carry = dummy, 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            curr.next = ListNode(carry % 10)
            curr = curr.next
            carry //= 10
        return dummy.next
```

go

```go
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy, carry := &ListNode{}, 0
	for curr := dummy; l1 != nil || l2 != nil || carry != 0; curr = curr.Next {
		if l1 != nil {
			carry += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			carry += l2.Val
			l2 = l2.Next
		}
		curr.Next = &ListNode{Val: carry % 10}
		carry /= 10
	}
	return dummy.Next
}
```

java

```java
class Solution {  
  public ListNode addTwoNumbers(ListNode l1, ListNode l2) {  
    int carry = 0;  
    ListNode dummy = new ListNode();  
    ListNode curr = dummy;  
    while (l1 != null || l2 != null || carry > 0) {  
      if (l1 != null) {  
        carry += l1.val;  
        l1 = l1.next;  
      }  
      if (l2 != null) {  
        carry += l2.val;  
        l2 = l2.next;  
      }  
      curr.next = new ListNode(carry % 10);  
      curr = curr.next;  
      carry /= 10;  
    }  
    return dummy.next;  
  }  
}
```

#### 21. Merge Two Sorted Lists

合并两个有序链表

迭代 go

```go
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
    dummyHead := &ListNode{}
    pointer := dummyHead
    for list1 != nil && list2 != nil {
        if list1.Val < list2.Val {
            pointer.Next = list1
            list1 = list1.Next
        }else {
            pointer.Next = list2
            list2 = list2.Next
        }
        pointer = pointer.Next
    }
    if list1 != nil {
        pointer.Next = list1
    }
    if list2 != nil {
        pointer.Next = list2
    }
    return dummyHead.Next
}
```

#### 23. Merge k Sorted Lists

go

```go
func mergeKLists(lists []*ListNode) *ListNode {
    if len(lists) == 0 {
        return nil
    }
    for len(lists) > 1 {
        l1, l2 := lists[0], lists[1]
        lists = lists[2:]
        lists = append(lists, merge2Lists(l1, l2))
    }
    return lists[0]
}
func merge2Lists(l1, l2 *ListNode) *ListNode {
    if l1 == nil {
        return l2
    }
    if l2 == nil {
        return l1
    }
    if l1.Val < l2.Val {
        l1.Next = merge2Lists(l1.Next, l2)
        return l1
    } else {
        l2.Next = merge2Lists(l2.Next, l1)
        return l2
    }
}
```

java

```java
class Solution {  
    public ListNode mergeKLists(ListNode[] lists) {  
        if (lists == null || lists.length == 0) {  
            return null;  
        }  
        LinkedList<ListNode> res = new LinkedList<>();  
        Collections.addAll(res, lists);  
        while (res.size() > 1) {  
            ListNode l1 = res.remove();  
            ListNode l2 = res.remove();  
            res.offer(merge2(l1, l2));  
        }  
        return res.get(0);  
    }  
  
    ListNode merge2(ListNode l1, ListNode l2) {  
        if (l1 == null) return l2;  
        if (l2 == null) return l1;  
        if (l1.val < l2.val) {  
            l1.next = merge2(l1.next, l2);  
            return l1;  
        } else {  
            l2.next = merge2(l2.next, l1);  
            return l2;  
        }  
    }  
}
```

#### 88. Merge Sorted Array

go

```go
func merge(nums1 []int, m int, nums2 []int, n int) {
   i, j, k := m-1, n-1, len(nums1)-1
   for j >= 0 {
      if i >= 0 && nums1[i] > nums2[j] {
         nums1[k] = nums1[i]
         i--
      } else {
         nums1[k] = nums2[j]
         j--
      }
      k--
   }
}
```
#### 315. Count of Smaller Numbers After Self

java

```java
public class Solution {
    record pair(int val, int idx) {

    }

    public List<Integer> countSmaller(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        pair[] tuples = new pair[n];
        for (int i = 0; i < n; i++) {
            tuples[i] = new pair(nums[i], i);
        }
        mergeSort(0, n, res, tuples);
        return Arrays.stream(res).boxed().toList();
    }

    void mergeSort(int begin, int end, int[] res, pair[] tuples) {
        if (end - begin < 2) return;
        int mid = (begin + end) / 2;
        mergeSort(begin, mid, res, tuples);
        mergeSort(mid, end, res, tuples);
        cal(begin, mid, end, res, tuples);
        merge2(tuples, begin, mid, end);
    }

    void merge2(pair[] A, int begin, int mid, int end) {
        int i = begin, j = mid, pt = 0;
        pair[] tmp = new pair[end - begin];
        while (i < mid && j < end) {
            if (A[i].val < A[j].val) {
                tmp[pt++] = A[i++];
            } else {
                tmp[pt++] = A[j++];
            }
        }
        while (i < mid) tmp[pt++] = A[i++];
        while (j < end) tmp[pt++] = A[j++];
        System.arraycopy(tmp, 0, A, begin, end - begin);
    }

    void cal(int begin, int mid, int end, int[] res, pair[] tuples) {
        for (int i = begin, j = mid; i != mid; i++) {
            while (j != end && tuples[i].val > tuples[j].val) {
                ++j;
            }
            res[tuples[i].idx] += j - mid;
        }
    }
}
```

#### 327. Count of Range Sum

Go

```go
func countRangeSum(nums []int, lower int, upper int) int {
	n := len(nums)
	first := make([]int, n+1)
	for i := range nums {
		first[i+1] = first[i] + nums[i]
	}
	res := 0
	mergeSort(first, &res, lower, upper)
	return res
}
func mergeSort(first []int, res *int, lower int, upper int) {
	n := len(first)
	if n < 2 {
		return
	}
	mid := n >> 1
	left := first[:mid]
	right := first[mid:]
	mergeSort(left, res, lower, upper)
	mergeSort(right, res, lower, upper)
	cal(first, mid, n, res, lower, upper)
	merge2(first, left, right)
}
func cal(first []int, mid int, n int, res *int, lower int, upper int) {
	i, j := mid, mid
	for _, left := range first[:mid] {
		// the index of the first 'right'
		for i < n && first[i]-left < lower {
			i++
		}
		// the index of the last 'right' + 1
		for j < n && first[j]-left <= upper {
			j++
		}
		*res += j - i
	}
}
func merge2(nums []int, A []int, B []int) {
	m, n := len(A), len(B)
	res := make([]int, m+n)
	i, j, k := 0, 0, 0
	for ; i < m && j < n; k++ {
		if A[i] < B[j] {
			res[k] = A[i]
			i++
		} else {
			res[k] = B[j]
			j++
		}
	}
	for ; i < m; k++ {
		res[k] = A[i]
		i++
	}
	for ; j < n; k++ {
		res[k] = B[j]
		j++
	}
	copy(nums, res)
}
```

rust

```rust
impl Solution {  
    pub fn count_range_sum(nums: Vec<i32>, lower: i32, upper: i32) -> i32 {  
        let mut res = 0;  
        let n = nums.len();  
        let mut prefix = vec![0; n + 1];  
        for i in 0..n {  
            prefix[i + 1] = prefix[i] + nums[i] as i64;  
        }  
        Solution::merge_sort(&mut prefix, &mut res, lower, upper);  
        res  
    }  
    fn merge_sort(nums: &mut [i64], res: &mut i32, lower: i32, upper: i32) {  
        let n = nums.len();  
        if n < 2 {  
            return;  
        }  
        let mid = n >> 1;  
        Solution::merge_sort(&mut nums[..mid], res, lower, upper);  
        Solution::merge_sort(&mut nums[mid..], res, lower, upper);  
        Solution::cal(nums, mid, n, res, lower, upper);  
        Solution::merge2(nums, mid, n);  
    }  
    fn cal(prefix: &[i64], mid: usize, n: usize, count: &mut i32, lower: i32, upper: i32) {  
        let mut i = mid;  
        let mut j = mid;  
        for &left in &prefix[..mid] {  
            while i < n && prefix[i] - left < lower as i64 {  
                i += 1;  
            }  
            while j < n && prefix[j] - left <= upper as i64 {  
                j += 1;  
            }  
            *count += (j - i) as i32;  
        }  
    }  
    fn merge2(nums: &mut [i64], mid: usize, n: usize) {  
        let mut i = 0;  
        let mut j = mid;  
        let mut k = 0;  
        let mut res = vec![0; n];  
  
        while i < mid && j < n {  
            if nums[i] < nums[j] {  
                res[k] = nums[i];  
                i += 1;  
            } else {  
                res[k] = nums[j];  
                j += 1;  
            }  
            k += 1;  
        }  
        while i < mid {  
            res[k] = nums[i];  
            i += 1;  
            k += 1;  
        }  
        while j < n {  
            res[k] = nums[j];  
            j += 1;  
            k += 1;  
        }  
        nums.clone_from_slice(&res)  
    }  
}
```

#### 493. Reverse Pairs

https://leetcode.com/problems/reverse-pairs/discuss/97287/C%2B%2B-with-iterators

go

```go
func reversePairs(nums []int) int {
	res := 0
	mergeSort(nums, &res)
	return res
}
func mergeSort(nums []int, res *int) {
	if len(nums) < 2 {
		return
	}
	mid := len(nums) >> 1
	left := nums[:mid]
	right := nums[mid:]
	mergeSort(left, res)
	mergeSort(right, res)
	cal(left, right, res)
	merge2(nums, left, right)
}
func cal(left []int, right []int, res *int) {
	j := 0
	for i := range left {
		for j < len(right) && left[i] > 2*right[j] {
			j++
		}
		*res += j
	}
}
func merge2(nums []int, A []int, B []int) {
	m, n := len(A), len(B)
	res := make([]int, m+n)
	i, j, k := 0, 0, 0
	for ; i < m && j < n; k++ {
		if A[i] < B[j] {
			res[k] = A[i]
			i++
		} else {
			res[k] = B[j]
			j++
		}
	}
	for ; i < m; k++ {
		res[k] = A[i]
		i++
	}
	for ; j < n; k++ {
		res[k] = B[j]
		j++
	}
	copy(nums, res)
}
```

rust 50ms

```rust
impl Solution {
    pub fn reverse_pairs(nums: Vec<i32>) -> i32 {
        let mut res = 0;
        let mut nums = nums.clone();
        Solution::merge_sort(&mut nums, &mut res);
        res
    }
    fn merge_sort(nums: &mut [i32], res: &mut i32) {
        let n = nums.len();
        if n < 2 {
            return;
        }
        let mid = n >> 1;
        Solution::merge_sort(&mut nums[..mid], res);
        Solution::merge_sort(&mut nums[mid..], res);
        Solution::cal(nums, mid, n, res);
        Solution::merge2(nums, mid, n);
    }
    fn cal(nums: &[i32], mid: usize, n: usize, count: &mut i32) {
        let mut i = mid;
        for x in &nums[..mid] {
            while i < n && *x as i64 > 2 * nums[i] as i64 {
                i += 1;
            }
            *count += (i - mid) as i32;
        }
    }
    fn merge2(nums: &mut [i32], mid: usize, n: usize) {
        let mut i = 0;
        let mut j = mid;
        let mut k = 0;
        let mut res = vec![0; n];
        
        while i < mid && j < n {
            if nums[i] < nums[j] {
                res[k] = nums[i];
                i += 1;
            } else {
                res[k] = nums[j];
                j += 1;
            }
            k += 1;
        }
        while i < mid {
            res[k] = nums[i];
            i += 1;
            k += 1;
        }
        while j < n {
            res[k] = nums[j];
            j += 1;
            k += 1;
        }
        nums.clone_from_slice(&res)
    }
}
```

java

```java
public class Solution {
    int ans;

    public int reversePairs(int[] nums) {
        int n = nums.length;
        ans = 0;
        mergeSort(0, n, nums);
        return ans;
    }

    void mergeSort(int begin, int end, int[] nums) {
        if (end - begin < 2) return;
        int mid = (begin + end) / 2;
        mergeSort(begin, mid, nums);
        mergeSort(mid, end, nums);
        cal(begin, mid, end, nums);
        merge2(nums, begin, mid, end);
    }

    void merge2(int[] A, int begin, int mid, int end) {
        int i = begin, j = mid, pt = 0;
        int[] tmp = new int[end - begin];
        while (i < mid && j < end) {
            if (A[i] < A[j]) {
                tmp[pt++] = A[i++];
            } else {
                tmp[pt++] = A[j++];
            }
        }
        while (i < mid) tmp[pt++] = A[i++];
        while (j < end) tmp[pt++] = A[j++];
        System.arraycopy(tmp, 0, A, begin, end - begin);
    }

    void cal(int start, int mid, int end, int[] A) {
        for (int i = start, j = mid; i < mid; i++) {
            while (j < end && A[i] > 2L * A[j]) j++;
            ans += j - mid;
        }
    }
}
```


#### 977.Squares of a Sorted Array

给定一个非降序数组，返回每个元素的平方组成的数组，也要非降序

在原有基础上平方的话，最小的负数的平方可能是最大的。

采用双指针分别指向给定数组的头尾，再创建结果集res，比较胜出者填入res

```go
func sortedSquares(nums []int) []int {
   res := make([]int, len(nums))
   i, j := 0, len(nums)-1
   for i <= j {
      for k := len(nums) - 1; k >= 0; k-- {
         if nums[i]*nums[i] <= nums[j]*nums[j] {
            res[k] = nums[j] * nums[j]
            j--
         } else {
            res[k] = nums[i] * nums[i]
            i++
         }
      }
   }
   return res
}
```
