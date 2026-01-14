## 双路快排 (Dual-Pivot / Two-way Partitioning)

**核心思想：均匀分布**
双路快排将切分元素均匀分布在两个区间 `<= pivot` 和 `>= pivot`。

**核心思想：处理重复元素**
虽然两个相等值的交换看似没有意义，但这种策略可以让等于 `pivot` 的元素平均分布在数组的头尾，从而避免在包含大量重复元素的情况下（如 `[2, 2, ..., 2]`）退化为 $O(N^2)$。

**实现细节：为什么 `partition` 返回指针 r？**
因为使用的是 Hoare 分区方案（或其变体），最终 `l` 和 `r` 会交错或重合。划分点通常选为第一个区间的最后一个元素（即 `r`），保证 `[start, r]` 为左区间，`[r+1, end]` 为右区间。

**实现细节：关键点**
`pivot` 不能选右边界（最后一个元素）。从语义来说，将数组分成 `<=` 和 `>` 两部分。如果数组已经有序且 `pivot` 选了最大值（即最后一个元素），那么右区间将为空，左区间包含所有元素（且没变小），会导致无限递归（死循环）。

**实现细节：指针移动逻辑**
`while (nums[l] < pivotVal) l++;`
`while (nums[r] > pivotVal) r--;`
遇到不符合条件的就停下来，**遇到等于 `pivot` 的也停下来交换**。这正是为了“均匀分布”重复元素。指针 r 右侧绝对大于 pivot。

**优化**
小数组（如长度 < 60）使用插入排序（Insertion Sort），因为小数组时插入排序常数更小且缓存友好。

---

#### 215. Kth Largest Element in an Array

递归折半查找，时间复杂度 $O(N)$。注意：Hoare 分区返回的索引 `p` 将数组分为 `[0, p]` 和 `[p+1, len-1]`，并不保证 `nums[p]` 处于最终有序位置，因此递归时区间不能跳过 `p`。

Go

```go
import "math/rand"

func findKthLargest(nums []int, k int) int {
    n := len(nums)
    // 第 k 大 即 第 n - k 小 (0-based index)
    return quickSelect(nums, n-k)
}

func quickSelect(nums []int, k int) int {
    if len(nums) == 1 {
        return nums[0]
    }
    
    // partition 返回左区间的最后一个索引 p
    p := partition(nums)
    
    if k <= p {
        return quickSelect(nums[:p+1], k)
    } else {
        return quickSelect(nums[p+1:], k-(p+1))
    }
}

func partition(nums []int) int {
    l, r := 0, len(nums)-1
    // 随机选取 pivot，范围 [0, r-1]，避开 r 以防无限递归
    pivotIdx := rand.Intn(r) 
    pivotVal := nums[pivotIdx]
    
    for {
        for nums[l] < pivotVal { l++ }
        for nums[r] > pivotVal { r-- }
        if l >= r { return r }
        nums[l], nums[r] = nums[r], nums[l]
        l++
        r--
    }
}
```

Python

```python
import random
from typing import List

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # 第 k 大即第 n-k 小（0-based index）
        target = n - k
        return self.quick_select(nums, 0, n - 1, target)

    def quick_select(self, nums: List[int], l: int, r: int, k: int) -> int:
        if l >= r:
            return nums[l]
        
        pivot = self.partition(nums, l, r)
        
        if k <= pivot:
            return self.quick_select(nums, l, pivot, k)
        else:
            return self.quick_select(nums, pivot + 1, r, k)

    def partition(self, nums: List[int], l: int, r: int) -> int:
        # 随机选取 pivot，范围 [l, r-1]，避开 r 以防无限递归
        pivot_idx = random.randint(l, r - 1)
        pivot_val = nums[pivot_idx]
        
        while True:
            while nums[l] < pivot_val:
                l += 1
            while nums[r] > pivot_val:
                r -= 1
            if l >= r:
                return r
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
```

Rust

```rust
impl Solution {
    pub fn find_kth_largest(mut nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len();
        // 第 k 大 即 第 n - k 小的元素 (0-based)
        let target = n - k as usize;
        Self::quick_select(&mut nums, target)
    }

    fn quick_select(arr: &mut [i32], k: usize) -> i32 {
        if arr.len() == 1 {
            return arr[0];
        }
        let pivot = Self::partition(arr);
        
        // Hoare partition: [0..=pivot] <= val, [pivot+1..] >= val
        if k <= pivot {
            Self::quick_select(&mut arr[0..=pivot], k)
        } else {
            Self::quick_select(&mut arr[pivot+1..], k - (pivot + 1))
        }
    }

    fn partition(arr: &mut [i32]) -> usize {
        let mut l = 0;
        let mut r = arr.len() - 1;
        // 随机选择 pivot，范围 [l, r)，即排除最后一个元素以防无限递归
        let pivot_index = rand::random_range(l..r); 
        let pivot_value = arr[pivot_index];
        loop {
            while arr[l] < pivot_value { l += 1; }
            while arr[r] > pivot_value { r -= 1; }
            if l >= r { break; }
            arr.swap(l, r);
            l += 1;
            r -= 1;
        }
        r
    }
}
```

Java

```java
import java.util.concurrent.ThreadLocalRandom;

public class Solution {
    public int findKthLargest(int[] nums, int k) {
        int n = nums.length;
        // 第 k 大即索引为 n-k 的元素（排序后）
        return quickSelect(nums, 0, n - 1, n - k);
    }

    private int quickSelect(int[] nums, int l, int r, int k) {
        if (l >= r) return nums[l];
        int pivot = partition(nums, l, r);
        
        // Hoare partition 分割为 [l, pivot] 和 [pivot+1, r]
        if (k <= pivot) {
            return quickSelect(nums, l, pivot, k);
        } else {
            return quickSelect(nums, pivot + 1, r, k);
        }
    }

    private int partition(int[] nums, int l, int r) {
        // 随机选取 pivot，范围 [l, r-1]，避开 r 以防无限递归
        int pivot = l + ThreadLocalRandom.current().nextInt(r - l);
        int pivotVal = nums[pivot];
        while (true) {
            while (nums[l] < pivotVal) l++;
            while (nums[r] > pivotVal) r--;
            if (l >= r) {
                break;
            }
            swap(nums, l, r);
            l++;
            r--;
        }
        return r;
    }

    private void swap(int[] nums, int l, int r) {
        int tmp = nums[l];
        nums[l] = nums[r];
        nums[r] = tmp;
    }
}
```

#### 912. Sort an Array

Go

```go
import (
	"math/rand"
)

func sortArray(nums []int) []int {
	n := len(nums)
	if n < 2 {
		return nums
	}
	quickSort(nums)
	return nums
}

func quickSort(nums []int) {
	if len(nums) < 60 {
		insertionSort(nums)
		return
	}
	p := partition(nums)
	quickSort(nums[:p+1])
	quickSort(nums[p+1:])
}

func insertionSort(nums []int) {
	for i := 1; i < len(nums); i++ {
		key := nums[i]
		j := i
		for j > 0 && nums[j-1] > key {
			nums[j] = nums[j-1]
			j--
		}
		nums[j] = key
	}
}

func partition(nums []int) int {
	l, r := 0, len(nums)-1
	// 随机选取 pivot，范围 [0, r-1]，避开 r 以防无限递归
	pivotIdx := rand.Intn(r)
	pivotVal := nums[pivotIdx]

	for {
		for nums[l] < pivotVal {
			l++
		}
		for nums[r] > pivotVal {
			r--
		}
		if l >= r {
			return r
		}
		nums[l], nums[r] = nums[r], nums[l]
		l++
		r--
	}
}
```

Python

```python
import random
from typing import List

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        if n < 2:
            return nums
        self.quick_sort(nums, 0, n - 1)
        return nums

    def quick_sort(self, nums: List[int], l: int, r: int):
        if r - l < 60:
            self.insertion_sort(nums, l, r)
            return
        
        pivot = self.partition(nums, l, r)
        self.quick_sort(nums, l, pivot)
        self.quick_sort(nums, pivot + 1, r)

    def insertion_sort(self, nums: List[int], l: int, r: int):
        for i in range(l + 1, r + 1):
            key = nums[i]
            j = i
            while j > l and nums[j - 1] > key:
                nums[j] = nums[j - 1]
                j -= 1
            nums[j] = key

    def partition(self, nums: List[int], l: int, r: int) -> int:
        # 随机选取 pivot，范围 [l, r-1]，避开 r 以防无限递归
        pivot_idx = random.randint(l, r - 1)
        pivot_val = nums[pivot_idx]
        
        while True:
            while nums[l] < pivot_val:
                l += 1
            while nums[r] > pivot_val:
                r -= 1
            if l >= r:
                return r
            
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
```

Rust

```rust
impl Solution {
    pub fn sort_array(mut nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        if n < 2 {
            return nums;
        }
        Self::quick_sort_recursion(&mut nums);
        nums
    }

    fn quick_sort_recursion(arr: &mut [i32]) {
        if arr.len() < 60 {
            Self::insertion_sort(arr);
            return;
        }
        let pivot = Self::partition(arr);
        let (left, right) = arr.split_at_mut(pivot + 1);
        Self::quick_sort_recursion(left);
        Self::quick_sort_recursion(right);
    }

    fn insertion_sort(arr: &mut [i32]) {
        for i in 1..arr.len() {
            let key = arr[i];
            let mut j = i;
            while j > 0 && arr[j - 1] > key {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = key;
        }
    }

    fn partition(arr: &mut [i32]) -> usize {
        let mut l = 0;
        let mut r = arr.len() - 1;
        // 随机选择 pivot，范围 [l, r)，即排除最后一个元素以防无限递归
        let pivot_index = rand::random_range(l..r); 
        let pivot_value = arr[pivot_index];
        
        loop {
            while arr[l] < pivot_value {
                l += 1;
            }
            while arr[r] > pivot_value {
                r -= 1;
            }
            if l >= r {
                break;
            }
            arr.swap(l, r); 
            l += 1;
            r -= 1;
        }
        r
    }
}
```

Java

```java
import java.util.concurrent.ThreadLocalRandom;

public class Solution {
    public int[] sortArray(int[] nums) {
        if (nums == null) return null;
        int n = nums.length;
        if (n < 2) return nums;
        quickSort(nums, 0, n - 1);
        return nums;
    }

    private void insertionSort(int[] nums, int l, int r) {
        for (int i = l + 1; i < r + 1; i++) {
            int key = nums[i], j = i;
            while (j > l && nums[j - 1] > key) {
                nums[j] = nums[j - 1];
                j--;
            }
            nums[j] = key;
        }
    }

    private void quickSort(int[] nums, int l, int r) {
        if (r - l < 60) {
            insertionSort(nums, l, r);
            return;
        }
        int pivot = partition(nums, l, r);
        quickSort(nums, l, pivot);
        quickSort(nums, pivot + 1, r);
    }

    private int partition(int[] nums, int l, int r) {
        // 随机选取 pivot，范围 [l, r-1]，避开 r 以防无限递归
        int pivot = l + ThreadLocalRandom.current().nextInt(r - l);
        int pivotVal = nums[pivot];
        while (true) {
            while (nums[l] < pivotVal) l++;
            while (nums[r] > pivotVal) r--;
            if (l >= r) {
                break;
            }
            swap(nums, l, r);
            l++;
            r--;
        }
        return r;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```
