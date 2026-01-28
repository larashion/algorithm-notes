## 双路快排 (Two-way Partitioning)

**核心思想：均匀分布**
双路快排将切分元素均匀分布在两个区间 `<= pivot` 和 `>= pivot`。这不仅能处理重复元素，还能在包含大量重复元素的情况下（如 `[2, 2, ..., 2]`）避免退化为 $O(N^2)$。

**实现细节：Move-to-Head 策略**
本文件统一采用将 `pivot` 交换至区间头部 `l` 的策略：
1. **随机选取**：在当前区间 `[l, r)` 中随机选取一个索引作为 `pivot`。
2. **交换至头部**：将 `pivot` 交换至索引 `l`，暂存 `pivotVal`。
3. **双向扫描**：左指针 `i` 从 `l+1` 开始，右指针 `j` 从 `r-1` 开始，向中间靠拢并交换违规元素。
4. **归位**：扫描结束后，将头部索引 `l` 处的 `pivot` 与 `j` 处元素交换。此时 `j` 为 `pivot` 的最终位置。
5. **递归**：下一轮递归区间分别为 `[l, j)` 和 `[j+1, r)`，**完全跳过已经归位的索引 `j`**。

这种策略允许等于 `pivot` 的元素平均分布在数组两侧。

**优化**
小数组（如长度 < 47）使用插入排序（Insertion Sort），因为小数组时插入排序常数更小且缓存友好。

---

#### 215. Kth Largest Element in an Array

递归折半查找，时间复杂度 $O(N)$。注意：采用 Move-to-Head 策略的分区函数返回的索引 `j` 保证 **`nums[j]` 已处于最终有序位置**，因此递归时可以根据 `j` 与 `k` 的关系直接剪枝或缩小区间。

Go

```go
import "math/rand"

func findKthLargest(nums []int, k int) int {
    n := len(nums)
    // 第 k 大 即 第 n - k 小 (0-based index)
    return quickSelect(nums, 0, n, n-k)
}

func quickSelect(nums []int, l, r, k int) int {
    if r-l <= 1 {
        return nums[l]
    }
    
    // partition 返回 pivot 的最终位置 j
    j := partition(nums, l, r)
    
    if k == j {
        return nums[j]
    }
    if k < j {
        return quickSelect(nums, l, j, k)
    }
    return quickSelect(nums, j+1, r, k)
}

func partition(nums []int, l, r int) int {
    // 随机选取 pivot 并交换到头部
    pivotIdx := l + rand.Intn(r-l)
    // Move to Head
    nums[l], nums[pivotIdx] = nums[pivotIdx], nums[l]
    pivotVal := nums[l]
    
    i, j := l+1, r-1
    for {
        // 向右找到第一个 >= pivotVal 的元素
        for i <= j && nums[i] < pivotVal { i++ }
        // 向左找到第一个 <= pivotVal 的元素
        for i <= j && nums[j] > pivotVal { j-- }
        if i >= j { break }
        nums[i], nums[j] = nums[j], nums[i]
        i++
        j--
    }
    // 将 pivot 放到最终位置 j
    nums[l], nums[j] = nums[j], nums[l]
    return j
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
        return self.quick_select(nums, 0, n, target)

    def quick_select(self, nums: List[int], l: int, r: int, target: int) -> int:
        if r - l <= 1:
            return nums[l]
        
        # partition 返回 pivot 的最终位置 j
        j = self.partition(nums, l, r)
        
        if target == j:
            return nums[j]
        if target < j:
            return self.quick_select(nums, l, j, target)
        return self.quick_select(nums, j + 1, r, target)

    def partition(self, nums: List[int], l: int, r: int) -> int:
        # 随机选取 pivot 并交换到头部
        pivot_idx = random.randint(l, r - 1)
        # Move to Head
        nums[l], nums[pivot_idx] = nums[pivot_idx], nums[l]
        pivot_val = nums[l]
        
        i, j = l + 1, r - 1
        while True:
            # 向右找到第一个 >= pivotVal 的元素
            while i <= j and nums[i] < pivot_val:
                i += 1
            # 向左找到第一个 <= pivotVal 的元素
            while i <= j and nums[j] > pivot_val:
                j -= 1
            if i >= j:
                break
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
            
        # 将 pivot 放到最终位置 j
        nums[l], nums[j] = nums[j], nums[l]
        return j
```

Rust

```rust
impl Solution {
    pub fn find_kth_largest(mut nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len();
        // 第 k 大 即 第 n - k 小的元素 (0-based)
        let target = n - k as usize;
        Self::quick_select(&mut nums, 0, n, target)
    }

    fn quick_select(arr: &mut [i32], l: usize, r: usize, k: usize) -> i32 {
        if r - l <= 1 {
            return arr[l];
        }
        let j = Self::partition(arr, l, r);
        
        if k == j {
            return arr[j];
        } 
        if k < j {
            return Self::quick_select(arr, l, j, k);
        } 
        Self::quick_select(arr, j + 1, r, k)
    }

    fn partition(arr: &mut [i32], l: usize, r: usize) -> usize {
        // 随机选取 pivot 并交换到头部
        let pivot_index = rand::random_range(l..r); 
        // Move to Head
        arr.swap(l, pivot_index);
        let pivot_value = arr[l];
        
        let mut i = l + 1;
        let mut j = r - 1;
        
        loop {
            // 向右找到第一个 >= pivotVal 的元素
            while i <= j && arr[i] < pivot_value { i += 1; }
            // 向左找到第一个 <= pivotVal 的元素
            while i <= j && arr[j] > pivot_value { j -= 1; }
            if i >= j { break; }
            arr.swap(i, j);
            i += 1;
            j -= 1;
        }
        // 将 pivot 放到最终位置 j
        arr.swap(l, j);
        j
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
        return quickSelect(nums, 0, n, n - k);
    }

    private int quickSelect(int[] nums, int l, int r, int k) {
        if (r - l <= 1) return nums[l];
        
        int j = partition(nums, l, r);
        
        if (k == j) {
            return nums[j];
        } 
        if (k < j) {
            return quickSelect(nums, l, j, k);
        } 
        return quickSelect(nums, j + 1, r, k);
    }    private int partition(int[] nums, int l, int r) {
        // 随机选取 pivot 并交换到头部
        int pivotIdx = l + ThreadLocalRandom.current().nextInt(r - l);
        // Move to Head
        swap(nums, l, pivotIdx);
        int pivotVal = nums[l];
        
        int i = l + 1, j = r - 1;
        while (true) {
            // 向右找到第一个 >= pivotVal 的元素
            while (i <= j && nums[i] < pivotVal) i++;
            // 向左找到第一个 <= pivotVal 的元素
            while (i <= j && nums[j] > pivotVal) j--;
            if (i >= j) {
                break;
            }
            swap(nums, i, j);
            i++;
            j--;
        }
        // 将 pivot 放到最终位置 j
        swap(nums, l, j);
        return j;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
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
	quickSort(nums, 0, n)
	return nums
}

func quickSort(nums []int, l, r int) {
	if r-l < 47 {
		insertionSort(nums, l, r)
		return
	}
	// j 是 pivot 的最终位置
	j := partition(nums, l, r)
	quickSort(nums, l, j)
	quickSort(nums, j+1, r)
}

func insertionSort(nums []int, l, r int) {
	for i := l + 1; i < r; i++ {
		key := nums[i]
		j := i
		for j > l && nums[j-1] > key {
			nums[j] = nums[j-1]
			j--
		}
		nums[j] = key
	}
}

func partition(nums []int, l, r int) int {
	// 随机选取 pivot 并交换到头部
	pivotIdx := l + rand.Intn(r-l)
	// Move to Head
	nums[l], nums[pivotIdx] = nums[pivotIdx], nums[l]
	pivotVal := nums[l]

	i, j := l+1, r-1
	for {
		for i <= j && nums[i] < pivotVal {
			i++
		}
		for i <= j && nums[j] > pivotVal {
			j--
		}
		if i >= j {
			break
		}
		nums[i], nums[j] = nums[j], nums[i]
		i++
		j--
	}
	// 将 pivot 放到最终位置 j
	nums[l], nums[j] = nums[j], nums[l]
	return j
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
        self.quick_sort(nums, 0, n)
        return nums

    def quick_sort(self, nums: List[int], l: int, r: int):
        if r - l < 47:
            self.insertion_sort(nums, l, r)
            return
        
        # j 是 pivot 的最终位置
        j = self.partition(nums, l, r)
        self.quick_sort(nums, l, j)
        self.quick_sort(nums, j + 1, r)

    def insertion_sort(self, nums: List[int], l: int, r: int):
        for i in range(l + 1, r):
            key = nums[i]
            j = i
            while j > l and nums[j - 1] > key:
                nums[j] = nums[j - 1]
                j -= 1
            nums[j] = key

    def partition(self, nums: List[int], l: int, r: int) -> int:
        # 随机选取 pivot 并交换到头部
        pivot_idx = random.randint(l, r - 1)
        # Move to Head
        nums[l], nums[pivot_idx] = nums[pivot_idx], nums[l]
        pivot_val = nums[l]
        
        i, j = l + 1, r - 1
        while True:
            while i <= j and nums[i] < pivot_val:
                i += 1
            while i <= j and nums[j] > pivot_val:
                j -= 1
            if i >= j:
                break
            
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
        
        # 将 pivot 放到最终位置 j
        nums[l], nums[j] = nums[j], nums[l]
        return j
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
        let len = arr.len();
        if len < 47 {
            Self::insertion_sort(arr);
            return;
        }
        let j = Self::partition(arr);
        // j 是 pivot 的最终位置 (相对索引)
        // arr[0..j] 是左半边, arr[j] 是 pivot, arr[j+1..] 是右半边
        let (left, right) = arr.split_at_mut(j);
        Self::quick_sort_recursion(left);
        Self::quick_sort_recursion(&mut right[1..]);
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
        let len = arr.len();
        // 随机选取 pivot 并交换到头部
        let pivot_index = rand::random_range(0..len); 
        // Move to Head
        arr.swap(0, pivot_index);
        let pivot_value = arr[0];
        
        let mut i = 1;
        let mut j = len - 1;
        
        loop {
            while i <= j && arr[i] < pivot_value {
                i += 1;
            }
            while i <= j && arr[j] > pivot_value {
                j -= 1;
            }
            if i >= j {
                break;
            }
            arr.swap(i, j); 
            i += 1;
            j -= 1;
        }
        // 将 pivot 放到最终位置 j
        arr.swap(0, j);
        j
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
        quickSort(nums, 0, n);
        return nums;
    }

    private void insertionSort(int[] nums, int l, int r) {
        for (int i = l + 1; i < r; i++) {
            int key = nums[i], j = i;
            while (j > l && nums[j - 1] > key) {
                nums[j] = nums[j - 1];
                j--;
            }
            nums[j] = key;
        }
    }

    private void quickSort(int[] nums, int l, int r) {
        if (r - l < 47) {
            insertionSort(nums, l, r);
            return;
        }
        int j = partition(nums, l, r);
        quickSort(nums, l, j);
        quickSort(nums, j + 1, r);
    }

    private int partition(int[] nums, int l, int r) {
        // 随机选取 pivot 并交换到头部
        int pivotIdx = l + ThreadLocalRandom.current().nextInt(r - l);
        // Move to Head
        swap(nums, l, pivotIdx);
        int pivotVal = nums[l];
        
        int i = l + 1, j = r - 1;
        while (true) {
            while (i <= j && nums[i] < pivotVal) i++;
            while (i <= j && nums[j] > pivotVal) j--;
            if (i >= j) {
                break;
            }
            swap(nums, i, j);
            i++;
            j--;
        }
        // 将 pivot 放到最终位置 j
        swap(nums, l, j);
        return j;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```
java示例
```java
/**
 * Single-threaded QuickSort implementation using Dual-Way Partitioning.
 * <p>
 * Design choices:
 * 1. Left-closed, Right-open intervals for standard compliance.
 * 2. Randomized Pivot placed at Head (l) to guarantee partition invariant.
 * 3. Dual-Way scanning to handle duplicate elements efficiently (avoiding O(N^2) on all-equal arrays).
 */
class QuickSort implements Sorter {

    private static final int INSERTION_THRESHOLD = 47;
    private final Random random = new Random();

    @Override
    public void sort(int[] nums) {
        if (nums == null || nums.length < 2) return;
        quickSort(nums, 0, nums.length);
    }

    private void quickSort(int[] nums, int l, int r) {
        if (r - l <= INSERTION_THRESHOLD) {
            insertionSort(nums, l, r);
            return;
        }
        // last index in the left half
        int j = partition(nums, l, r);

        quickSort(nums, l, j);
        quickSort(nums, j + 1, r);
    }

    private int partition(int[] nums, int l, int r) {
        // Random Selection: Avoid worst-case on sorted arrays
        int pivotIdx = l + random.nextInt(r - l);
        // Pivot Value
        int pivot = nums[pivotIdx];
        // Move to Head
        swap(nums, l, pivotIdx);
        // l is the pivot
        int i = l + 1;
        int j = r - 1;
        while (true) {
            while (i <= j && nums[i] < pivot) {
                i++;
            }
            while (i <= j && nums[j] > pivot) {
                j--;
            }
            if (i >= j) {
                break;
            }
            swap(nums, i, j);
            i++;
            j--;
        }
        // j stopped at a value <= pivot.
        swap(nums, l, j);
        return j;
    }

    private void insertionSort(int[] nums, int l, int r) {
        for (int i = l + 1; i < r; i++) {
            int key = nums[i];
            int j = i;
            while (j > l && nums[j - 1] > key) {
                nums[j] = nums[j - 1];
                j--;
            }
            nums[j] = key;
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```