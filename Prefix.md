#### 53. Maximum Subarray

有负数，必须从nums[ 0 ] 开始累加

```java
class Solution {  
    public int maxSubArray(int[] nums) {  
        int res = nums[0], acc = nums[0];  
        for (int i = 1; i < nums.length; i++) {  
            acc = Math.max(acc + nums[i], nums[i]);  
            res = Math.max(res, acc);  
        }  
        return res;  
    }  
}
```

#### 56. Merge Intervals

go

```go
func merge(intervals [][]int) [][]int {
	var events [][2]int
	for _, interval := range intervals {
		events = append(events, [2]int{interval[0], 1})
		events = append(events, [2]int{interval[1] + 1, -1})
	}
	sort.Slice(events, func(i, j int) bool {
		return events[i][0] < events[j][0] || events[i][0] == events[j][0] && events[i][1] < events[j][1]
	})
	var result [][]int
	cover := 0
	start := -1
	for _, event := range events {
		if cover == 0 {
			start = event[0]
		}
		cover += event[1]
		if cover == 0 {
			result = append(result, []int{start, event[0] - 1})
		}
	}
	return result
}
```

#### 238. Product of Array Except Self

java

```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; i++)
            res[i] = res[i - 1] * nums[i - 1];
        int right = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] *= right;
            right *= nums[i];
        }
        return res;
    }
}
```

#### 307. Range Sum Query - Mutable

java

```java
class NumArray {  
    int[] tree;  
    int[] a;  
    int n;  
  
    int lowBit(int x) {  
        return x & -x;  
    }  
  
    int query(int x) {  
        int ans = 0;  
        for (int i = x; i > 0; i -= lowBit(i)) ans += tree[i];  
        return ans;  
    }  
  
    void add(int x, int delta) {  
        for (int i = x; i < n + 1; i += lowBit(i)) tree[i] += delta;  
    }  
  
    public NumArray(int[] nums) {  
        a = nums;  
        n = nums.length;  
        tree = new int[n + 1];  
        for (int i = 0; i < n; i++) add(i + 1, a[i]);  
    }  
  
    public void update(int i, int val) {  
        add(i + 1, val - a[i]);  
        a[i] = val;  
    }  
  
    public int sumRange(int l, int r) {  
        return query(r + 1) - query(l);  
    }  
}
```

Go

```Go
type NumArray struct {
	a    []int
	tree []int
	n    int
}

func Constructor(nums []int) NumArray {
	n := len(nums)
	tree := make([]int, n+1)
	for i := 0; i < n; i++ {
		add(i+1, nums[i], tree)
	}
	return NumArray{a: nums, tree: tree, n: n}
}
func add(x int, delta int, tree []int) {
	for ; x < len(tree); x += x & -x {
		tree[x] += delta
	}
}
func (t *NumArray) Update(index int, val int) {
	add(index+1, val-t.a[index], t.tree)
	t.a[index] = val
}

func (t *NumArray) SumRange(left int, right int) int {
	return query(right+1, t.tree) - query(left, t.tree)
}
func query(x int, tree []int) int {
	res := 0
	for ; x > 0; x -= x & -x {
		res += tree[x]
	}
	return res
}
```

#### 523. Continuous Subarray Sum

java

```java
class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        int n = nums.length;
        int runningSum = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for (int i = 0; i < n; i++) {
            runningSum += nums[i];
            runningSum %= k;
            if (map.containsKey(runningSum)) {
                if (i - map.get(runningSum) > 1) return true;
                continue;
            }
            map.put(runningSum, i);
        }
        return false;
    }
}
```

go

```go
func checkSubarraySum(nums []int, k int) bool {
	sum := 0
	m := make(map[int]int)
	m[0] = -1
	for i, num := range nums {
		sum += num
		sum %= k
		if _, ok := m[sum]; !ok {
			m[sum] = i
			continue
		}
		if i-m[sum] > 1 {
			return true
		}
	}
	return false
}
```

#### 560. Subarray Sum Equals K

本题不能用常规划窗

java

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int res = 0, acc = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int num : nums) {
            acc += num;
            res += map.getOrDefault(acc - k, 0);
            map.put(acc, map.getOrDefault(acc, 0) + 1);
        }
        return res;
    }
}
```

#### 1109. Corporate Flight Bookings

java

```java
class Solution {
    public int[] corpFlightBookings(int[][] bookings, int n) {
        int[] res = new int[n];
        for (int[] b : bookings) {
            res[b[0] - 1] += b[2];
            if (b[1] < n) res[b[1]] -= b[2];
        }
        for (int i = 1; i < n; i++) {
            res[i] += res[i - 1];
        }
        return res;
    }
}
```

#### 1074. Number of Submatrices That Sum to Target

java

```java
class Solution {  
    public int numSubmatrixSumTarget(int[][] matrix, int target) {  
        int m = matrix.length, n = matrix[0].length;  
        int[][] A = new int[m + 1][n];  
        for (int i = 0; i < m; i++) {  
            for (int j = 0; j < n; j++) {  
                A[i + 1][j] = A[i][j] + matrix[i][j];  
            }  
        }  
        int[] row = new int[n];  
        int res = 0;  
        for (int up = 0; up < m; up++) {  
            for (int down = up; down < m; down++) {  
                for (int i = 0; i < n; i++) {  
                    row[i] = A[down + 1][i] - A[up][i];  
                }  
                res += subarraySum(row, target);  
            }  
        }  
        return res;  
    }  
  
    public int subarraySum(int[] nums, int k) {  
        int res = 0, acc = 0;  
        HashMap<Integer, Integer> map = new HashMap<>();  
        map.put(0, 1);  
        for (int num : nums) {  
            acc += num;  
            res += map.getOrDefault(acc - k, 0);  
            map.put(acc, map.getOrDefault(acc, 0) + 1);  
        }  
        return res;  
    }  
}
```

#### 1352. Product of the Last K Numbers

java

```java
class ProductOfNumbers {
    ArrayList<Integer> list;

    public ProductOfNumbers() {
        list = new ArrayList<>(List.of(1));
    }

    public void add(int num) {
        if (num == 0) {
            list = new ArrayList<>(List.of(1));
            return;
        }
        list.add(list.get(list.size() - 1) * num);
    }

    public int getProduct(int k) {
        int n = list.size();
        return k < n ? list.get(n - 1) / list.get(n - k - 1) : 0;
    }
}
```


