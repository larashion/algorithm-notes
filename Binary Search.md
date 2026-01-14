#### 744. Find Smallest Letter Greater Than Target

```java
class Solution {
    public char nextGreatestLetter(char[] letters, char target) {
        int n = letters.length;
        int left = 0, right = n;
        while (left < right) {
            int mid = (left + right) >> 1;
            if (letters[mid] <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (left == n) {
            return letters[0];
        }
        return letters[left];
    }
}
```

#### 33. Search in Rotated Sorted Array


```go
func search(nums []int, target int) int {
    left, right := 0, len(nums)
    for left < right {
        mid := (left + right) / 2
        switch {
        case target < nums[0] && nums[0] < nums[mid]:
            left = mid + 1
        case target >= nums[0] && nums[0] > nums[mid]:
            right = mid
        case nums[mid] < target:
            left = mid + 1
        case nums[mid] > target:
            right = mid
        default:
            return mid
        }
    }
    return -1
}
```

#### 34. Find First and Last Position of Element in Sorted Array

go

```go
func searchRange(nums []int, target int) []int {
	return []int{FindFirst(nums, target), FindLast(nums, target)}
}

func FindFirst(nums []int, target int) int {
	left, right, res := 0, len(nums), -1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
		if nums[mid] == target {
			res = mid
		}
	}
	return res
}
func FindLast(nums []int, target int) int {
	left, right, res := 0, len(nums), -1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] <= target {
			left = mid + 1
		} else {
			right = mid
		}
		if nums[mid] == target {
			res = mid
		}
	}
	return res
}
```

python

```python
class Solution:
    def find_first(self, nums: List[int], target: int) -> int:
        left, right, res = 0, len(nums), -1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] >= target:
                right = mid
            else:
                left = mid + 1
            if nums[mid] == target:
                res = mid
        return res

    def find_last(self, nums: List[int], target: int) -> int:
        left, right, res = 0, len(nums), -1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid
            if nums[mid] == target:
                res = mid
        return res

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        return [self.find_first(nums, target), self.find_last(nums, target)]
```

#### 35. Search Insert Position



```go
func searchInsert(nums []int, target int) int {
   left, right := 0, len(nums)
   for left < right {
      mid := (left + right) / 2
      switch {
      case nums[mid] == target:
         return mid
      case nums[mid] > target:
         right = mid
      default:
         left = mid + 1
      }
   }
   return left
}
```

#### 69. Sqrt(x)

输入一整数，求平方根

二分查找

```go
func mySqrt(x int) int {
	if x == 0 || x == 1 {
		return x
	}
	left, right := 1, x
	for left < right {
		mid := (left + right) >> 1
		switch {
		case mid == x/mid:
			return mid
		case mid > x/mid:
			right = mid
		case mid < x/mid:
			left = mid + 1
		}
	}
	return left - 1
}
```

#### 74. Search a 2D Matrix

```go
func searchMatrix(matrix [][]int, target int) bool {
   m, n := len(matrix), len(matrix[0])
   lp, rp := 0, m*n
   for lp < rp {
      mid := (lp + rp) / 2
      switch {
      case matrix[mid/n][mid%n] == target:
         return true
      case matrix[mid/n][mid%n] > target:
         rp = mid
      case matrix[mid/n][mid%n] < target:
         lp = mid + 1
      }
   }
   return false
}
```

#### 153. Find Minimum in Rotated Sorted Array

```go
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] > nums[right] {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return nums[left]
}
```
#### 162. Find Peak Element

https://leetcode.com/problems/find-peak-element/discuss/50232/Find-the-maximum-by-binary-search-(recursion-and-iteration)

```go
func findPeakElement(nums []int) int {
   left, right := 0, len(nums)-1
   for left < right {
      mid1 := (left + right) / 2
      mid2 := mid1 + 1
      if nums[mid1] < nums[mid2] {
         left = mid2
      } else {
         right = mid1
      }
   }
   return left
}
```



#### 240. Search a 2D Matrix II

go

```go
func searchMatrix(matrix [][]int, target int) bool {
   for i := range matrix {
      if matrix[i][0] <= target && matrix[i][len(matrix[0])-1] >= target {
         left, right := 0, len(matrix[0])
         for left < right {
            mid := (left + right) / 2
            switch {
            case matrix[i][mid] == target:
               return true
            case matrix[i][mid] > target:
               right = mid
            default:
               left = mid + 1
            }
         }
      }
   }
   return false
}
```


#### 278. First Bad Version

```go
func firstBadVersion(n int) int {  
   left, right := 1, n+1  
   for left < right {  
      mid := (left + right) / 2  
      if isBadVersion(mid) {  
         right = mid  
      } else {  
         left = mid + 1  
      }  
   }  
   return left  
}
```

go

```go
func firstBadVersion(n int) int {  
   return sort.Search(n, isBadVersion)  
}
```

#### 378. Kth Smallest Element in a Sorted Matrix

```go
func kthSmallest(matrix [][]int, k int) int {  
   m, n := len(matrix), len(matrix[0])  
   left, right := matrix[0][0], matrix[m-1][n-1]  
   for left < right {  
      mid := (left + right) >> 1  
      count := 0  
      for i := 0; i < m; i++ {  
         j := n - 1  
         for j >= 0 && matrix[i][j] > mid {  
            j--  
         }  
         count += j + 1  
      }  
      if count < k {  
         left = mid + 1  
      } else {  
         right = mid  
      }  
   }  
   return left  
}
```

#### 410. Split Array Largest Sum

java

```java
class Solution {
    public int splitArray(int[] nums, int m) {
        int n = nums.length;
        int left = Arrays.stream(nums).max().getAsInt();
        int right = Arrays.stream(nums).sum();
        while (left < right) {
            int mid = (left + right) / 2;
            if (valid(nums, m, mid)) right = mid;
            else left = mid + 1;
        }
        return right;
    }

    boolean valid(int[] nums, int m, int size) {
        int box = 0, count = 1;
        for (int num : nums) {
            if (box + num <= size) box += num;
            else {
                count++;
                box = num;
            }
        }
        return count <= m;
    }
}
```

#### 475. Heaters

java

```java
class Solution {
    public int findRadius(int[] houses, int[] heaters) {
        Arrays.sort(heaters);
        int res = 0;
        for (int house : houses) {
            int i = Arrays.binarySearch(heaters, house);
            if (i >= 0) continue;
            i = ~i;
            int dist1 = i - 1 >= 0 ? house - heaters[i - 1] : Integer.MAX_VALUE;
            int dist2 = i < heaters.length ? heaters[i] - house : Integer.MAX_VALUE;
            res = Math.max(res, Math.min(dist1, dist2));
        }
        return res;
    }
}
```

go

```go
func findRadius(houses []int, heaters []int) int {
	sort.Ints(heaters)
	res := 0
	for _, house := range houses {
		i := sort.SearchInts(heaters, house)
		dist1 := math.MaxInt32
		if i-1 >= 0 {
			dist1 = house - heaters[i-1]
		}
		dist2 := math.MaxInt32
		if i < len(heaters) {
			dist2 = heaters[i] - house
		}
		res = max(res, min(dist1, dist2))
	}
	return res
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}
```

#### 658. Find K Closest Elements

https://leetcode.com/problems/find-k-closest-elements/solutions/106426/JavaC++Python-Binary-Search-O(log(N-K)-+-K)/



```go
func findClosestElements(arr []int, k int, x int) []int {
   left, right := 0, len(arr)-k
   for left < right {
      mid := (left + right) / 2
      if x-arr[mid] > arr[mid+k]-x {
         left = mid + 1
      } else {
         right = mid
      }
   }
   return arr[left : left+k]
}
```

#### 704. Binary Search

```go
func search(nums []int, target int) int {  
   left, right := 0, len(nums)  
   for left < right {  
      mid := (left + right) >> 1  
      if nums[mid] > target {  
         right = mid  
         continue  
      }  
      if nums[mid] < target {  
         left = mid + 1  
         continue  
      }  
      return mid  
   }  
   return -1  
}
```

#### 852. Peak Index in a Mountain Array

```go
func peakIndexInMountainArray(arr []int) int {
	left, right := 0, len(arr)
	for left < right {
		mid := (left + right) >> 1
		if arr[mid] > arr[mid+1] {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}
```

#### 875. Koko Eating Bananas

go

```go
func minEatingSpeed(piles []int, h int) int {
	left, right := 1, maxValue(piles)
	for left < right {
		mid := (left + right) / 2
		if canEat(piles, h, mid) {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return right
}
func canEat(piles []int, h int, mid int) bool {
	timer := 0
	for _, v := range piles {
		timer += v / mid
		if v%mid != 0 {
			timer++
		}
	}
	return timer <= h
}
func maxValue(values []int) int {
	res := values[0]
	for _, value := range values {
		if value > res {
			res = value
		}
	}
	return res
}
```

java

```java
class Solution {
    public int minEatingSpeed(int[] piles, int H) {
        int left = 1, right = Arrays.stream(piles).max().getAsInt();
        while (left < right) {
            // Don't forget ()
            int mid = left + ((right - left) >> 1);
            if (canEatAll(piles, mid, H)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    private boolean canEatAll(int[] piles, int mid, int H) {
        long count = 0;
        for (int num : piles) {
            count += num / mid;
            if (num % mid != 0) count++;
        }
        return count <= H;
    }
}
```


#### 1482. Minimum Number of Days to Make m Bouquets

java

```java
class Solution {
    public int minDays(int[] bloomDay, int m, int k) {
        int last = Arrays.stream(bloomDay).max().getAsInt();
        int left = 1, right = last + 1;
        while (left < right) {
            // Don't forget ()
            int mid = left + ((right - left) >> 1);
            if (valid(bloomDay, mid, m, k)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left == last + 1 ? -1 : left;
    }

    private boolean valid(int[] bloomDay, int mid, int m, int k) {
        int bouquet = 0;
        int adjacent = 0;
        for (int day : bloomDay) {
            if (day <= mid) {
                adjacent++;
                if (adjacent >= k) {
                    bouquet++;
                    adjacent = 0;
                }
            } else adjacent = 0;
        }
        return bouquet >= m;
    }
}
```

#### 1539. Kth Missing Positive Number

java

```java
class Solution {
    public int findKthPositive(int[] arr, int k) {
        int left = 0, right = arr.length;
        while (left < right) {
            int mid = (left + right) >> 1;
            // the number of missing positive integer
            if (arr[mid] - mid - 1 < k) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        // arr[left - 1] + k - (arr[left - 1] - (left - 1) - 1)
        return left + k;
    }
}
```

