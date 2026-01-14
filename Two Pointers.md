#### 1498. Number of Subsequences That Satisfy the Given Sum Condition

java

```java
class Solution {
    public int numSubseq(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int l = 0, r = n - 1, res = 0;
        int mod = (int) 1e9 + 7;
        // there is a risk of integer overflow when calculating the power
        int[] pow = new int[n];
        pow[0] = 1;
        for (int i = 1; i < n; i++) {
            pow[i] = pow[i - 1] * 2 % mod;
        }
        while (l <= r) {
            if (nums[l] + nums[r] <= target) {
                res = (res + pow[r - l]) % mod;
                l++;
            } else {
                r--;
            }
        }
        return res;
    }
}
```

#### 11. Container With Most Water

```go
func maxArea(height []int) int {  
   n := len(height)  
   left, right := 0, n-1  
   res := 0  
   for left < right {  
      var lower int  
      if height[left] < height[right] {  
         lower = height[left]  
         left++  
      } else {  
         lower = height[right]  
         right--  
      }  
      res = max(res, lower*(right-left+1))  
   }  
   return res  
}  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}
```

#### 15. 3Sum


```go
func threeSum(nums []int) [][]int {  
   var res [][]int  
   sort.Ints(nums)  
   for i := range nums {  
      if i > 0 && nums[i] == nums[i-1] {  
         continue  
      }  
      if nums[i] > 0 {  
         return res  
      }  
      n := len(nums)  
      j, k := i+1, n-1  
      for j < k {  
         sum := nums[i] + nums[j] + nums[k]  
         switch {  
         case sum > 0:  
            k--  
         case sum < 0:  
            j++  
         default:  
            res = append(res, []int{nums[i], nums[j], nums[k]})  
            for j < k && nums[k] == nums[k-1] {  
               k--  
            }  
            for j < k && nums[j] == nums[j+1] {  
               j++  
            }  
            j++  
            k--  
         }  
      }  
   }  
   return res  
}
```

#### 16. 3Sum Closest



```go
func threeSumClosest(nums []int, target int) int {  
   res := nums[0] + nums[1] + nums[2]  
   sort.Ints(nums)  
   for i := range nums {  
      if i > 0 && nums[i] == nums[i-1] {  
         continue  
      }  
      n := len(nums)  
      j, k := i+1, n-1  
      for j < k {  
         sum := nums[i] + nums[j] + nums[k]  
         if abs(sum-target) < abs(res-target) {  
            res = sum  
         }  
         switch {  
         case sum > target:  
            k--  
         case sum < target:  
            j++  
         default:  
            return target  
         }  
      }  
   }  
   return res  
}  
func abs(a int) int {  
   if a < 0 {  
      return -a  
   }  
   return a  
}
```

#### 42. Trapping Rain Water

java

```java
class Solution {
    public int trap(int[] height) {
        int prevLeft = 0, prevRight = 0, res = 0;
        for (int left = 0, right = height.length - 1; left <= right; ) {
            if (height[left] < height[right]) {
                res += Math.max(prevLeft - height[left], 0);
                prevLeft = Math.max(prevLeft, height[left]);
                left++;
            } else {
                res += Math.max(prevRight - height[right], 0);
                prevRight = Math.max(prevRight, height[right]);
                right--;
            }
        }
        return res;
    }
}
```

#### 58. Length of Last Word

go

```go
func lengthOfLastWord(s string) int {
   count, right := 0, len(s)-1
   for right >= 0 && s[right] == ' ' {
      right--
   }
   for right >= 0 && s[right] != ' ' {
      count++
      right--
   }
   return count
}
```

#### 125. Valid Palindrome

Go

```go
func isPalindrome(s string) bool {
	str := []byte(s)
	bt := bytes.ToLower(str)
	n := len(s)
	for left, right := 0, n-1; left < right; {
		if !isLetterOrNumber(bt[left]) {
			left++
			continue
		}
		if !isLetterOrNumber(bt[right]) {
			right--
			continue
		}
		if bt[left] == bt[right] {
			left++
			right--
			continue
		}
		return false
	}
	return true
}
func isLetterOrNumber(b byte) bool {
	if b >= 'a' && b <= 'z' || b <= '9' && b >= '0' {
		return true
	}
	return false
}
```


#### 151. Reverse Words in a String
O(1) space



O(n) space

rust

```rust
impl Solution {
    pub fn reverse_words(s: String) -> String {
        s.split_whitespace().rev().filter(|x| !x.is_empty()).collect::<Vec<&str>>().join(" ")
    }
}
```

java

```java
class Solution {
    public String reverseWords(String s) {
        List<String> res = new ArrayList<>();
        Arrays.stream(s.split(" ")).filter(a -> !a.isBlank()).forEach(res::add);
        Collections.reverse(res);
        return String.join(" ", res);
    }
}
```

go

```go
func reverseWords(s string) string {  
   words := strings.Split(s, " ")  
   var selected []string  
   for _, word := range words {  
      if len(word) != 0 {  
         selected = append(selected, word)  
      }  
   }  
   reverse(selected)  
   return strings.Join(selected, " ")  
}  
func reverse(words []string) {  
   left, right := 0, len(words)-1  
   for left < right {  
      words[left], words[right] = words[right], words[left]  
      left++  
      right--  
   }  
}
```

O(1) space

太精妙了https://leetcode.com/problems/reverse-words-in-a-string/solutions/47720/clean-java-two-pointers-solution-no-trim-no-split-no-stringbuilder/

```go
func reverseWords(s string) string {
	arr := []byte(s)
	n := len(s)
	reverseBetween(arr, 0, n-1)
	reverseWordByWord(arr, n)
	return cleanSpaces(arr, n)
}
func reverseBetween(arr []byte, start int, end int) {
	for start < end {
		arr[start], arr[end] = arr[end], arr[start]
		start++
		end--
	}
}
func reverseWordByWord(arr []byte, n int) {
	i, j := 0, 0
	for i < n {
		for i < j || i < n && arr[i] == ' ' {
			i++
		}
		for j < i || j < n && arr[j] != ' ' {
			j++
		}
		reverseBetween(arr, i, j-1)
	}
}
func cleanSpaces(arr []byte, n int) string {
	i, j := 0, 0
	for j < n {
		for j < n && arr[j] == ' ' {
			j++
		}
		for j < n && arr[j] != ' ' {
			arr[i] = arr[j]
			i++
			j++
		}
		for j < n && arr[j] == ' ' {
			j++
		}
		if j < n {
			arr[i] = ' '
			i++
		}
	}
	return string(arr[:i])
}
```


#### 167. Two Sum II - Input Array Is Sorted

```go
func twoSum(numbers []int, target int) []int {
   left, right := 0, len(numbers)-1
   for left < right {
      sum := numbers[left] + numbers[right]
      switch {
      case sum > target:
         right--
      case sum < target:
         left++
      default:
         return []int{left + 1, right + 1}
      }
   }
   return nil
}
```


#### 475. Heaters

java

```java
class Solution {
    public int findRadius(int[] houses, int[] heaters) {
        Arrays.sort(houses);
        Arrays.sort(heaters);
        int i = 0, res = 0;
        for (int house : houses) {
            while (i + 1 < heaters.length && heaters[i + 1] - house < house - heaters[i]) {
                i++;
            }
            res = Math.max(res, Math.abs(heaters[i] - house));
        }
        return res;
    }
}
```


#### 811. Subdomain Visit Count

java

```java
class Solution {  
    public List<String> subdomainVisits(String[] cpdomains) {  
        HashMap<String, Integer> map = new HashMap<>();  
        for (String cd : cpdomains) {  
            int i = cd.indexOf(" "), n = cd.length();  
            int count = Integer.parseInt(cd.substring(0, i));  
            for (i++; i < n; i++) {  
                String curr = cd.substring(i);  
                map.put(curr, map.getOrDefault(curr, 0) + count);  
                while (i < n && cd.charAt(i) != '.') i++;  
            }  
        }  
        return map.keySet().stream().map(key -> map.get(key) + " " + key).toList();  
    }  
}
```

#### 925. Long Pressed Name



```go
func isLongPressedName(name string, typed string) bool {
	m := len(name)
	i := 0
	for j := range typed {
		if i < m && name[i] == typed[j] {
			i++
		} else if j == 0 || typed[j] != typed[j-1] {
			return false
		}
	}
	return i == m
}
```