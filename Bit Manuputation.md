#### 50. Pow(x, n)

go

```go
func myPow(x float64, n int) float64 {
	if n < 0 {
		n = -n
		x = 1 / x
	}
	res := 1.0
	for ; n > 0; n >>= 1 {
		if n&1 != 0 {
			res *= x
		}
		x *= x
	}
	return res
}
```

#### 136. Single Number

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for n in nums:
            res ^= n
        return res
```

#### 137. Single Number II

```go
func singleNumber(nums []int) int {
   ans := int32(0)
   for i := 0; i < 32; i++ {
      counter := 0
      for _, num := range nums {
         counter += num >> i & 1
      }
      if counter%3 > 0 {
         ans |= 1 << i
      }
   }
   return int(ans)
}
```

#### 190. Reverse Bits

go

```go
func reverseBits(num uint32) uint32 {
	var res uint32 = 0
	for i := 0; i < 32; i++ {
		res = (res << 1) | (num >> i & 1)
	}
	return res
}
```

#### 191. Number of 1 Bits

一个无符号的数字，求其二进制位上有几个 1 

go

```go
func hammingWeight(num uint32) int {
	count := 0
	for ; num != 0; num &= num - 1 {
		count++
	}
	return count
}
```


#### 201. Bitwise AND of Numbers Range

go

```go
func rangeBitwiseAnd(left int, right int) int {
   res := 0
   for left != right {
      left >>= 1
      right >>= 1
      res++
   }
   return left << res
}
```

#### 231. Power of Two

```go
func isPowerOfTwo(n int) bool {
	return n > 0 && n&-n == n
	//return n > 0 && n&(n-1) == 0
}
```

#### [260. Single Number III](https://leetcode.com/problems/single-number-iii/)

```java
class Solution {  
    public int[] singleNumber(int[] nums) {  
        int xor = 0;  
        for (int num : nums) {  
            xor ^= num;  
        }  
        int diff = xor & -xor;  
        int[] res = {0, 0};  
        for (int num : nums) {  
            if ((num & diff) == 0) {  
                res[0] ^= num;  
            } else {  
                res[1] ^= num;  
            }  
        }  
        return res;  
    }  
}
```
#### 268. Missing Number

array containing `n` distinct numbers in the range `[0, n]` , find the missing one.

```go
func missingNumber(nums []int) int {
	x, n := 0, len(nums)
	for i, num := range nums {
		x ^= i ^ num
	}
	return x ^ n
}
```


#### 289. Game of Life

```python
# python
class Solution:  
    def gameOfLife(self, board: List[List[int]]) -> None:  
        """  
        Do not return anything, modify board in-place instead.        """        
        m, n = len(board), len(board[0])  
        for i in range(m):  
            for j in range(n):  
                count = 0  
                # expand to check the lives  
                for x in range(max(i - 1, 0), min(i + 2, m)):  
                    for y in range(max(j - 1, 0), min(j + 2, n)):  
                        count += board[x][y] & 1  
                # Any dead cell with exactly three live neighbors becomes a live cell  
                if count == 3 or count - board[i][j] == 3:  
                    # use the 2-bit to store the new state  
                    board[i][j] |= 2  
        for i in range(m):  
            for j in range(n):  
                board[i][j] >>= 1
```

```go
func gameOfLife(board [][]int) {  
   m, n := len(board), len(board[0])  
   for i := range board {  
      for j := range board[0] {  
         count := 0  
         for x := max(i-1, 0); x < min(i+2, m); x++ {  
            for y := max(j-1, 0); y < min(j+2, n); y++ {  
               count += board[x][y] & 1  
            }  
         }  
         if count == 3 || count-board[i][j] == 3 {  
            board[i][j] |= 2  
         }  
      }  
   }  
   for i := range board {  
      for j := range board[0] {  
         board[i][j] >>= 1  
      }  
   }  
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


#### 338. Counting Bits

```go
func countBits(n int) []int {
	ans := make([]int, n+1)
	for i := 1; i < n+1; i++ {
		ans[i] = ans[i&(i-1)] + 1
	}
	return ans
}
```

#### 389. Find the Difference

Go

```go
func findTheDifference(s string, t string) byte {  
   char := t[len(t)-1]  
   for i := range s {  
      char ^= s[i]  
      char ^= t[i]  
   }  
   return char  
}
```

#### 342. Power of Four

```go
func isPowerOfFour(n int) bool {  
   return n > 0 && n&(n-1) == 0 && (n-1)%3 == 0  
}
```

#### 371. Sum of Two Integers

go

```go
func getSum(a int, b int) int {  
   for b != 0 {  
      a, b = a^b, a&b<<1  
   }  
   return a  
}
```
