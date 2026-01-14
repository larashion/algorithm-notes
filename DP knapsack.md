如果是01背包，内层循环是倒序，外层循环物品

完全背包中物品不止使用一次，内层循环变为正序

如果考虑物品的进入顺序，内层循环物品



#### 416. Partition Equal Subset Sum

Input: [1, 5, 11, 5]

Output: true

Explanation: The array can be partitioned as [1, 5, 5] and [11].

01背包

Go

```go
func canPartition(nums []int) bool {
	sum := 0
	for _, v := range nums {
		sum += v
	}
	if sum&1 == 1 {
		return false
	}
	target := sum / 2
	dp := make([]bool, target+1)
	dp[0] = true
	for _, num := range nums {
		for j := target; j >= num; j-- {
			dp[j] = dp[j] || dp[j-num]
		}
	}
	return dp[target]
}
```



#### 474. Ones and Zeroes

01背包

```go
func findMaxForm(strs []string, m int, n int) int {
   dp := make([][]int, m+1)
   for i := range dp {
      dp[i] = make([]int, n+1)
   }
   for _, str := range strs {
      zero := 0
      one := 0
      for s := range str {
         if str[s] == '1' {
            one++
         } else {
            zero++
         }
      }
      for i := m; i >= zero; i-- {
         for j := n; j >= one; j-- {
            dp[i][j] = max(dp[i][j], dp[i-zero][j-one]+1)
         }
      }
   }
   return dp[m][n]
}
```

#### 494. Target Sum

01背包

```go
func findTargetSumWays(nums []int, target int) int {
    sum := 0
    for _, n := range nums {
        sum += n
    }
    // p - n = target
    // p + n = sum
    // p = (sum + target)/2
    if sum < target || sum&1 != target&1 {
        return 0
    }
    return subset(nums, (sum+target)/2)
}
func subset(nums []int, target int) int {
    if target < 0 {
        return 0
    }
    dp := make([]int, target+1)
    dp[0] = 1
    for _, num := range nums {
        for i := target; i >= num; i-- {
            dp[i] += dp[i-num]
        }
    }
    return dp[target]
}
```



#### 322. Coin Change

完全背包

1D dp

```go
func coinChange(coins []int, amount int) int {
   n := amount
   dp := make([]int, n+1)
   for i := 1; i < len(dp); i++ {
      dp[i] = math.MaxInt - 1
   }
   for _, coin := range coins {
      for j := coin; j < n+1; j++ {
         dp[j] = min(dp[j], 1+dp[j-coin])
      }
   }
   if dp[n] == math.MaxInt-1 {
      dp[n] = -1
   }
   return dp[n]
}
```

#### 518. Coin Change 2

完全背包

1D dp

```go
func change(amount int, coins []int) int {
   dp := make([]int, amount+1)
   dp[0] = 1
   for _, coin := range coins {
      for i := coin; i < amount+1; i++ {
         dp[i] += dp[i-coin]
      }
   }
   return dp[amount]
}
```



#### 279. Perfect Squares

https://leetcode.com/problems/perfect-squares/discuss/71512/Static-DP-C%2B%2B-12-ms-Python-172-ms-Ruby-384-ms

go

```go
func numSquares(n int) int {
	dp := make([]int, n+1)
	for i := 1; i < n+1; i++ {
		dp[i] = math.MaxInt32
		for j := 1; j*j <= i; j++ {
			dp[i] = min(dp[i], dp[i-j*j]+1)
		}
	}
	return dp[n]
}
```



#### 871. Minimum Number of Refueling Stops

go

```go
func minRefuelStops(target int, startFuel int, stations [][]int) int {
	n := len(stations)
	dp := make([]int, n+1)
	dp[0] = startFuel
	for j := range stations {
		// refueling times
		for i := j; i >= 0 && dp[i] >= stations[j][0]; i-- {
			dp[i+1] = max(dp[i+1], dp[i]+stations[j][1])
		}
	}
	for i := range dp {
		if dp[i] >= target {
			return i
		}
	}
	return -1
}
```

#### 1049. Last Stone Weight II

```go
func lastStoneWeightII(stones []int) int {
   sum := 0
   for _, v := range stones {
      sum += v
   }
   target := sum / 2
   dp := make([]int, target+1)
   for i := range stones {
      for j := target; j >= stones[i]; j-- {
         dp[j] = max(dp[j], dp[j-stones[i]]+stones[i])
      }
   }
   return sum - 2*dp[target]
}
```