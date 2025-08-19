package com.example.codetools

import java.util.LinkedList
import java.util.PriorityQueue
import kotlin.math.abs
import kotlin.math.sqrt

object MathCode {
    private val MODULO = 1_000_000_007

    fun isPalindrome(x: Int): Boolean {
        if (x < 0) return false
//        val s = x.toString()
//        var left = 0
//        var right = s.lastIndex
//        while (left < right) {
//            if (s[left] != s[right]) {
//                return false
//            }
//            left++
//            right--
//        }
//        return true

//        var original = x
//        var reversed = 0
//        while (original != 0) {
//            val remainder = original % 10
//            reversed = reversed * 10 + remainder
//            original /= 10
//        }
//        return x == reversed

        var scale = 10
        while (x / scale >= 10) {
            scale *= 10
        }
        var current = x
        while (current > 0) {
            if (current / scale != current % 10) return false
            current = (current % scale) / 10
            scale /= 100
        }
        return true
    }

    fun romanToInt(s: String): Int {
        val romanMap = HashMap<Char, Int>()
        romanMap['I'] = 1
        romanMap['V'] = 5
        romanMap['X'] = 10
        romanMap['L'] = 50
        romanMap['C'] = 100
        romanMap['D'] = 500
        romanMap['M'] = 1000
        var result = 0;
        for (i in s.indices) {
            if (i + 1 < s.length) {
                if (romanMap[s[i]]!! < romanMap[s[i + 1]]!!) {
                    result -= romanMap[s[i]]!!
                } else {
                    result += romanMap[s[i]]!!
                }
            } else {
                result += romanMap[s[i]]!!
            }
        }
        return result

//        var toCheck = s;
//        var result = 0;
//        while (toCheck.isNotEmpty()) {
//            var symbol = ""
//            if (toCheck.length >= 2) {
//                symbol = toCheck.take(2)
//                if (romanMap.containsKey(symbol)) {
//                    result += romanMap[symbol]!!
//                    toCheck = toCheck.substring(2)
//                    continue
//                }
//            }
//            symbol = toCheck.take(1)
//            if (romanMap.containsKey(symbol)) {
//                result += romanMap[symbol]!!
//                toCheck = toCheck.substring(1)
//            }
//        }

//        var result = 0;
//        var i = 0
//        var s1 = ' '
//        var s2 = ' '
//        while (i < s.length) {
//            s1 = s[i]
//            i++
//            when (s1) {
//                'I' -> {
//                    if (i < s.length) {
//                        s2 = s[i]
//                        when (s2) {
//                            'V' -> {
//                                result += 4
//                                i++
//                            }
//                            'X' -> {
//                                result += 9
//                                i++
//                            }
//                            else -> {
//                                result += 1
//                            }
//                        }
//                    } else {
//                        result += 1
//                    }
//                }
//                'X' -> {
//                    if (i < s.length) {
//                        s2 = s[i]
//                        when (s2) {
//                            'L' -> {
//                                result += 40
//                                i++
//                            }
//                            'C' -> {
//                                result += 90
//                                i++
//                            }
//                            else -> {
//                                result += 10
//                            }
//                        }
//                    } else {
//                        result += 10
//                    }
//
//                }
//                'C' -> {
//                    if (i < s.length) {
//                        s2 = s[i]
//                        when (s2) {
//                            'D' -> {
//                                result += 400
//                                i++
//                            }
//                            'M' -> {
//                                result += 900
//                                i++
//                            }
//                            else -> {
//                                result += 100
//                            }
//                        }
//                    } else {
//                        result += 100
//                    }
//                }
//                'V' -> {
//                    result += 5
//                }
//                'L' -> {
//                    result += 50
//                }
//                'D' -> {
//                    result += 500
//                }
//                'M' -> {
//                    result += 1000
//                }
//            }
//        }
//
//        return result;
    }

    fun intToRoman(num: Int): String {
        val values = intArrayOf(1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
        val symbols = arrayOf("M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I")

        var i = 0
        var current = num
        val result = StringBuilder()
        while (current > 0) {
            while (current >= values[i]) {
                current -= values[i]
                result.append(symbols[i])
            }
            i++
        }
        return result.toString()
    }

    fun mySqrt(x: Int): Int {
        if (x == 0 || x == 1) return x
        var left = 0
        var right = x
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (mid == x / mid) return mid
            else if (mid > x / mid) right = mid - 1
            else left = mid + 1
        }
        return right
    }

    fun climbStairs(n: Int): Int {
        if (n == 0 || n == 1) return 1
        var prev = 1
        var curr = 1
        var tmp = 1
        for (i in 1 until n) {
            tmp = curr
            curr += prev
            prev = tmp
        }
        return curr
//        if (n == 1) return 1
//        if (n == 2) return 2
//        val dp = IntArray(n + 1)
//        dp[1] = 1
//        dp[2] = 2
//        for (i in 3..n) {
//            dp[i] = dp[i - 1] + dp[i - 2]
//        }
//        return dp[n]
    }

    fun pivotInteger(n: Int): Int {
        val totalSum = sumOf(1, n)
//        for (i in 1..n) {
//            val left = sumOf(1, i)
//            val right = totalSum - left + i
//            if (left == right) return i
//        }
        var start = 0
        var end = n
        while (start <= end) {
            val mid = start + (end - start) / 2
            val left = sumOf(1, mid)
            val right = totalSum - left + mid
            if (left == right) return mid
            if (left < right) start = mid + 1
            if (left > right) end = mid - 1
        }

        return -1
    }

    fun sumOf(start: Int, end: Int): Int {
        var sum = 0
        for (i in start..end) {
            sum += i
        }
        return sum
    }

    fun sumBase(n: Int, k: Int): Int {
        var sum = 0
        var curr = n
        while (curr != 0) {
            sum += curr % k
            curr /= k
        }
        return sum

//        return if (n == 0) {
//            0
//        } else {
//            n % k + sumBase(n / k, k)
//        }
    }

    fun arrangeCoins(n: Int): Int {
//        var sum = 0
//        var i = 1
//        while (i <= n) {
//            when {
//                sum == n - i -> return i
//                sum > n - i -> return i - 1
//                else -> {
//                    sum += i
//                    i++
//                }
//            }
//        }
//        return i

        var start = 1
        var end = n
        var result = 0
        while (start <= end) {
            val mid = start + (end - start) / 2
            val currentSum = mid.toLong() * (mid + 1) / 2
            if (currentSum <= n) {
                result = mid
                start = mid + 1
                if (currentSum + mid + 1 > n) {
                    break
                }
            } else {
                end = mid - 1
            }
        }
        return result
    }

    fun findSum(i: Int): Long {
        // Calculate the sum of consecutive positive integers from 1 to i
        return i.toLong() * (i + 1) / 2
    }

    fun tribonacci(n: Int): Int {
        //0 <= n <= 37
        val dp = IntArray(38) { 0 }
        dp[1] = 1
        dp[2] = 1
        for (i in 3..n) {
            dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
        }
        return dp[n]
    }

    fun myPow(x: Double, n: Int): Double {
        if (x == 0.0) return 0.0
        if (x == 1.0) return 1.0
        if (x == -1.0) return if (n % 2 == 0) 1.0 else -1.0
        if (n == 0) return 1.0
        var tempX = x
        var tempN = if (n < 0) {
            n * -1L
        } else {
            n * 1L
        }
        var result = 1.0
        while (tempN > 0) {
            if (tempN % 2.0 == 0.0) {
                tempX *= tempX
                tempN /= 2
            } else {
                result *= tempX
                tempN--
            }
        }
        if (n < 0) result = 1.0 / result
        return result
    }

    fun isHappy(n: Int): Boolean {
        val squares = intArrayOf(0, 1, 4, 9, 16, 25, 36, 49, 64, 81)
        val computedMap = mutableSetOf<Int>()
        var current = n
        while (current != 1) {
            if (computedMap.contains(current)) return false
            var temp = current
            var sum = 0
            while (temp > 0) {
                sum += squares[temp % 10]
                temp = temp / 10
            }
            computedMap.add(current)
            current = sum
        }
        return true
    }

    fun judgeSquareSum(c: Int): Boolean {
        var start = 0L
        var end = sqrt(c.toDouble()).toLong()
        var current = 0L
        while (start <= end) {
            current = start * start + end * end
            when {
                current == c.toLong() -> return true
                current < c.toLong() -> start++
                current > c.toLong() -> end--
            }
        }
        return false
    }

    fun numWaterBottles(numBottles: Int, numExchange: Int): Int {
        var result = numBottles
        var empty = numBottles
        while (empty >= numExchange) {
            empty = empty % numExchange + empty / numExchange
            result += empty / numExchange
        }
        return result
    }

    fun trailingZeroes(n: Int): Int {
        // of n!
        var num = n
        var result = 0
        while (num > 0) {
            num /= 5
            result += num
        }
        return result
    }

    fun fractionAddition(expression: String): String {
        // -1/2+1/2+1/3 , number from [1,10]
        var start = 0
        var end = 0

        fun parse(e: String): Pair<Int, Int> {
            if (e.isEmpty()) return Pair(0, 1)
            val slashIndex = e.indexOf('/')
            return Pair(e.substring(0, slashIndex).toInt(), e.substring(slashIndex + 1).toInt())
        }

        fun gcd(a: Int, b: Int): Int {
            return if (b == 0) a else gcd(b, a % b)
        }

        fun lcm(a: Int, b: Int): Int {
            return (a.toLong() * b / gcd(a, b)).toInt()
        }

        fun add(num1: Pair<Int, Int>, num2: Pair<Int, Int>): Pair<Int, Int> {
            val denominator = lcm(num1.second, num2.second)
            val numerator =
                denominator / num1.second * num1.first + denominator / num2.second * num2.first
            val gcd = abs(gcd(numerator, denominator))
            return Pair(numerator / gcd, denominator / gcd)
        }

        var result = Pair(0, 1)
        while (end in expression.indices) {
            if (expression[end] == '+' || expression[end] == '-') {
                result = add(result, parse(expression.substring(start, end)))
                start = end
            }
            end++
        }
        result = add(result, parse(expression.substring(start)))
        return result.first.toString() + "/" + result.second
    }

    fun countEven(num: Int): Int {
        var curr = num
        var sum = 0
        while (curr > 0) {
            sum += curr % 10
            curr /= 10
        }
        return if (sum % 2 == 0) {
            num / 2
        } else {
            (num - 1) / 2
        }
    }

    fun minimumSum(num: Int): Int {
        val pq = PriorityQueue<Int>()
        var num = num
        while (num > 0) {
            pq.offer(num % 10)
            num /= 10
        }
        var sum = pq.poll()!! * 10
        sum += pq.poll()!! * 10
        sum += pq.poll()!!
        sum += pq.poll()!!
        return sum
    }

    fun generatePrimes(under: Int): IntArray {
        val isPrime = BooleanArray(under) { true }
        isPrime[0] = false
        isPrime[1] = false
        for (i in 2 until under) {
            if (isPrime[i]) {
                for (j in i * 2 until under step i) { // or start from i * i
                    isPrime[j] = false
                }
            }
        }
        return isPrime.indices.filter { isPrime[it] }.toIntArray()
    }

    fun reverse(x: Int): Int {
        var current = if (x < 0) -x else x
        var ans = 0
        while (current != 0) {
            if (ans > (Int.MAX_VALUE - current % 10) / 10) {
                return 0
            }
            ans = ans * 10 + current % 10
            current /= 10
        }
        return if (x < 0) -ans else ans
    }

    fun punishmentNumber(n: Int): Int {

//        fun canPartition(num: Int, target: Int): Boolean {
//            if (target < 0 || num < target) {
//                return false
//            }
//
//            return if (num == target) {
//                true
//            } else canPartition(num / 10, target - num % 10) ||
//                    canPartition(num / 100, target - num % 100) ||
//                    canPartition(num / 1000, target - num % 1000)
//        }

        fun test(num: Int): Int {
            val square = num * num
            val queue = LinkedList<Pair<Int, Int>>()
            queue.offer(0 to square)
            while (queue.isNotEmpty()) {
                val (sum, rest) = queue.poll()!!
                var multi = 1
                while (rest / multi != 0) {
                    val newSum = sum + rest / multi
                    val newRest = rest % multi
                    if (newRest == 0) {
                        if (newSum == num) return square
                    } else {
                        queue.offer(newSum to newRest)
                    }
                    multi *= 10
                }
            }
            return 0
        }

        var result = 0
        for (i in 1..n) {
            result += test(i)
        }
        return result
    }

    fun separateSquares(squares: Array<IntArray>): Double {
        var start = Double.MAX_VALUE
        var end = Double.MIN_VALUE
        for ((x, y, l) in squares) {
            start = minOf(start, y * 1.0)
            end = maxOf(end, y * 1.0 + l)
        }

        fun compare(line: Double): Int {
            var s1 = 0.0
            var s2 = 0.0
            for ((x, y, l) in squares) {
                if (y + l <= line) {
                    s1 += 1.0 * l * l
                } else if (y >= line) {
                    s2 += 1.0 * l * l
                } else {
                    s1 += 1.0 * (line - y) * l
                    s2 += 1.0 * (y + l - line) * l
                }
            }
            return s1.compareTo(s2)
        }

        while (end - start > 0.00001) {
            val mid = start + (end - start) / 2
            if (compare(mid) >= 0) {
                end = mid
            } else {
                start = mid
            }
        }
        return end
    }

    fun modPow(base: Long, exponent: Long): Long {
        var result = 1L
        var b = base % MODULO
        var exp = exponent

        while (exp > 0) {
            if (exp and 1L == 1L) {
                result = (result * b) % MODULO
            }
            b = (b * b) % MODULO
            exp = exp shr 1
        }
        return result
    }


    fun minMaxDifference(num: Int): Int {
        var multiplier = 1
        var current = num / 10
        while (current != 0) {
            current /= 10
            multiplier *= 10
        }
        current = num
        var toMax = -1
        var toMin = -1
        var result = 0
        while (multiplier != 0) {
            val digit = current / multiplier
            var delta = 0
            if (digit != toMax && toMax != -1) {
                delta = digit
            } else {
                delta = 9
                if (toMax == -1 && digit != 9) {
                    toMax = digit
                }
            }
            if (digit != toMin && toMin != -1) {
                delta -= digit
            } else if (toMin == -1 && digit != 0) {
                toMin = digit
            }
            result += delta * multiplier
            current %= multiplier
            multiplier /= 10
        }
        return result
    }

    fun kMirrorSum(k: Int, n: Int): Long {
        fun isBaseKMirror(num: Int): Boolean {
            val kBase = num.toString(k)
            var right = kBase.length - 1
            var left = 0
            while (left < right) {
                if (kBase[left++] != kBase[right--]) return false
            }
            return true
        }

        var result = 0L
        var count = 0
        for (num in 1..9) {
            if (isBaseKMirror(num)) {
                result += num
                if (++count == n) return result
            }
        }
        for (length in 2 until 32) {
            val seedLength = (length + 1) / 2
            val start = Math.pow(10.0, (seedLength - 1).toDouble()).toInt()
            val end = Math.pow(10.0, seedLength.toDouble()).toInt() - 1
            for (part in start..end) {
                val s = part.toString()
                var suffix = s.reversed()
                if (length % 2 != 0) {
                    suffix = suffix.substring(1)
                }
                val num = (s + suffix).toInt()
                if (isBaseKMirror(num)) {
                    result += num
                    if (++count == n) return result
                }
            }
        }
        return result
    }

    fun numSubseq(nums: IntArray, target: Int): Int {
        nums.sort()
        val n = nums.size
        var result = 0
        var left = 0
        var right = n - 1

        // Precompute powers of 2 for efficiency
        val pow2 = IntArray(n) { 1 }
        for (i in 1 until n) {
            pow2[i] = (pow2[i - 1] * 2) % MODULO
        }

        while (left <= right) {
            if (nums[left] + nums[right] <= target) {
                result = (result + pow2[right - left]) % MODULO
                left++ // Try a larger minimum
            } else {
                right-- // Try a smaller maximum
            }
        }

        return result
    }

    fun isPowerOfTwo(n: Int): Boolean {
        return n > 0 && (n and (n - 1) == 0)
    }

    fun reorderedPowerOf2(n: Int): Boolean {
        fun isPowerOf2(num: Int): Boolean {
            return num and (num - 1) == 0
        }

        val freq = IntArray(10)
        var current = n
        while (current != 0) {
            val digit = current % 10
            freq[digit]++
            current /= 10
        }

        fun dfs(num: Int): Boolean {
            if (freq.sum() == 0) {
                return isPowerOf2(num)
            }
            var found = false
            for (digit in 0..9) {
                if (num == 0 && digit == 0) continue
                if (freq[digit] != 0) {
                    freq[digit]--
                    found = found || dfs(num * 10 + digit)
                    freq[digit]++
                }
            }
            return found
        }

        return dfs(0)
    }

    fun isPowerOfThree(n: Int): Boolean {
        // 3^19 is the largest power of 3 that fits in an int
        return n > 0 && 1162261467 % n == 0
    }

    fun isPowerOfFour(n: Int): Boolean {
        // n will have only one 1 bit, and n - 1 is divisible by 3, because 4^n = (3 + 1)^n
        return n > 0 && (n and (n - 1)) == 0 && (n - 1) % 3 == 0
//        if (n <= 0) return false
//        var current = n
//        var index = -1
//        while (current > 0) {
//            index++
//            if (current and 1 == 1 && current != 1) return false
//            current = current shr 1
//        }
//        return index % 2 == 0
    }

    fun productQueries(n: Int, queries: Array<IntArray>): IntArray {
        var index = 0
        var current = n
        var lastSum = 0
        val prefixSum = mutableListOf(0)
        while (current != 0) {
            if (current and 1 == 1) {
                prefixSum.add(lastSum + index)
                lastSum += index
            }
            current = current shr 1
            index++
        }

        val cache = Array(31) { IntArray(31) }

        return IntArray(queries.size) { i ->
            val left = queries[i][0]
            val right = queries[i][1]
            if (cache[left][right] != 0) {
                return@IntArray cache[left][right]
            } else {
                var power = prefixSum[queries[i][1] + 1] - prefixSum[queries[i][0]]
                var result = 1L
                var base = 2L
                while (power > 0) {
                    if (power and 1 == 1) {
                        result = (result * base) % MODULO
                    }
                    base = (base * base) % MODULO
                    power = power shr 1
                }
                cache[left][right] = result.toInt()
                return@IntArray cache[left][right]
            }
        }
    }


    fun numberOfWays(n: Int, x: Int): Int {
        // ways-to-express-an-integer-as-sum-of-powers
        val dp = IntArray(n + 1)
        dp[0] = 1
        for (i in 1..n) {
            var num = i
            for (times in 0 until x - 1) {
                if (num > n) break
                num *= i
            }
            for (j in n downTo num) {
                dp[j] = (dp[j] + dp[j - num]) % MODULO
            }
        }
        return dp[n]
    }

    fun new21Game(n: Int, k: Int, maxPts: Int): Double {
        if (k == 0 || n >= k + maxPts) {
            return 1.0
        }

        val dp = DoubleArray(k + maxPts)
        var windowSum = 0.0
        for (i in k until k + maxPts) {
            dp[i] = if (i <= n) 1.0 else 0.0
            windowSum += dp[i]
        }

        for (i in k - 1 downTo 0) {
            dp[i] = windowSum / maxPts
            windowSum = windowSum - dp[i + maxPts] + dp[i]
        }
        return dp[0]
    }

    fun judgePoint24(cards: IntArray): Boolean {
        val array = doubleArrayOf(
            cards[0].toDouble(),
            cards[1].toDouble(),
            cards[2].toDouble(),
            cards[3].toDouble()
        )

        fun compute(x: Double, y: Double): DoubleArray {
            return doubleArrayOf(x + y, x - y, y - x, x * y, x / y, y / x)
        }

        fun dfs(nums: DoubleArray): Boolean {
            if (nums.size == 1) return abs(nums[0] - 24) < 0.0001

            for (i in nums.indices) {
                for (j in i + 1 until nums.size) {
                    val next = DoubleArray(nums.size - 1)
                    var index = 0
                    for (k in nums.indices) {
                        if (k != i && k != j) next[index++] = nums[k]
                    }
                    for (d in compute(nums[i], nums[j])) {
                        next[next.size - 1] = d
                        if (dfs(next)) return true
                    }
                }
            }
            return false
        }

        return dfs(array)
    }

    fun zeroFilledSubarray(nums: IntArray): Long {
        var count = 0L
        var streak = 0
        for (i in nums.indices) {
            if (nums[i] != 0) {
                streak = 0
            } else {
                count += ++streak
            }
        }
        return count
    }

    fun numberOfArithmeticSlices(nums: IntArray): Int {
        var count = 0
        var streak = 0
        for (i in 2 until nums.size) {
            if (nums[i] - nums[i - 1] != nums[i - 1] - nums[i - 2]) {
                streak = 0
            } else {
                count += ++streak
            }
        }
        return count
    }

    fun checkArithmeticSubarrays(nums: IntArray, l: IntArray, r: IntArray): List<Boolean> {
        val result = BooleanArray(l.size)

        fun isValid(subArray: IntArray): Boolean {
            if (subArray.size <= 1) return false
            if (subArray.size == 2) return true
            subArray.sort()
            for (i in 2 until subArray.size) {
                if (subArray[i] - subArray[i - 1] != subArray[i - 1] - subArray[i - 2]) {
                    return false
                }
            }
            return true
        }

        for (i in l.indices) {
            result[i] = isValid(nums.copyOfRange(l[i], r[i] + 1))
        }

        return result.toList()
    }
}