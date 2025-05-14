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
            val numerator = denominator / num1.second * num1.first + denominator / num2.second * num2.first
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

    fun generatePrimes(under: Int) : IntArray{
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
        for (i in 1 .. n) {
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

        fun compare(line: Double) : Int {
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
}