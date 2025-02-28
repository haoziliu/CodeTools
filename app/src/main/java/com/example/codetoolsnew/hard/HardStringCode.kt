package com.example.codetools.hard

import java.util.LinkedList
import kotlin.math.abs
import kotlin.math.pow

object HardStringCode {

    fun findRotateSteps(ring: String, key: String): Int {
        //abcdefgxxxxxxgcbxx
        //bcga

        val maxIndex = ring.length - 1
        val matrix = Array(key.length) { IntArray(ring.length) { Int.MAX_VALUE } }
        var pL = 0
        var pR = 0
        var leftMoved = 0
        var rightMoved = 0

        val queue = ArrayDeque<Int>()
        queue.add(0)
        var queueSize = 0
        key.forEachIndexed { index, char ->
            queueSize = queue.size

            for (i in 0 until queueSize) {
                val start = queue.removeFirst()
                pL = start
                pR = start

                leftMoved = if (index == 0 || matrix[index - 1][start] == Int.MAX_VALUE) {
                    0
                } else {
                    matrix[index - 1][start]
                }
                rightMoved = leftMoved

                while (ring[pL] != char || ring[pR] != char) {
                    if (ring[pL] != char) {
                        pL--
                        leftMoved++
                        if (pL == -1) {
                            pL = maxIndex
                        }
                    }
                    if (ring[pR] != char) {
                        pR++
                        rightMoved++
                        if (pR == maxIndex + 1) {
                            pR = 0
                        }
                    }
                }
                if (pL != pR) {
                    matrix[index][pL] = minOf(leftMoved + 1, matrix[index][pL])
                    if (!queue.contains(pL)) {
                        queue.addLast(pL)
                    }
                    matrix[index][pR] = minOf(rightMoved + 1, matrix[index][pR])
                    if (!queue.contains(pR)) {
                        queue.addLast(pR)
                    }
                } else {
                    matrix[index][pL] = minOf(leftMoved + 1, rightMoved + 1, matrix[index][pL])
                    if (!queue.contains(pL)) {
                        queue.addLast(pL)
                    }
                }
            }
        }

        return matrix[key.length - 1].minOf { it }
    }

    fun checkRecord(n: Int): Int {
        // 'A' < 2; 'L' consecutive < 3; 'P'

        // 6 states: (hasA(0,1), consecutiveDays[0, 1, 2])

        // state can go to:
        // 000 -> 000, 100, 001
        // 001 -> 000, 100, 010
        // 010 -> 000, 100
        // 100 -> 100,    , 101
        // 101 -> 100,    , 110
        // 110 -> 100,

        // state can be reached by:
        // 000 <- 000, 001, 010
        // 001 <- 000
        // 010 <- 001
        // 100 <- 000, 001, 010, 100, 101, 110
        // 101 <- 100
        // 110 <- 101

        val MOD = 1000000007
        val dp = Array(n + 1) { IntArray(7) { 0 } }
        dp[0][0] = 1
        var result = 0

        for (i in 1..n) {
            dp[i][0] = ((dp[i - 1][0] + dp[i - 1][1]) % MOD + dp[i - 1][2]) % MOD
            dp[i][1] = dp[i - 1][0]
            dp[i][2] = dp[i - 1][1]
            dp[i][4] = (((((dp[i - 1][0] + dp[i - 1][1]) % MOD +
                    dp[i - 1][2]) % MOD +
                    dp[i - 1][4]) % MOD +
                    dp[i - 1][5]) % MOD +
                    dp[i - 1][6]) % MOD
            dp[i][5] = dp[i - 1][4]
            dp[i][6] = dp[i - 1][5]
        }
        dp[n].forEach {
            result = (result + it) % MOD
        }
        return result
    }

//    fun countOfAtoms(formula: String): String {
//        fun analyse(start: Int, end: Int): Map<String, Int> {
//            val atomCount = mutableMapOf<String, Int>()
//            var leftIndex = -1
//            var leftCount = 0
//            var i = start
//
//            while (i < end) {
//                if (leftCount == 0) {
//                    if (formula[i] == '(') {
//                        leftIndex = i
//                        leftCount++
//                        i++
//                    } else if (formula[i] in 'A'..'Z') {
//                        val nameSB = StringBuilder().append(formula[i++])
//                        while (i < end && formula[i] in 'a'..'z') {
//                            nameSB.append(formula[i++])
//                        }
//                        val count: Int = if (i < end && formula[i].isDigit()) {
//                            val countSB = StringBuilder().append(formula[i++])
//                            while (i < end && formula[i].isDigit()) {
//                                countSB.append(formula[i++])
//                            }
//                            countSB.toString().toInt()
//                        } else {
//                            1
//                        }
//                        nameSB.toString().let { name ->
//                            atomCount[name] = atomCount.getOrDefault(name, 0) + count
//                        }
//                    }
//                } else if (formula[i] == '(') {
//                    leftCount++
//                    i++
//                } else if (formula[i] == ')') {
//                    if (leftCount != 1) {
//                        leftCount--
//                        i++
//                        continue
//                    }
//                    val map = analyse(leftIndex + 1, i)
//                    leftCount--
//                    i++
//                    val count: Int = if (i < end && formula[i].isDigit()) {
//                        val countSB = StringBuilder().append(formula[i++])
//                        while (i < end && formula[i].isDigit()) {
//                            countSB.append(formula[i++])
//                        }
//                        countSB.toString().toInt()
//                    } else {
//                        1
//                    }
//                    map.forEach { (key, value) ->
//                        atomCount[key] = atomCount.getOrDefault(key, 0) + value * count
//                    }
//                } else {
//                    i++
//                }
//            }
//            return atomCount
//        }
//
//        val result = StringBuilder()
//        analyse(0, formula.length).toSortedMap().forEach { (key, value) ->
//            result.append(key)
//            if (value > 1) {
//                result.append(value)
//            }
//        }
//        return result.toString()
//    }

    fun countOfAtoms(formula: String): String {
        val stack = LinkedList<MutableMap<String, Int>>()
        stack.push(mutableMapOf())
        var i = 0
        val n = formula.length

        fun parseAtom(): String {
            val sb = StringBuilder()
            sb.append(formula[i++])
            while (i < n && formula[i].isLowerCase()) {
                sb.append(formula[i++])
            }
            return sb.toString()
        }

        fun parseCount(): Int {
            if (i == n || !formula[i].isDigit()) return 1
            val sb = StringBuilder()
            while (i < n && formula[i].isDigit()) {
                sb.append(formula[i++])
            }
            return sb.toString().toInt()
        }

        while (i < n) {
            when (formula[i]) {
                '(' -> {
                    stack.push(mutableMapOf())
                    i++
                }

                ')' -> {
                    i++
                    val count = parseCount()
                    val top = stack.pop()
                    val current = stack.peek()!!
                    for ((atom, num) in top) {
                        current[atom] = current.getOrDefault(atom, 0) + num * count
                    }
                }

                else -> {
                    val name = parseAtom()
                    val count = parseCount()
                    val current = stack.peek()!!
                    current[name] = current.getOrDefault(name, 0) + count
                }
            }
        }

        val result = StringBuilder()
        stack.peek()?.toSortedMap()?.forEach { (key, value) ->
            result.append(key)
            if (value > 1) {
                result.append(value)
            }
        }
        return result.toString()
    }

    fun numberToWords(num: Int): String {
        // 12345 -> Twelve Thousand Three Hundred Forty Five
        val LESS_THAN_20 = arrayOf(
            "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
            "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
            "Seventeen", "Eighteen", "Nineteen"
        )
        val TENS = arrayOf(
            "", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty",
            "Seventy", "Eighty", "Ninety"
        )
        val THOUSANDS = arrayOf("", "Thousand", "Million", "Billion")

        fun helper(num: Int): String {
            return when {
                num == 0 -> ""
                num < 20 -> LESS_THAN_20[num] + " "
                num < 100 -> TENS[num / 10] + " " + helper(num % 10)
                else -> LESS_THAN_20[num / 100] + " Hundred " + helper(num % 100)
            }
        }

        var current = num
        var i = 0
        var words = ""
        while (current > 0) {
            if (current % 1000 != 0) {
                words = helper(current % 1000) + THOUSANDS[i] + " " + words
            }
            current /= 1000
            i++
        }
        return words.trim()
    }

    fun strangePrinterIterative(s: String): Int {
        val n = s.length
        if (n == 0) return 0
        // dp[i][j] 表示将子串s[i:j]打印出来所需的最少步骤数
        val dp = Array(n) { IntArray(n) }
        // dp[i][i] 表示打印单个字符需要1次
        for (i in 0 until n) {
            dp[i][i] = 1
        }

        // 逐步增加子串长度
        for (length in 2..n) {
            for (i in 0..n - length) {
                val j = i + length - 1
                dp[i][j] = dp[i][j - 1] + 1 // 初始情况：s[i:j] 作为独立子串

                // 尝试分割点 k，并优化步骤数
                for (k in i until j) {
                    if (s[k] == s[j]) {
                        dp[i][j] = minOf(dp[i][j], dp[i][k] + dp[k + 1][j - 1])
                    } else {
                        dp[i][j] = minOf(dp[i][j], dp[i][k] + dp[k + 1][j])
                    }
                }
            }
        }

        return dp[0][n - 1]
    }

    fun strangePrinterRecursive(s: String): Int {
        fun removeDuplicates(s: String): String {
            val uniqueChars = StringBuilder()
            var i = 0
            while (i < s.length) {
                val currentChar = s[i]
                uniqueChars.append(currentChar)
                while (i < s.length && s[i] == currentChar) {
                    i++
                }
            }
            return uniqueChars.toString()
        }

        val shorted = removeDuplicates(s)
        val n = shorted.length
        val dp = Array(n) { IntArray(n) }

        fun minimumTurns(start: Int, end: Int): Int {
            if (start > end) return 0
            if (dp[start][end] > 0) return dp[start][end]

            // Initialize with worst case: print each character separately
            var minTurns = 1 + minimumTurns(start + 1, end)

            // Try to optimize by finding matching characters
            for (k in start + 1..end) {
                if (shorted[k] == shorted[start]) {
                    // If match found, try splitting the problem
                    minTurns =
                        minOf(minTurns, minimumTurns(start, k - 1) + minimumTurns(k + 1, end))
                }
            }
            dp[start][end] = minTurns
            return minTurns
        }

        return minimumTurns(0, n - 1)
    }

    fun nearestPalindromic(n: String): String {
        val number = n.toLong()
        val len: Int = n.length
        if (len == 1) return n.toInt().dec().toString()

        val center = if (len % 2 == 0) len / 2 - 1 else len / 2
        val firstHalf = n.substring(0, center + 1)

        fun getPalindrome(firstHalf: String, even: () -> Boolean): String = when {
            even() -> firstHalf + firstHalf.reversed()
            else -> firstHalf + firstHalf.dropLast(1).reversed()
        }

        return listOf(
            getPalindrome(firstHalf) { len % 2 == 0 } // 12321
            , getPalindrome(firstHalf.toLong().inc().toString()) { len % 2 == 0 } // 12421
            , getPalindrome(firstHalf.toLong().dec().toString()) { len % 2 == 0 } // 12221
            , 10.0.pow(len - 1).toLong().dec().toString() // 9999
            , 10.0.pow(len).toLong().inc().toString() // 100001
        )
            .map { it.toLong() }
            .filter { it != number }
            .minWithOrNull { o1, o2 ->
                val abs1 = abs(o1 - number)
                val abs2 = abs(o2 - number)
                if (abs1 != abs2) {
                    abs1.compareTo(abs2)
                } else {
                    o1.compareTo(o2)
                }
            }
            .toString()
    }

    fun kmpGetNext(s: String): IntArray {
        val next = IntArray(s.length) // 记录字符匹配失败之后跳转位置
        var i = 0
        var j = -1
        next[0] = -1

        while (i < s.length - 1) {
            if (j == -1 || s[i] == s[j]) {
                i++
                j++
                next[i] = j
            } else {
                j = next[j] //  如果匹配失败，通过next回退j的值，继续寻找之前可能的前缀
            }
        }
        return next
    }

    fun prefixLength(s: String): IntArray {
        val prefixArray = IntArray(s.length) // 记录最长前后缀的长度
        var j = 0  // 当前前缀的长度
        for (i in 1 until s.length) {
            // 回退：如果不匹配，通过前缀数组寻找上一个可匹配的前缀
            while (j > 0 && s[j] != s[i]) {
                j = prefixArray[j - 1]
            }
            // 如果匹配，前缀长度+1
            if (s[j] == s[i]) {
                j++
            }
            // 更新当前的前缀长度
            prefixArray[i] = j
        }
        return prefixArray
    }

    // KMP
    fun longestPrefix(s: String): String {
        val prefixArray = IntArray(s.length)
        var j = 0  // 当前前缀的长度
        for (i in 1 until s.length) {
            // 回退：如果不匹配，通过前缀数组寻找上一个可匹配的前缀
            while (j > 0 && s[j] != s[i]) {
                j = prefixArray[j - 1]
            }
            // 如果匹配，前缀长度+1
            if (s[j] == s[i]) {
                j++
            }
            // 更新当前的前缀长度
            prefixArray[i] = j
        }
        return s.substring(0, prefixArray.last())
    }

    fun fullJustify(words: Array<String>, maxWidth: Int): List<String> {
        val result = mutableListOf<String>()
        var start = 0
        var length = 0
        var necessarySpaces = -1
        for (end in words.indices) {
            length += words[end].length
            necessarySpaces++
            if (length + necessarySpaces > maxWidth) {
                // found a line
                length -= words[end].length
                val sb = StringBuilder()
                val totalSpaces = maxWidth - length
                val spaceSlots = end - 1 - start
                if (spaceSlots == 0) {
                    sb.append(words[start])
                    repeat(totalSpaces) {
                        sb.append(" ")
                    }
                } else {
                    val evenSpace = totalSpaces / spaceSlots
                    var extraSpace = totalSpaces % spaceSlots
                    for (i in start until end) {
                        sb.append(words[i])
                        if (i != end - 1) {
                            repeat(evenSpace) {
                                sb.append(" ")
                            }
                            if (extraSpace != 0) {
                                sb.append(" ")
                                extraSpace--
                            }
                        }
                    }
                }
                result.add(sb.toString())

                // new beginning
                start = end
                length = words[end].length
                necessarySpaces = 0
            }
        }

        if (length != 0) {
            val sb = StringBuilder()
            for (i in start..words.lastIndex) {
                sb.append(words[i])
                if (i != words.lastIndex) {
                    sb.append(" ")
                }
            }
            val totalSpaces = maxWidth - length - necessarySpaces
            repeat(totalSpaces) {
                sb.append(" ")
            }
            result.add(sb.toString())
        }
        return result
    }

    fun parseBoolExpr(expression: String): Boolean {
        val operations = LinkedList<Char>()
        val booleans = LinkedList<LinkedList<Boolean>>()
        for (char in expression) {
            if (char == '&' || char == '|' || char == '!') {
                operations.push(char)
            } else if (char == '(') {
                booleans.push(LinkedList<Boolean>())
            } else if (char == ')') {
                val operation = operations.poll()!!
                val sub = booleans.poll()!!
                var result = sub.poll()!!
                when (operation) {
                    '&' -> {
                        while (sub.isNotEmpty()) {
                            result = result and sub.poll()!!
                        }
                    }

                    '|' -> {
                        while (sub.isNotEmpty()) {
                            result = result or sub.poll()!!
                        }
                    }

                    '!' -> result = !result
                }
                if (booleans.peek() == null) {
                    return result
                } else {
                    booleans.peek()!!.push(result)
                }
            } else if (char == ',') {
                continue

            } else {
                booleans.peek()!!.push(char == 't')
            }
        }
        return false
    }

//    fun ladderLength(beginWord: String, endWord: String, wordList: List<String>): Int {
//         // good when wordList isn't very long
//        fun wordDistance(from: String, to: String): Int {
//            var count = 0
//            for (i in from.indices) {
//                if (from[i] != to[i]) count++
//            }
//            return count
//        }
//
//        val seen = hashSetOf<String>()
//        val queue = LinkedList<Pair<String, Int>>()
//        queue.offer(beginWord to 1)
//        seen.add(beginWord)
//        while (queue.isNotEmpty()) {
//            val (word, count) = queue.poll()!!
//            for (midWord in wordList) {
//                if (midWord in seen) continue
//                if (wordDistance(word, midWord) != 1) continue
//                if (midWord == endWord) return count + 1
//                seen.add(midWord)
//                queue.offer(midWord to count + 1)
//            }
//        }
//        return 0
//    }

    fun ladderLength(beginWord: String, endWord: String, wordList: List<String>): Int {
        // when wordList is too long, we change word on the go
        val wordSet = wordList.toMutableSet()
        val queue = LinkedList<Pair<String, Int>>()
        queue.offer(beginWord to 1)
        while (queue.isNotEmpty()) {
            val (word, count) = queue.poll()!!
            val sb = StringBuilder(word)
            for (i in sb.indices) {
                val oldChar = sb[i]
                for (newChar in 'a'..'z') {
                    if (oldChar == newChar) continue
                    sb[i] = newChar
                    val newWord = sb.toString()
                    if (newWord in wordSet) {
                        if (newWord == endWord) return count + 1
                        wordSet.remove(newWord)
                        queue.offer(newWord to count + 1)
                    }
                }
                sb[i] = oldChar
            }
        }
        return 0
    }

    fun getLengthOfOptimalCompression(s: String, k: Int): Int {

        fun calculateLength(count: Int): Int {
            return when {
                count == 0 -> 0
                count == 1 -> 1
                count in 2..9 -> 2
                count in 10..99 -> 3
                else -> 4
            }
        }

        val n = s.length
        // dp[i][j] means in first i letters, deleted j to get best (minimum) length
        val dp = Array(n + 1) { IntArray(k + 1) { Int.MAX_VALUE } }
        for (j in 0..k) {
            dp[0][j] = 0
        }
        for (i in 1..n) {
            for (j in 0..k) {
                // don't delete current letter
                var same = 0
                var diff = 0
                for (l in i downTo 1) {
                    if (s[l - 1] == s[i - 1]) same++ else diff++
                    if (j - diff >= 0) {
                        dp[i][j] = minOf(dp[i][j], dp[l - 1][j - diff] + calculateLength(same))
                    } else break
                }

                // delete current letter
                if (j > 0) {
                    dp[i][j] = minOf(dp[i][j], dp[i - 1][j - 1])
                }
            }
        }

        return dp[n][k]
    }

    fun minWindow(s: String, t: String): String {

        fun getIndex(letter: Char): Int {
            return if (letter.code < 'a'.code) {
                letter - 'A'
            } else {
                letter + 26 - 'a'
            }
        }

        var count = t.length
        val freq = IntArray(52)
        for (letter in t) {
            val letterIndex = getIndex(letter)
            freq[letterIndex]++
        }
        var minLength = Int.MAX_VALUE
        var result = ""
        var start = 0
        for (end in s.indices) {
            var letterIndex = getIndex(s[end])
            freq[letterIndex]--
            if (freq[letterIndex] >= 0) {
                count--
            }
            while (start <= end && count == 0) {
                letterIndex = getIndex(s[start])
                freq[letterIndex]++
                if (freq[letterIndex] > 0) {
                    count++
                }
                start++
            }
            // start - 1 is the last valid pos
            if (start > 0 && end - start + 2 < minLength) {
                minLength = end - start + 2
                result = s.substring(start - 1, end + 1)
            }
        }
        return result
    }

    fun calculate(s: String): Int {
//        val values = LinkedList<Int>()
//        val operations = LinkedList<Char>()
//        val numSb = StringBuilder()
//        var isStarting = true
//        for ((i, char) in s.withIndex()) {
//            if (char == ' ') {
//                continue
//            } else if (char == '(') {
//                isStarting = true
//            } else if (char == ')') {
//                if (operations.isNotEmpty()) {
//                    val last = values.pop()!!
//                    val top = values.pop()!!
//                    values.push(if (operations.pop() == '+') top + last else top - last)
//                }
//            } else if (char == '+' || char == '-') {
//                operations.push(char)
//                if (char == '-' && isStarting) {
//                    values.push(0)
//                    isStarting = false
//                }
//            } else {
//                numSb.append(char)
//                if (i + 1 == s.length || !s[i + 1].isDigit()) {
//                    val last = numSb.toString().toInt()
//                    numSb.clear()
//                    if (isStarting) {
//                        values.push(last)
//                        isStarting = false
//                    } else {
//                        val top = values.pop()!!
//                        values.push(if (operations.isEmpty() || operations.pop() == '+') top + last else top - last)
//                    }
//                }
//            }
//        }
//        return values.pop()
        var value = 0
        val shouldRevert = LinkedList<Boolean>()
        shouldRevert.push(false)
        var isMinus = false
        var currentNum = 0

        fun updateValue() {
            if (isMinus) value -= currentNum else value += currentNum
            currentNum = 0
        }

        for (char in s) {
            when (char) {
                ' ' -> Unit
                '(' -> shouldRevert.push(isMinus)
                ')' -> shouldRevert.pop()

                '+' -> {
                    updateValue()
                    isMinus = shouldRevert.peek()!!
                }

                '-' -> {
                    updateValue()
                    isMinus = !shouldRevert.peek()!!
                }

                else -> currentNum = currentNum * 10 + char.digitToInt()
            }
        }
        updateValue()
        return value
    }

    fun numWays(words: Array<String>, target: String): Int {
        val MODULO = 1_000_000_007
        val wordLength = words[0].length
        val targetLength = target.length
        val freq = Array(wordLength) { IntArray(26) }
        for (i in words.indices) {
            for (j in words[0].indices) {
                freq[j][words[i][j] - 'a']++
            }
        }

        val dp = LongArray(targetLength + 1)
        dp[0] = 1
        for (i in 1..wordLength) {
            var previous = dp[0]
            for (j in 1..targetLength) {
                val tmp = dp[j]
                dp[j] =
                    (dp[j] + (1L * previous * freq[i - 1][target[j - 1] - 'a']) % MODULO) % MODULO
                previous = tmp
            }
        }
        return dp[targetLength].toInt()
    }

    fun allSamePrefixSuffix(s: String): List<String> { // abcabdddabcab -> [ab, abcab]
        // kmp
        // 1. 构建前缀函数数组 lps: pi，pi[i] 表示在 s[0..i] 中最长相同前后缀的长度
        val n = s.length
        val pi = IntArray(n) { 0 }

        var j = 0 // j 既是“已经匹配的长度”，也是“当前要比较字符的下标”
        for (i in 1 until n) {
            while (j > 0 && s[i] != s[j]) {
                j = pi[j - 1]  // 回退
            }
            if (s[i] == s[j]) {
                j++
            }
            pi[i] = j
        }

        // 2. 从最后 pi[n - 1] 开始，往前不停地追溯，收集所有「相同前后缀」长度
        val result = mutableListOf<Int>()
        var length = pi[n - 1]
        while (length > 0) {
            result.add(length)
            length = pi[length - 1]
        }
        // 这里可以把长度转换成对应的前缀子串
        return result
            .sorted()  // 如果想从短到长，可以先排序
            .map { s.substring(0, it) }
    }

    fun shortestMatchingSubstring(s: String, p: String): Int {
        var length = Int.MAX_VALUE
        val parts = p.split('*').toList()
        if (parts[0].length == 0 && parts[1].length == 0 && parts[2].length == 0) return 0
        var start = s.indexOf(parts[0])
        while (start != -1 && start < s.length) {
            val mid = s.indexOf(parts[1], start + parts[0].length)
            if (mid != -1) {
                val end = s.indexOf(parts[2], mid + parts[1].length)
                if (end != -1) {
                    length = minOf(length, end + parts[2].length - start)
                }
            }
            start = s.indexOf(parts[0], start + 1)
        }
        return length
    }

    fun longestCommonSubsequence(str1: String, str2: String): String {
        // LCS
        val m = str1.length
        val n = str2.length
        val dp = Array(m + 1) { IntArray(n + 1) }
        for (i in m - 1 downTo 0) {
            for (j in n - 1 downTo 0) {
                dp[i][j] = if (str1[i] == str2[j]) {
                    dp[i + 1][j + 1] + 1
                } else {
                    maxOf(dp[i + 1][j], dp[i][j + 1])
                }
            }
        }
        var i = 0
        var j = 0
        val lcs = StringBuilder()
        while (i < m && j < n) {
            if (str1[i] == str2[j]) {
                lcs.append(str1[i])
                i++
                j++
            } else if (dp[i + 1][j] >= dp[i][j + 1]) {
                i++
            } else {
                j++
            }
        }
        return lcs.toString()
    }

    fun shortestCommonSupersequence(str1: String, str2: String): String {
        // SCS
        val m = str1.length
        val n = str2.length
        val dp = Array(m + 1) { IntArray(n + 1) }
        for (i in m - 1 downTo 0) {
            for (j in n - 1 downTo 0) {
                dp[i][j] = if (str1[i] == str2[j]) {
                    dp[i + 1][j + 1] + 1
                } else {
                    maxOf(dp[i + 1][j], dp[i][j + 1])
                }
            }
        }

        val sb = StringBuilder()
        var i = 0
        var j = 0
        while (i < m || j < n) {
            if (i == m) {
                sb.append(str2[j++])
            } else if (j == n) {
                sb.append(str1[i++])
            } else if (str1[i] == str2[j]) {
                sb.append(str1[i++])
                j++
            } else if (dp[i + 1][j] > dp[i][j + 1]) {
                sb.append(str1[i++])
            } else {
                sb.append(str2[j++])
            }
        }
        return sb.toString()
    }
}