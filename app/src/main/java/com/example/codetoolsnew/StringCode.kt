package com.example.codetools

import java.util.*
import kotlin.collections.ArrayDeque
import kotlin.math.abs
import kotlin.math.max

object StringCode {
    fun longestCommonPrefix(strs: Array<String>): String {
        //        if (strs.isEmpty()) return ""
        //        var prefix = strs[0]
        //        for (i in 1 until strs.size) {
        //            while (strs[i].indexOf(prefix) != 0) {
        //                prefix = prefix.substring(0, prefix.length - 1)
        //                if (prefix.isEmpty()) return ""
        //            }
        //        }
        //        return prefix
        val result = StringBuilder()
        strs[0].forEachIndexed { i, c ->
            if (strs.all { i < it.length && c == it[i] }) {
                result.append(c)
            } else {
                return result.toString()
            }
        }
        return result.toString()
    }

    fun isBracketsValid(s: String): Boolean {
        val testStack = Stack<Char>()
        if (s.length % 2 != 0) return false
        s.forEachIndexed { index, it ->
            if (s.length - index < testStack.size) return false
            if (it == '(' || it == '{' || it == '[') {
                testStack.push(it)
            } else if (it == ')' && testStack.isNotEmpty() && testStack.peek() == '(' ||
                it == '}' && testStack.isNotEmpty() && testStack.peek() == '{' ||
                it == ']' && testStack.isNotEmpty() && testStack.peek() == '['
            ) {
                testStack.pop()
            } else {
                return false
            }
        }
        if (testStack.isNotEmpty()) return false
        return true
    }

    fun strStr(haystack: String, needle: String): Int {
//        if (haystack == needle) return 0
//        val l = needle.length
//        for (i in 0 until haystack.length - l + 1) {
//            if (haystack.substring(i, i + l) == needle) {
//                return i
//            }
//        }
//        return -1
        var i = 0
        var j = 0
        while (j < haystack.length) {
            if (haystack[j] == needle[i]) {
                i++
                if (i == needle.length) {
                    return j - i + 1
                }
            } else {
                j -= i // reset j to the position after the first matched character
                i = 0
            }
            j++
        }
        return -1
    }

    fun lengthOfLastWord(s: String): Int {
//        return s.trim().substringAfterLast(" ").length

//        var length = 0
//        for (i in s.length -1 downTo 0) {
//            if (s[i] != ' ') {
//                length++
//            } else if (s[i] == ' ' && length > 0) {
//                return length
//            }
//        }
//        return length

        var end: Int? = null
        for (i in s.length - 1 downTo 0) {
            if (s[i] != ' ' && end == null) {
                end = i
            } else if (s[i] == ' ' && end != null) {
                return end - i
            }
        }
        return if (end == null) s.length else end + 1
    }

    fun addBinary(a: String, b: String): String {
        val res = StringBuilder()
        var i = a.length - 1
        var j = b.length - 1
        var carry = 0
        while (i >= 0 || j >= 0) {
            var sum = carry
            if (i >= 0) sum += a[i--] - '0'
            if (j >= 0) sum += b[j--] - '0'
            carry = if (sum > 1) 1 else 0
            res.append(sum % 2)
        }
        if (carry != 0) res.append(carry)
        return res.reverse().toString()
    }

    fun lengthOfLongestNoRepeatingSubstring(s: String): Int {
        if (s.isEmpty()) return 0

        //        val freq = IntArray(256)
        //        var start = 0
        //        var maxLength = 0
        //        for (end in s.indices) {
        //            freq[s[end].code] += 1
        //            while (freq[s[end].code] > 1) {
        //                freq[s[start++].code] -= 1
        //            }
        //            maxLength = maxOf(maxLength, end - start + 1)
        //        }
        //        return maxLength

        val charIndexMap = IntArray(256) { -1 }
        var start = 0
        var maxLength = 0
        for (end in s.indices) {
            if (charIndexMap[s[end].code] >= start) {
                start = charIndexMap[s[end].code] + 1
            }
            charIndexMap[s[end].code] = end
            maxLength = max(maxLength, end - start + 1)
        }
        return maxLength
    }

    fun isIsomorphic(s: String, t: String): Boolean {
        if (s.isEmpty()) return true
//        val chatMap = mutableMapOf<Char, Char>()
//        val seenSet = mutableSetOf<Char>()
//        for (i in s.indices) {
//            if (chatMap.containsKey(s[i])) {
//                if (chatMap[s[i]] != t[i]) {
//                    return false
//                }
//            } else if (seenSet.contains(t[i])) {
//                return false
//            } else {
//                chatMap[s[i]] = t[i]
//                seenSet.add(t[i])
//            }
//        }
//        return true

        val sToT = IntArray(256) { -1 }
        val tToS = IntArray(256) { -1 }
        for (i in s.indices) {
            val charS = s[i].code
            val charT = t[i].code

            if (sToT[charS] == -1 && tToS[charT] == -1) {
                sToT[charS] = charT
                tToS[charT] = charS
            } else {
                if (sToT[charS] != charT || tToS[charT] != charS) {
                    return false
                }
            }
        }
        return true
    }

    fun maxDepth(s: String): Int {
        // ((A)) parentheses depth
        var maxDepth = 0
        var openCount = 0
        s.forEach {
            if (it == '(') {
                openCount++
            } else if (it == ')') {
                maxDepth = max(maxDepth, openCount)
                openCount--
            }
        }
        return maxDepth
    }

    fun makeGood(s: String): String {
        // remove adjacent lower/upper case same letters pair
        // "leEeetcode" -> "leetcode"
        // Stack or Deque is less performant
        val result = StringBuilder()
        for (char in s) {
            if (result.isNotEmpty() && sameLetterButCase(result.last(), char)) {
                result.deleteCharAt(result.length - 1)
            } else {
                result.append(char)
            }
        }
        return result.toString()
    }

    fun sameLetterButCase(s1: Char, s2: Char): Boolean {
        return if (s1.isLowerCase() && s2.isUpperCase()) {
            s1 == s2.lowercaseChar()
        } else if (s1.isUpperCase() && s2.isLowerCase()) {
            s1.lowercaseChar() == s2
        } else {
            false
        }
    }

    fun minRemoveParentheses(s: String): String {
        // lee(t(c)o)de) -> lee(t(c)o)de or lee(t(co)de) or lee(t(c)ode)
        val result = StringBuilder()
        val mark = IntArray(s.length) { 0 }
        val openPositions = ArrayDeque<Int>()
        for (i in s.indices) {
            if (s[i] == '(') {
                mark[i] = 1
                openPositions.addLast(i)
            } else if (s[i] == ')') {
                if (openPositions.isNotEmpty()) {
                    mark[openPositions.removeLast()] = 0
                } else {
                    mark[i] = -1
                }
            }
        }
        for (i in s.indices) {
            if (mark[i] == 0) {
                result.append(s[i])
            }
        }
        return result.toString()
    }

    fun checkValidParenthesesWithStar(s: String): Boolean {
        // s contains '(' ')' '*'. '*' can be treated as anything
        // check possible open count range
        var openCountMin = 0
        var openCountMax = 0
        for (char in s) {
            when (char) {
                '(' -> {
                    openCountMin++
                    openCountMax++
                }

                ')' -> {
                    openCountMin--
                    openCountMax--
                }

                '*' -> {
                    openCountMin--
                    openCountMax++
                }
            }
            // if max < 0, it means there are too many closing parentheses, even * can't help
            if (openCountMax < 0) return false
            // if min < 0, it means one of the * can't be used as closing parentheses, we reset min to 0
            if (openCountMin < 0) openCountMin = 0
        }
        // at the end if min == 0, it means all the open parentheses can be closed
        return openCountMin == 0
    }

    fun isPalindromeOnlyLetters(s: String): Boolean {
        var i = 0
        var j = s.length - 1
        while (i < j && i < s.length && j > 0) {
            if (!s[i].isLetterOrDigit()) {
                i++
            } else if (!s[j].isLetterOrDigit()) {
                j--
            } else if (s[i++].lowercase() != s[j--].lowercase()) {
                return false
            }
        }
        return true
    }

    fun canConstruct(ransomNote: String, magazine: String): Boolean {
        val resCount = IntArray(26) { 0 }
        magazine.forEach { c ->
            resCount[c - 'a']++
        }
        ransomNote.forEach { c ->
            if (resCount[c - 'a'] == 0) {
                return false
            } else {
                resCount[c - 'a']--
            }
        }
        return true
    }

    fun isSubsequence(s: String, t: String): Boolean {
//        t.fold(0) { initial, char ->
//            initial + if (initial in s.indices && char == s[initial]) 1 else 0
//        } == s.length

        var i = 0
        var j = 0
        while (i in s.indices && j in t.indices) {
            while (t[j] != s[i]) {
                j++
                if (j == t.length) {
                    return false
                }
            }
            i++
            j++
        }
        return i == s.length
    }

    fun removeKdigits(num: String, k: Int): String {
        if (num.length == k) return "0"
        val passed = Stack<Char>()
        var i = 0
        var count = 0
        while (i < num.length) {
            if (count == k || passed.isEmpty() || num[i] >= passed.peek()) {
                passed.push(num[i])
                i++
            } else {
                passed.removeLast()
                count++
            }
        }
        val result = StringBuilder()
        count = 0
        var leadingZeros = true
        for (index in passed.indices) {
            if (passed[index] != '0' || !leadingZeros) {
                result.append(passed[index])
                leadingZeros = false
            }
            count++
            if (count == num.length - k) break
        }
        return if (result.isEmpty()) {
            "0"
        } else {
            result.toString()
        }
    }

    // brute-force
    fun wordBreak(s: String, wordDict: List<String>): Boolean {
        val groupedDict = wordDict.groupBy { it[0] }

        fun wordBreak(s: String): Boolean {
            if (s.isEmpty()) return true
            return groupedDict[s[0]]?.any {
                if (s.startsWith(it)) {
                    wordBreak(s.substring(it.length))
                } else {
                    false
                }
            } ?: false
        }

        return wordBreak(s)
    }

    fun wordBreakTrie(s: String, wordDict: List<String>): Boolean {
        class TrieNode {
            val children = mutableMapOf<Char, TrieNode>()
            var isWord = false
        }

        fun buildTrie(): TrieNode {
            val root = TrieNode()
            for (word in wordDict) {
                var node = root
                for (char in word) {
                    node = node.children.getOrPut(char) { TrieNode() }
                }
                node.isWord = true
            }
            return root
        }

        val n = s.length
        val trieRoot = buildTrie()
        // dp[i] 表示前 ( i ) 个字符是否可以被分割成字典中的单词
        val dp = BooleanArray(n + 1)
        dp[0] = true

        for (i in 0 until n) {
            if (!dp[i]) continue
            var node = trieRoot
            for (j in i until n) {
                node = node.children[s[j]] ?: break
                if (node.isWord) {
                    dp[j + 1] = true
                }
            }
        }
        return dp[n]
    }

    fun wordBreakDP(s: String, wordDict: List<String>): Boolean {
        val n = s.length
        // dp stores whether s[0, i-1] can be segmented into words in wordDict
        val dp = BooleanArray(n + 1) { false }
        dp[0] = true
        for (i in 1..n) {
            // for each i, the way to find whether it can be segmented is to check whether there is a j
            for (j in 0 until i) {
                // break the string into two parts: s[0, j-1] and s[j, i-1]
                if (dp[j] && wordDict.contains(s.substring(j, i))) {
                    dp[i] = true
                    break
                }
            }
        }
        return dp[n]
    }

    fun wordBreakListBfs(s: String, wordDict: List<String>): List<String> {
        val n = s.length
        val result = mutableListOf<MutableList<String>>()
        val queue = ArrayDeque<Pair<Int, List<String>>>()

        queue.add(Pair(0, listOf()))
        while (queue.isNotEmpty()) {
            val current = queue.removeFirst()
            for (i in current.first + 1..n) {
                val word = s.substring(current.first, i)
                if (wordDict.contains(word)) {
                    if (i == n) {
                        result.add(ArrayList(current.second).apply { add(word) })
                    } else {
                        queue.addLast(Pair(i, ArrayList(current.second).apply { add(word) }))
                    }
                }
            }
        }

        return result.map { it.joinToString(" ") }
    }

    fun wordBreakListDp(s: String, wordDict: List<String>): List<String> {
        val n = s.length
        // dp stores s[0, i-1]'s all possible valid segmented words
        val dp = Array(n + 1) { mutableListOf<MutableList<String>>() }
        for (i in 1..n) {
            // for each i, the way to find whether it can be segmented is to check whether there is a j
            for (j in 0 until i) {
                // break the string into two parts: s[0, j-1] and s[j, i-1]
                val currentString = s.substring(j, i)
                if ((j == 0 || dp[j].isNotEmpty()) && wordDict.contains(currentString)) {
                    if (j == 0) {
                        dp[i].add(arrayListOf(currentString))
                    } else {
                        dp[j].forEach {
                            dp[i].add(ArrayList(it).apply { add(currentString) })
                        }
                    }
                }
            }
        }
        return dp[n].map { it.joinToString(" ") }
    }

    fun wordPattern(pattern: String, s: String): Boolean {
        val patternToWord = HashMap<Int, String>()
        val wordToPattern = HashMap<String, Int>()
        var patterIndex = 0
        var i = 0
        val sb = StringBuilder()
        var currentPattern = 0
        var currentWord = ""
        while (i < s.length || sb.isNotEmpty()) {
            if (i < s.length && s[i] != ' ') {
                sb.append(s[i])
            } else {
                if (patterIndex > pattern.length - 1) return false
                currentPattern = pattern[patterIndex].code
                currentWord = sb.toString()
                if (patternToWord[currentPattern] == null && wordToPattern[currentWord] == null) {
                    patternToWord[currentPattern] = currentWord
                    wordToPattern[currentWord] = currentPattern
                } else {
                    if (patternToWord[currentPattern] != currentWord || wordToPattern[currentWord] != currentPattern) {
                        return false
                    }
                }
                sb.clear()
                patterIndex++
            }
            i++
        }
        if (patterIndex < pattern.length) return false
        return true
    }

    fun longestIdealSubsequence(s: String, k: Int): Int {
        // ideal: difference between every two adjacent letters is less than or equal to k
        if (k == 25) return s.length
        val lengthMap = IntArray(128) // only 'a' to 'z' have value
        var maxLength = 1
        var curLength = 0
        for (ch in s) {
            val range = IntRange(maxOf('a'.code, ch.code - k), minOf('z'.code, ch.code + k))
            var max = 0
            for (charCode in range) {
                max = max(lengthMap[charCode], max)
            }
            curLength = max + 1
            lengthMap[ch.code] = curLength
            maxLength = max(curLength, maxLength)
        }
        return maxLength
    }

    fun reversePrefix(word: String, ch: Char): String {
        val index = word.indexOf(ch)
        if (index == -1) return word
        val sb = StringBuilder(word)
        for (i in 0..index / 2) {
            val temp = sb[i]
            sb[i] = sb[index - i]
            sb[index - i] = temp
        }
        return sb.toString()
    }

    fun compareVersion(version1: String, version2: String): Int {
        var i1 = 0
        var i2 = 0
        val length1 = version1.length
        val length2 = version2.length
        var value1 = 0
        var value2 = 0
        var tmp = 0

        while (i1 < length1 || i2 < length2) {
            if (i1 < length1) {
                tmp = version1.indexOf('.', i1)
                if (tmp == -1) {
                    value1 = version1.substring(i1).toInt()
                    i1 = length1
                } else {
                    value1 = version1.substring(i1, tmp).toInt()
                    i1 = tmp + 1
                }
            } else {
                value1 = 0
            }

            if (i2 < length2) {
                tmp = version2.indexOf('.', i2)
                if (tmp == -1) {
                    value2 = version2.substring(i2).toInt()
                    i2 = length2
                } else {
                    value2 = version2.substring(i2, tmp).toInt()
                    i2 = tmp + 1
                }
            } else {
                value2 = 0
            }

            if (value1 == value2) {
                continue
            } else if (value1 < value2) {
                return -1
            } else if (value1 > value2) {
                return 1
            }
        }
        return 0
    }

    fun reverseWords(s: String): String {
        val list = LinkedList<String>()
        var i = 0
        var j = 0
        while (j < s.length) {
            if (s[j] == ' ') {
                if (s[i] == ' ') {
                    i++
                    j++
                    continue
                } else {
                    list.addLast(s.substring(i, j))
                    i = ++j
                }
            } else {
                j++
            }
        }
        if (i != s.length) {
            list.addLast(s.substring(i))
        }
        val sb = StringBuilder()
        for (wordIndex in list.size - 1 downTo 0) {
            sb.append(list[wordIndex])
            if (wordIndex != 0) {
                sb.append(' ')
            }
        }
        return sb.toString()
    }

    fun simplifyPath(path: String): String {
        val pathStack = ArrayDeque<String>()
        var latestSlash: Int
        var startingIndex = 1
        var segment: String
        while (startingIndex < path.length) {
            latestSlash = path.indexOf('/', startingIndex)
            if (latestSlash == -1) {
                segment = path.substring(startingIndex)
                latestSlash = path.lastIndex
            } else {
                segment = path.substring(startingIndex, latestSlash)
            }
            if (segment == "." || segment == "") {
                //ignore
            } else if (segment == "..") {
                pathStack.removeLastOrNull()
            } else {
                pathStack.addLast(segment)
            }
            startingIndex = latestSlash + 1
        }
        return pathStack.joinToString("/", "/")
    }

    fun letterCombinations(digits: String): List<String> {
        if (digits.isEmpty()) return listOf()
        val mapping = arrayOf("", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz")

        // dfs + backtracking
//        val result = mutableListOf<String>()
//        val sb = StringBuilder()
//
//        fun dfs(index: Int) {
//            if (index == digits.length) {
//                result.add(sb.toString())
//                return
//            }
//            val num = digits[index] - '0'
//            for (char in mapping[num]) {
//                sb.append(char)
//                dfs(index + 1)
//                sb.deleteAt(sb.lastIndex)
//            }
//        }
//
//        dfs(0)
//        return result

        // fold
        return digits.fold(listOf("")) { acc: List<String>, c: Char ->
            mapping[c - '0'].flatMap { newChar ->
                acc.map {
                    it + newChar
                }
            }
        }

//        val queue = ArrayDeque<String>()
//        queue.add("")
//        digits.forEach { digit ->
//            val size = queue.size
//            repeat(size) {
//                queue.removeFirst().let { current ->
//                    mapping[digit - '0'].forEach { newChar ->
//                        queue.addLast(current + newChar)
//                    }
//                }
//            }
//        }
//        return queue.toList()
    }

    fun isAnagram(s: String, t: String): Boolean {
        if (s.length != t.length) return false
        val freq = IntArray(26) { 0 }
        for (i in s.indices) {
            freq[s[i] - 'a']++
            freq[t[i] - 'a']--
        }
        for (f in freq) {
            if (f != 0) return false
        }
        return true
    }

    fun partitionPalindrome(s: String): List<List<String>> {
        val n = s.length
        val dp = Array(n) { BooleanArray(n) } // means if s[i..j] palindrome
        for (i in n - 1 downTo 0) {
            for (j in i until n) {
                dp[i][j] = (s[i] == s[j] && (j - i < 3 || dp[i + 1][j - 1]))
            }
        }

//        fun testValidSubstring(divIndices: List<Int>): MutableList<String>? {
//            val currentList = mutableListOf<String>()
//            var start = 0
//            for (index in divIndices + s.lastIndex) {
//                if (!dp[start][index]) {
//                    return null
//                }
//                currentList.add(s.substring(start, index + 1))
//                start = index + 1
//            }
//            return currentList
//        }
//
//        val result = mutableListOf<List<String>>()
//        // possible substring count: 2^(n - 1)
//        val divPossible = n - 1
//        for (i in 0 until (1 shl divPossible)) {
//            val divIndices = mutableListOf<Int>()
//            for (j in 0 until divPossible) {
//                if ((i and (1 shl j)) != 0) {
//                    divIndices.add(j)
//                }
//            }
//            testValidSubstring(divIndices)?.let {
//                result.add(it)
//            }
//        }
//        return result


        val result = mutableListOf<List<String>>()
        val currentPartition = mutableListOf<String>()

        fun dfs(start: Int) {
            if (start == n) {
                result.add(ArrayList(currentPartition))
                return
            }

            for (end in start until n) {
                if (dp[start][end]) {
                    currentPartition.add(s.substring(start, end + 1))
                    dfs(end + 1)
                    currentPartition.removeLast()
                }
            }
        }

        dfs(0)
        return result
    }

    fun equalSubstring(s: String, t: String, maxCost: Int): Int {
        val delta = IntArray(s.length)
        for (i in s.indices) {
            delta[i] = abs(s[i] - t[i])
        }
        var start = 0
        var sum = 0
        var length = 0
        for (end in delta.indices) {
            sum += delta[end]
            while (sum > maxCost) {
                sum -= delta[start]
                start++
            }
            length = maxOf(length, end - start + 1)
        }
        return length
    }

    fun numSteps(s: String): Int {
        var steps = 0
        var carry = false
        var index = s.lastIndex
        while (index != 0) {
            if (s[index] == '1') {
                if (carry) {
                    steps++
                } else {
                    carry = true
                    steps += 2
                }
            } else {
                if (carry) {
                    steps += 2
                } else {
                    steps++
                }
            }
            index--
        }
        return steps + if (carry) 1 else 0
    }

    fun longestPalindrome(s: String): String {
        val size = s.length
        if (size < 2) return s

        // s[i, j] palindrome or not; Two BooleanArray to rolling store
        var previous = BooleanArray(size)
        var current = BooleanArray(size)
        var longest = 0
        var start = -1
        for (i in size - 1 downTo 0) {
            for (j in i until size) {
                current[j] = s[i] == s[j] && (j - i < 3 || previous[j - 1])
                if (current[j]) {
                    val currentSize = j - i + 1
                    if (currentSize > longest) {
                        longest = currentSize
                        start = i
                    }
                }
            }
            // Swap current and previous
            val temp = previous
            previous = current
            current = temp
        }
        return s.substring(start, start + longest)
    }

    fun longestPalindromeCanBuild(s: String): Int {
        val freqMap = IntArray(128)
        s.forEach { char ->
            freqMap[char.code]++
        }
        var length = 0
        var hasOdd = false
        freqMap.forEach { freq ->
            if (freq % 2 == 1) {
                length += freq - 1
                hasOdd = true
            } else {
                length += freq
            }
        }
        return length + if (hasOdd) 1 else 0
    }

    fun isInterleave(s1: String, s2: String, s3: String): Boolean {
        if (s1.length + s2.length != s3.length) return false
        if (s1.isEmpty()) return s2 == s3
        if (s2.isEmpty()) return s1 == s3

        // rolling storage
//        var previous = BooleanArray(s2.length + 1)
//        var current = BooleanArray(s2.length + 1)
//        var temp: BooleanArray
//        previous[0] = true
//        for (j in 1..s2.length) {
//            previous[j] = previous[j - 1] && s2[j - 1] == s3[j - 1]
//        }
//        for (i in 1..s1.length) {
//            for (j in 0..s2.length) {
//                if (j == 0) {
//                    current[0] = previous[0] && s1[i - 1] == s3[i - 1]
//                } else {
//                    current[j] = previous[j] && s1[i - 1] == s3[i + j - 1] || // 从上面来，取用s1的字符
//                            current[j - 1] && s2[j - 1] == s3[i + j - 1] // 从左边来，取用s2的字符
//                }
//            }
//            // Swap current and previous
//            temp = previous
//            previous = current
//            current = temp
//        }
//        return previous[s2.length]

        // use first i items in s1, first j items in s2
        val dp = Array(s1.length + 1) { BooleanArray(s2.length + 1) }
        dp[0][0] = true
        for (i in 1..s1.length) {
            dp[i][0] = dp[i - 1][0] && s1[i - 1] == s3[i - 1]
        }
        for (j in 1..s2.length) {
            dp[0][j] = dp[0][j - 1] && s2[j - 1] == s3[j - 1]
        }
        for (i in 1..s1.length) {
            for (j in 1..s2.length) {
                dp[i][j] = dp[i - 1][j] && s1[i - 1] == s3[i + j - 1] || // 从上面来，取用s1的字符
                        dp[i][j - 1] && s2[j - 1] == s3[i + j - 1] // 从左边来，取用s2的字符
            }
        }
        return dp[s1.length][s2.length]
    }

    fun reverseString(s: CharArray): Unit {
        var start = 0
        var end = s.lastIndex
        var tmp = ' '
        while (start < end) {
            if (s[start] != s[end]) {
                tmp = s[start]
                s[start] = s[end]
                s[end] = tmp
            }
            start++
            end--
        }
    }

    fun appendCharacters(s: String, t: String): Int {
        var i = 0
        var j = 0
        while (i in s.indices && j in t.indices) {
            if (s[i] == t[j]) {
                i++
                j++
            } else {
                i++
            }
        }
        return t.length - j
    }

    fun commonChars(words: Array<String>): List<String> {
        val previous = IntArray(128)
        words[0].forEach { char ->
            previous[char.code]++
        }

        for (i in 1 until words.size) {
            val current = IntArray(128)
            words[i].forEach { char ->
                current[char.code]++
            }

            for (j in current.indices) {
                previous[j] = minOf(previous[j], current[j])
            }
        }
        val result = mutableListOf<String>()
        for (i in previous.indices) {
            repeat(previous[i]) {
                result.add(i.toChar().toString())
            }
        }
        return result
    }

    fun replaceWords(dictionary: List<String>, sentence: String): String {
        val sortedDict = dictionary.sorted()
        val result = StringBuilder()
        sentence.split(" ").forEach { current ->
            val root = sortedDict.firstOrNull { current.startsWith(it) }
            result.append(root ?: current)
            result.append(" ")
        }
        return result.toString().trim()
    }

    fun reverseParentheses(s: String): String {
        val sb = StringBuilder(s)
        val startIndices = LinkedList<Int>()
        for (i in s.indices) {
            if (s[i] == '(') {
                startIndices.push(i)
            } else if (s[i] == ')') {
                var start = startIndices.pop() + 1
                var end = i - 1
                var tmp = ' '
                while (start < end) {
                    tmp = sb[start]
                    sb[start] = sb[end]
                    sb[end] = tmp
                    start++
                    end--
                }
            }
        }
        return sb.filter { it != '(' && it != ')' }.toString()
    }

    fun maximumGain(s: String, x: Int, y: Int): Int {
        // remove "ab" or "ba"
        var score = 0
        val charStack = LinkedList<Char>()
        val first: Char
        val second: Char
        val bigger: Int
        val smaller: Int
        if (x > y) {
            first = 'a'
            second = 'b'
            bigger = x
            smaller = y
        } else {
            first = 'b'
            second = 'a'
            bigger = y
            smaller = x
        }
        s.forEach { char ->
            if (char == second && charStack.peek() == first) {
                score += bigger
                charStack.pop()
            } else {
                charStack.push(char)
            }
        }
        val intermediateStack = LinkedList<Char>()
        while (charStack.isNotEmpty()) {
            intermediateStack.push(charStack.pop())
        }
        while (intermediateStack.isNotEmpty()) {
            val char = intermediateStack.pop()
            if (char == first && charStack.peek() == second) {
                score += smaller
                charStack.pop()
            } else {
                charStack.push(char)
            }
        }
        return score
    }

    fun minimumDeletions(s: String): Int {
        // make no pair of (i,j) such that i < j and s[i] = 'b' and s[j]= 'a'
//        var aAfter = s.count { it == 'a' }
//        var bBefore = 0
//        var result = Int.MAX_VALUE
//        for (i in s.indices) {
//            result = minOf(result, bBefore + aAfter)
//            if (s[i] == 'b') {
//                bBefore++
//            } else {
//                aAfter--
//            }
//        }
//        result = minOf(result, bBefore + aAfter)
//        return result
        var deleteCount = 0
        var bCount = 0
        s.forEach { ch ->
            if (ch == 'b') {
                bCount++
            } else {
                if (bCount > 0) {
                    deleteCount++
                    bCount--
                }
            }
        }
        return deleteCount
    }

    fun digitSum(s: String, k: Int): String {
        var currentS = s
        var count = 0
        var sum = 0
        while (currentS.length > k) {
            val sb = StringBuilder()
            currentS.forEach { char ->
                sum += char.digitToInt()
                count++
                if (count == k) {
                    sb.append(sum.toString())
                    sum = 0
                    count = 0
                }
            }
            if (count != 0) {
                sb.append(sum.toString())
                sum = 0
                count = 0
            }
            currentS = sb.toString()
        }
        return currentS
    }

    fun countSubstrings(s: String, c: Char): Long {
        val count = s.count { it == c }
        return if (count % 2 == 0) {
            (count + 1L) * (count / 2)
        } else {
            (count + 1L) * (count / 2) + count / 2 + 1
        }
//        var count = 0
//        var result = 0L
//        for (i in 0 until s.length) {
//            if (s[i] == c) {
//                count++
//                result += count
//            }
//        }
//        return result
    }

    fun countConsistentStrings(allowed: String, words: Array<String>): Int {
        val allowedChar = BooleanArray(26)
        for (char in allowed) {
            allowedChar[char - 'a'] = true
        }
        var count = words.size
        for (word in words) {
            for (char in word) {
                if (!allowedChar[char - 'a']) {
                    count--
                    break
                }
            }
        }
        return count
    }


    fun numDifferentIntegers(word: String): Int {
        var start = 0
        val digits = mutableSetOf<String>()
        var pending = false
        for (end in word.indices) {
            if (!word[end].isDigit()) {
                if (pending) {
                    digits.add(removeLeadingZeros(word.substring(start, end)))
                    pending = false
                }
                start = end
            } else {
                if (!pending) {
                    pending = true
                    start = end
                }
            }
        }
        if (pending) {
            digits.add(removeLeadingZeros(word.substring(start)))
        }
        return digits.size
    }

    fun removeLeadingZeros(from: String): String {
        var index = -1
        for (i in from.indices) {
            if (from[i] != '0') {
                index = i
                break
            }
        }
        return if (index == -1) {
            "0"
        } else {
            from.substring(index)
        }
    }

    fun winnerOfGame(colors: String): Boolean {
        // 2038. Remove Colored Pieces if Both Neighbors are the Same Color
        var aStreak = 0
        var bStreak = 0
        var aTimes = 0
        var bTimes = 0
        for (color in colors) {
            if (color == 'A') {
                if (bStreak >= 3) {
                    bTimes += bStreak - 2
                }
                aStreak++
                bStreak = 0
            } else {
                if (aStreak >= 3) {
                    aTimes += aStreak - 2
                }
                bStreak++
                aStreak = 0
            }
        }
        if (bStreak >= 3) {
            bTimes += bStreak - 2
        }
        if (aStreak >= 3) {
            aTimes += aStreak - 2
        }
        return aTimes > bTimes
    }

    fun countAndSay(n: Int): String {
        if (n == 1) return "1"
        val last = countAndSay(n - 1)
        var count = 1
        val sb = StringBuilder()
        for (i in 1 until last.length) {
            if (last[i] == last[i - 1]) {
                count++
            } else {
                sb.append("$count${last[i - 1]}")
                count = 1
            }
        }
        sb.append("$count${last.last()}")
        return sb.toString()
    }

    fun compress(chars: CharArray): Int {
        var write = 0
        var count = 1
        for (i in 1 until chars.size) {
            if (chars[i] == chars[i - 1]) {
                count++
            } else {
                chars[write++] = chars[i - 1]
                if (count != 1) {
                    for (digit in count.toString()) {
                        chars[write++] = digit
                    }
                }
                count = 1
            }
        }
        // last group
        chars[write++] = chars.last()
        if (count != 1) {
            for (digit in count.toString()) {
                chars[write++] = digit
            }
        }
        return write
    }

    fun diffWaysToCompute(expression: String): List<Int> {
        if (expression.isEmpty()) return listOf()
        val result = mutableListOf<Int>()
        for ((i, char) in expression.withIndex()) {
            if (!char.isDigit()) {
                val left = diffWaysToCompute(expression.substring(0, i))
                val right = diffWaysToCompute(expression.substring(i + 1))
                for (num1 in left) {
                    for (num2 in right) {
                        when (char) {
                            '+' -> result.add(num1 + num2)
                            '-' -> result.add(num1 - num2)
                            '*' -> result.add(num1 * num2)
                        }
                    }
                }
            }
        }
        if (result.isEmpty()) {
            result.add(expression.toInt())
        }
        return result
    }

    fun checkInclusion(s1: String, s2: String): Boolean {
        // dfs
        //            var total = s1.length
        //            val s2Length = s2.length
        //            if (total > s2Length) return false
        //
        //            val charFreq = IntArray(26)
        //            for (char in s1) {
        //                charFreq[char - 'a']++
        //            }
        //
        //            fun dfs(index: Int, toLeft: Boolean): Boolean {
        //                if (total == 0) return true
        //                if (index < 0 || index >= s2Length) return false
        //                val charCode = s2[index] - 'a'
        //                var found = false
        //                if (charFreq[charCode] > 0) {
        //                    charFreq[charCode]--
        //                    total--
        //                    found = dfs(if (toLeft) index - 1 else index + 1, toLeft)
        //                } else {
        //                    return false
        //                }
        //                charFreq[charCode]++
        //                total++
        //                return found
        //            }
        //
        //            for (i in s2.indices) {
        //                val found = dfs(i, true) || dfs(i, false)
        //                if (found) {
        //                    return true
        //                }
        //            }
        //
        //            return false

        // sliding window
        //        var total = s1.length
        //        var s2Length = s2.length
        //        if (total > s2Length) return false
        //
        //        val charFreq = IntArray(26)
        //        for (char in s1) {
        //            charFreq[char - 'a']++
        //        }
        //
        //        var start = 0
        //        for (end in s2.indices) {
        //            val charCode = s2[end] - 'a'
        //            charFreq[charCode]--
        //            total--
        //            while (charFreq[charCode] < 0) {
        //                charFreq[s2[start] - 'a']++
        //                start++
        //                total++
        //            }
        //            if (total == 0) return true
        //        }
        //
        //        return false

        val n1 = s1.length
        val n2 = s2.length
        if (n1 > n2) return false

        val array1 = IntArray(26)
        val array2 = IntArray(26)
        for (i in s1.indices) {
            array1[s1[i] - 'a']++
            array2[s2[i] - 'a']++
        }
        for (i in 0 until n2 - n1) {
            if (array1.contentEquals(array2)) return true
            array2[s2[i] - 'a']--
            array2[s2[i + n1] - 'a']++
        }
        return array1.contentEquals(array2)
    }

    fun areSentencesSimilar(sentence1: String, sentence2: String): Boolean {
        val shortArray: List<String>
        val longArray: List<String>
        if (sentence1.length < sentence2.length) {
            shortArray = sentence1.split(" ")
            longArray = sentence2.split(" ")
        } else {
            shortArray = sentence2.split(" ")
            longArray = sentence1.split(" ")
        }
        val shortLen = shortArray.size
        val longLen = longArray.size
        var l = 0
        var r = shortLen - 1
        var i = 0
        while (l < shortLen) {
            if (shortArray[l] != longArray[i]) break
            l++
            i++
        }
        i = longLen - 1
        while (r >= l) {
            if (shortArray[r] != longArray[i]) break
            r--
            i--
        }
        return l - 1 == r
    }

    fun minLengthAfterRemoving(s: String): Int {
        // "AB" "CD"
//        val sb = StringBuilder(s)
//        while (true) {
//            var i = sb.indexOf("AB")
//            if (i == -1) {
//                i = sb.indexOf("CD")
//            }
//            if (i == -1) break
//            sb.delete(i, i + 2)
//        }
//        return sb.length

        val stack = LinkedList<Char>()
        for (char in s) {
            if (char == 'B' && stack.peek() == 'A' || char == 'D' && stack.peek() == 'C') {
                stack.pop()
            } else {
                stack.push(char)
            }
        }
        return stack.size
    }

    fun minSwapsToBalance(s: String): Int {
//        val sb = StringBuilder()
//        for (char in s) {
//            if (sb.isEmpty() || char == '[') {
//                sb.append(char)
//            } else {
//                if (sb.last() == '[') {
//                    sb.deleteAt(sb.lastIndex)
//                } else {
//                    sb.append(char)
//                }
//            }
//        }
//        return (sb.length / 2 + 1) / 2

        var stack = 0 // un-matched "[" count
        var swaps = 0
        for (char in s) {
            if (char == '[') {
                stack++
            } else {
                if (stack == 0) { // no [ yet, need +1 swap; then after swap, +1 "[" stack
                    swaps++
                    stack++
                } else { // can close
                    stack--
                }
            }
        }
        return swaps
    }

    fun generateParenthesis(n: Int): List<String> {
        val result = mutableListOf<String>()
        val sb = StringBuilder()

        fun dfs(open: Int, total: Int) {
            if (total == n && open == 0) {
                result.add(sb.toString())
                return
            }
            if (total < n) {
                sb.append('(')
                dfs(open + 1, total + 1)
                sb.deleteAt(sb.lastIndex)
            }
            if (open > 0) {
                sb.append(')')
                dfs(open - 1, total)
                sb.deleteAt(sb.lastIndex)
            }
        }

        dfs(0, 0)
        return result
    }

    fun generateParenthesisDP(n: Int): List<String> {
        val dp = Array(n + 1) { mutableListOf<String>() }
        dp[0].add("")

        for (i in 1..n) {
            for (j in 0 until i) {
                // 左边取 dp[j] 的结果集
                for (left in dp[j]) {
                    // 右边取 dp[i - 1 - j] 的结果集
                    for (right in dp[i - 1 - j]) {
                        // 组合成 "(" + left + ")" + right 的形式，加入 dp[i]
                        dp[i].add("($left)$right")
                    }
                }
            }
        }
        return dp[n]
    }

    fun minimumSteps(s: String): Long {
        var zeroCount = 0
        var move = 0L
        for (i in s.lastIndex downTo 0) {
            if (s[i] == '0') {
                zeroCount++
            } else {
                move += zeroCount
            }
        }
        return move
    }

    fun longestDiverseString(a: Int, b: Int, c: Int): String {
        val pq = PriorityQueue<Pair<Int, Char>> { o1, o2 -> o2.first - o1.first }
        if (a > 0) pq.offer(a to 'a')
        if (b > 0) pq.offer(b to 'b')
        if (c > 0) pq.offer(c to 'c')
        var streak = 0
        var last = ' '
        val sb = StringBuilder()
        while (pq.isNotEmpty()) {
            val (num, char) = pq.poll()!!
            if (char != last) {
                last = char
                streak = 1
                sb.append(char)
                if (num - 1 != 0) {
                    pq.offer(num - 1 to char)
                }
            } else if (streak < 2) {
                streak++
                sb.append(char)
                if (num - 1 != 0) {
                    pq.offer(num - 1 to char)
                }
            } else {
                if (pq.isNotEmpty()) {
                    val (nextNum, nextChar) = pq.poll()!!
                    last = nextChar
                    streak = 1
                    sb.append(nextChar)
                    if (nextNum - 1 != 0) {
                        pq.offer(nextNum - 1 to nextChar)
                    }
                    pq.offer(num to char)
                }
            }
        }
        return sb.toString()
    }

    fun maximumSwap(num: Int): Int {
        val sb = StringBuilder("$num")

        // pq
//        val pq = PriorityQueue<Int>(compareByDescending<Int> { sb[it] }.thenBy { it })
//        for (i in sb.indices) {
//            pq.offer(i)
//        }
//        var index = 0
//        while (pq.isNotEmpty()) {
//            var j = pq.poll()!!
//            // find the first number not matching descending order
//            if (j != index) {
//                // find the furthest to swap
//                while (pq.isNotEmpty() && sb[pq.peek()!!] == sb[j]) {
//                    j = pq.poll()!!
//                }
//                val tmp = sb[index]
//                sb[index] = sb[j]
//                sb[j] = tmp
//                break
//            }
//            index++
//        }

        var minToSwap = -1
        var maxToSwap = -1
        var maxIndex = sb.lastIndex
        for (i in sb.lastIndex downTo 0) {
            if (sb[i] == sb[maxIndex]) continue
            if (sb[i] > sb[maxIndex]) {
                maxIndex = i
            } else {
                minToSwap = i
                maxToSwap = maxIndex
            }
        }
        if (maxToSwap != -1) {
            val tmp = sb[maxToSwap]
            sb[maxToSwap] = sb[minToSwap]
            sb[minToSwap] = tmp
        }

        return sb.toString().toInt()
    }


    fun minMutation(startGene: String, endGene: String, bank: Array<String>): Int {
        fun mutationCount(from: String, to: String): Int {
            var count = 0
            for (i in from.indices) {
                if (from[i] != to[i]) count++
            }
            return count
        }

        val seen = hashSetOf<String>()
        val queue = LinkedList<Pair<String, Int>>()
        queue.offer(startGene to 0)
        while (queue.isNotEmpty()) {
            val (gene, count) = queue.poll()!!
            bank.filter { mutationCount(gene, it) == 1 }.forEach { midGene ->
                if (midGene !in seen) {
                    seen.add(midGene)
                    if (midGene == endGene) return count + 1
                    queue.offer(midGene to count + 1)
                }
            }
        }
        return -1
    }

    fun rotateString(s: String, goal: String): Boolean {
        if (s.length != goal.length) return false
        //        val doubled = s + s
        //        return doubled.contains(goal)
        val sb = StringBuilder(s)
        for (i in sb.indices) {
            if (sb.toString() == goal) return true
            sb.append(sb[0])
            sb.deleteAt(0)
        }
        return false
    }

    fun minChanges(s: String): Int {
//        val dp = IntArray(s.length + 1) { Int.MAX_VALUE }
//        dp[0] = 0
//        for (i in 1 .. s.length) {
//            if (i % 2 == 1) continue
//            var zeroCount = 0
//            var oneCount = 0
//            for (j in i downTo 1) {
//                if (s[j - 1] == '1') oneCount++ else zeroCount++
//                if (dp[j - 1] != Int.MAX_VALUE) {
//                    val changes = minOf(zeroCount, oneCount)
//                    dp[i] = minOf(dp[i], changes + dp[j - 1])
//                }
//            }
//        }
//        return dp.last()
        var changes = 0
        for (i in 0 until s.length step 2) {
            if (s[i] != s[i + 1]) changes++
        }
        return changes
    }

    fun takeCharacters(s: String, k: Int): Int {
        val rest = IntArray(3) { -k }
        for (char in s) {
            rest[char - 'a']++
        }
        if (rest.any { it < 0 }) return -1

        var start = 0
        var maxRest = Int.MIN_VALUE
        for (end in s.indices) {
            rest[s[end] - 'a']--
            while (start <= end && rest.any { it < 0 }) {
                rest[s[start] - 'a']++
                start++
            }
            maxRest = maxOf(maxRest, end - start + 1)
        }
        return if (maxRest == Int.MIN_VALUE) -1 else s.length - maxRest
    }

    fun isPrefixOfWord(sentence: String, searchWord: String): Int {
        var wordIndex = 1 // words are 1-indexed
        var searchI = 0
        val searchLength = searchWord.length
        var valid = true
        for (i in sentence.indices) {
            if (sentence[i] == ' ') {
                valid = true
                searchI = 0
                wordIndex++
                continue
            }
            if (!valid) continue
            if (sentence[i] == searchWord[searchI]) {
                searchI++
                if (searchI == searchLength) return wordIndex
            } else {
                valid = false
            }
        }
        return -1
    }

    fun canChange(start: String, target: String): Boolean {
        val n = start.length
        var i = 0
        var j = 0
        while (i < n && j < n) {
            if (start[i] == '_') {
                i++
                continue
            }
            if (target[j] == '_') {
                j++
                continue
            }
            if (start[i] != target[j] || start[i] == 'L' && i < j || start[i] == 'R' && i > j) return false
            i++
            j++
        }
        while (i < n && start[i] == '_') i++
        while (j < n && target[j] == '_') j++
        return i == j
    }

    fun maximumLength(s: String): Int {
//        val occurs = Array(26) { IntArray(s.length + 1) }
//        var start = 0
//        var longest = -1
//        var end = 0
//        while (end <= s.length) {
//            if (end == s.length || s[start] != s[end]) {
//                // "aaaa"x1, "aaa"x2, "aa"x3, "a"x4
//                val length = end - start
//                for (i in 1 .. length) {
//                    val charIndex = s[start] - 'a'
//                    occurs[charIndex][i] += length - i + 1
//
//                    if (occurs[charIndex][i] >= 3) {
//                        longest = maxOf(longest, i)
//                    }
//                }
//                start = end
//            }
//            end++
//        }
//        return longest

        fun isValid(length: Int) : Boolean {
            val count = IntArray(26)
            var start = 0
            for (end in s.indices) {
                while (s[start] != s[end]) start++
                if (end - start + 1 >= length) {
                    count[s[end] - 'a']++
                }
                if (count[s[end] - 'a'] > 2) return true
            }
            return false
        }

        if (!isValid(1)) return -1

        var left = 2
        var right = s.length
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (isValid(mid)) {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return right
    }

    fun repeatLimitedString(s: String, repeatLimit: Int): String {
        val freq = IntArray(26)
        for (char in s) {
            freq[char - 'a']++
        }
        val sb = StringBuilder()
        var right = 25
        var left = 24
        while (right >= 0) {
            while (left >= 0 && freq[left] == 0) left--
            if (freq[right] > repeatLimit) {
                sb.append(CharArray(repeatLimit) { 'a' + right })
                freq[right] -= repeatLimit
                if (left < 0) break
                sb.append('a' + left)
                freq[left]--
            } else {
                sb.append(CharArray(freq[right]) { 'a' + right })
                right = left
                left--
            }
        }
        return sb.toString()
    }

    fun countGoodStrings(low: Int, high: Int, zero: Int, one: Int): Int {
        val MODULO = 1_000_000_007
        var count = 0
        val dp = IntArray(high + 1)
        dp[0] = 1

        fun addUp(lastOne: Int, lastZero: Int): Int {
            return ((if (lastZero >= 0) dp[lastZero] else 0) + (if (lastOne >= 0) dp[lastOne] else 0)) % MODULO
        }

        for (i in 1 until low) {
            dp[i] = addUp(i - one, i - zero)
        }
        for (i in low..high) {
            dp[i] = addUp(i - one, i - zero)
            count = (count + dp[i]) % MODULO
        }
        return count
    }

    fun count3LengthPalindromicSubsequences(s: String): Int {
        var count = 0
        val firstIndices = IntArray(26) { -1 }
        val lastIndices = IntArray(26) { -1 }
        for (i in s.indices) {
            val charIndex = s[i] - 'a'
            if (firstIndices[charIndex] == -1) {
                firstIndices[charIndex] = i
            }
            lastIndices[charIndex] = i
        }
        for (i in 0 until 26) {
            val lastIndex = lastIndices[i]
            val firstIndex = firstIndices[i]
            if (lastIndex < 2 || lastIndex - firstIndex < 2) continue
            val middle = BooleanArray(26)
            for (pos in (firstIndex + 1) until lastIndex) {
                val chIndex = s[pos] - 'a'
                // 如果还没标记过，就标记并计数 + 1
                if (!middle[chIndex]) {
                    middle[chIndex] = true
                    count++
                }
            }
        }
        return count
    }

    fun swapOneMakeSameDistinct(word1: String, word2: String): Boolean {
        val freq1 = IntArray(26)
        var count1 = 0
        for (c in word1) {
            if (freq1[c - 'a']++ == 0) {
                count1++
            }
        }
        val freq2 = IntArray(26)
        var count2 = 0
        for (c in word2) {
            if (freq2[c - 'a']++ == 0) {
                count2++
            }
        }
        for (i in 0 until 26) {
            if (freq1[i] == 0) continue
            for (j in 0 until 26) {
                if (freq2[j] == 0) continue
                if (i == j) {
                    if (count1 == count2) return true
                    continue
                }
                var newCount1 = count1
                var newCount2 = count2
                if (freq1[i] == 1) newCount1--
                if (freq1[j] == 0) newCount1++
                if (freq2[i] == 0) newCount2++
                if (freq2[j] == 1) newCount2--
                if (newCount1 == newCount2) return true
            }
        }
        return false
    }
}