package com.example.codetools

object BitCode {
    fun minFlipOperations(nums: IntArray, k: Int): Int {
        // XOR all nums to get k
        var currentResult = 0
        for (num in nums) {
            currentResult = currentResult xor num
        }

        var count = 0
        var currentK = k
        while (currentK != 0 || currentResult != 0) {
            if ((currentK and 1) != (currentResult and 1)) {
                count++
            }
            currentK = currentK ushr 1
            currentResult = currentResult ushr 1
        }

        return count
    }

    fun singleNumber(nums: IntArray): Int {
        // [4,1,2,1,2] => find 4
        // downward so we can get result in [0]
        for (i in nums.size - 2 downTo 0) {
            nums[i] = nums[i + 1] xor nums[i]
        }
        return nums[0]
    }

    fun singleNumberFrom3(nums: IntArray): Int {
        // 1 once, others 3 times
        // [0,1,0,1,0,1,99] => find 99
        var res = 0
        for (i in 0 until 32) {
            var sum = 0
            val x = (1 shl i)
            for (num in nums) {
                if (num and x != 0) {
                    sum += 1
                }
            }
            if (sum % 3 != 0) {
                res = res or x
            }
        }
        return res
    }

    fun reverseBits(n: Int): Int {
        var result = n and 1
        var currentN = n shr 1
        repeat(31) {
            result = (result shl 1) or (currentN and 1)
            currentN = currentN ushr 1
        }
        return result
    }

    fun hammingWeight(n: Int): Int {
        return if (n == 0) 0 else (n and 1) + hammingWeight(n shr 1)
    }

    fun wonderfulSubstrings(word: String): Long {
        // wonderful: a string where at most one letter appears an odd number of times.
        var bitmask = 0
        val count = LongArray(1024)
        count[0] = 1
        var result = 0L
        for (char in word) {
            bitmask = bitmask xor (1 shl (char - 'a'))
            for (i in 0 until 10) {
                result += count[bitmask xor (1 shl i)]
            }
            result += count[bitmask]
            count[bitmask]++
        }
        return result
    }


    fun twoSingleNumber(nums: IntArray): IntArray {
        var xorResult = 0
        for (i in nums.indices) {
            xorResult = xorResult xor nums[i]
        }
        var bitToTest = 1
        while (xorResult and bitToTest == 0) {
            bitToTest = bitToTest shl 1
        }
        val result = intArrayOf(0, 0)
        for (num in nums) {
            if (num and bitToTest == bitToTest) {
                result[0] = result[0] xor num
            } else {
                result[1] = result[1] xor num
            }
        }
        return result
    }

    fun rangeBitwiseAnd(left: Int, right: Int): Int {
        var l = left
        var r = right

//        // 找到 left 和 right 的公共前缀
//        while (l < r) {
//            r = r and (r - 1)
//        }
//        return r

        var shift = 0
        // Find the common prefix
        while (l < r) {
            l = l shr 1
            r = r shr 1
            shift++
        }
        // Shift the common prefix back to the left
        return l shl shift
    }

    fun findComplement(num: Int): Int {
//        var mask = 1L
//        while (mask <= num) {
//            mask = mask shl 1
//        }
//        return num xor (mask - 1).toInt()

        val mask = (num.takeHighestOneBit() shl 1) - 1
        return num xor mask
    }

    fun minBitFlips(start: Int, goal: Int): Int {
        var diff = start xor goal
        var count = 0
        while (diff != 0) {
            // remove right-most "1"
            diff = diff and (diff - 1)
            count++

            // check bit by bit
//            count += diff and 1
//            diff = diff ushr 1
        }
        return count
    }

    fun minFlips(a: Int, b: Int, c: Int): Int {
        // flip a, b to make a OR b == c
        var currentA = a
        var currentB = b
        var currentC = c
        var result = 0
        while (currentC != 0 || currentA != 0 || currentB != 0) {
            val count = (currentA and 1) + (currentB and 1) // how many 1s
            result += if (currentC and 1 == 1) {
                if (count == 0) 1 else 0
            } else {
                count
            }
            currentA = currentA ushr 1
            currentB = currentB ushr 1
            currentC = currentC ushr 1
        }
        return result
    }

    fun xorQueries(arr: IntArray, queries: Array<IntArray>): IntArray {
        val size = arr.size
        val prefixXor = IntArray(size)
        prefixXor[0] = arr[0]
        for (i in 1 until size) {
            prefixXor[i] = prefixXor[i - 1] xor arr[i]
        }
        return IntArray(queries.size) { i ->
            val left = queries[i][0]
            val right = queries[i][1]
            when (left) {
                right -> {
                    arr[left]
                }

                0 -> {
                    prefixXor[right]
                }

                else -> {
                    prefixXor[right] xor prefixXor[left - 1]
                }
            }
        }
    }

    fun longestSubArrayBitAnd(nums: IntArray): Int {
        var max = 0
        var maxLength = 0
        var count = 0
        for (num in nums) {
            if (num > max) {
                max = num
                count = 1
                maxLength = 1
            } else if (num == max) {
                count++
                maxLength = maxOf(maxLength, count)
            } else {
                count = 0
            }
        }
        return maxLength
    }

    fun findTheLongestSubstring(s: String): Int {
        // a, e, i, o, u are even times
        val vowelMaskMap = mutableMapOf(0 to -1)
        var maxLength = 0
        var mask = 0

        for (i in s.indices) {
            // Update mask based on current character
            when (s[i]) {
                'a' -> mask = mask xor 1
                'e' -> mask = mask xor (1 shl 1)
                'i' -> mask = mask xor (1 shl 2)
                'o' -> mask = mask xor (1 shl 3)
                'u' -> mask = mask xor (1 shl 4)
            }
            // If mask has been seen before, it means the substring between
            // the previous occurrence and the current index has even counts of all vowels.
            if (vowelMaskMap.containsKey(mask)) {
                maxLength = maxOf(maxLength, i - vowelMaskMap[mask]!!)
            } else {
                vowelMaskMap[mask] = i
            }
        }

        return maxLength
    }

    fun getMaximumXor(nums: IntArray, maximumBit: Int): IntArray {
        val n = nums.size
        val result = IntArray(n)
        var xorResult = 0
        val mask = (1 shl maximumBit) - 1 // create all 1 mask
        for (i in 0 until n) {
            xorResult = xorResult xor nums[i]
            result[n - 1 - i] = xorResult xor mask // xor flip
        }
        return result
    }

    fun minEnd(n: Int, x: Int): Long {
        var toAdd = (n - 1).toLong()
        var mask = 1L
        var current = x.toLong()
        while (toAdd != 0L) {
            while (current and mask != 0L) {
                mask = mask shl 1
            }
            if (toAdd and 1L == 1L) {
                current = current or mask
            }
            toAdd = toAdd shr 1
            mask = mask shl 1
        }
        return current
    }

    fun minimumSubarrayLength(nums: IntArray, k: Int): Int {
        // val kBit = IntArray(30)
        // for (bit in 0 until 30) {
        //     if (k and (1 shl bit) != 0) {
        //         kBit[bit] = 1
        //     }
        // }
        var result = Int.MAX_VALUE
        var orSum = 0
        val bitCount = IntArray(30)
        var start = 0
        for (end in nums.indices) {
            orSum = orSum or nums[end]
            for (bit in 0 until 30) {
                if (nums[end] and (1 shl bit) != 0) {
                    bitCount[bit]++
                }
            }
            while (start <= end && orSum >= k) {
                for (bit in 0 until 30) {
                    if (nums[start] and (1 shl bit) != 0) {
                        bitCount[bit]--
                        if (bitCount[bit] == 0) {
                            orSum = orSum xor (1 shl bit)
                        }
                    }
                }
                start++
            }
            if (start > 0) {
                result = minOf(result, end - (start - 1) + 1)
            }
        }
        return if (result == Int.MAX_VALUE) -1 else result
    }

    fun maxEqualRowsAfterFlips(matrix: Array<IntArray>): Int {
        val patternFreq = mutableMapOf<String, Int>()
        var max = 0
        val sb = StringBuilder()
        for (row in matrix) {
            sb.clear()
            if (row[0] == 1) {
                for (char in row) {
                    sb.append(char or 1)
                }
            } else {
                for (char in row) {
                    sb.append(char)
                }
            }
            max = maxOf(max, patternFreq.merge(sb.toString(), 1, Int::plus)!!)
        }
        return max
    }

    fun findThePrefixCommonArray(A: IntArray, B: IntArray): IntArray {
        var xorSum = 0L
        val result = IntArray(A.size)
        var count = 0
        for (i in A.indices) {
            xorSum = xorSum xor (1L shl A[i]) xor (1L shl B[i])
            if (A[i] == B[i]) {
                count++
            } else {
                if (xorSum and (1L shl A[i]) == 0L) {
                    count++
                }
                if (xorSum and (1L shl B[i]) == 0L) {
                    count++
                }
            }
            result[i] = count
        }
        return result
    }

    fun kthCharacter(k: Int): Char {
//        return 'a' + Integer.bitCount(k - 1)
        val arr = IntArray(513)
        for (i in 0 until 9) {
            val cou = 1 shl i
            for (j in 0 until cou) {
                arr[cou + j] = (arr[j] + 1) % 26
            }
        }
        return 'a' + arr[k - 1]
    }

    fun kthCharacter(k: Long, operations: IntArray): Char {
        var index = 0
        var kth = k - 1
        var count = 0
        while (kth > 0) {
            if (operations[index++] == 1) {
                count += (kth and 1L).toInt()
            }
            kth = kth shr 1
        }
        return 'a' + (count % 26)
    }

    fun maximumSubsequence(nums: IntArray): Int {
        // (sub[0] + sub[1]) % 2 == (sub[1] + sub[2]) % 2 == ... == (sub[x - 2] + sub[x - 1]) % 2
        var lastBit = nums[0] and 1
        var zeroCount = if (lastBit == 0) 1 else 0
        var alterCount = 1
        for (i in 1 until nums.size) {
            if (nums[i] and 1 == 0) {
                zeroCount++
            }
            if (nums[i] and 1 != lastBit) {
                alterCount++
                lastBit = lastBit xor 1
            }
        }
        return maxOf(zeroCount, nums.size - zeroCount, alterCount)
    }
}