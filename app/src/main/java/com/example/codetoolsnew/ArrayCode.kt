package com.example.codetools

import java.util.LinkedList
import java.util.PriorityQueue
import java.util.TreeMap
import java.util.TreeSet
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min


object ArrayCode {
    private val MODULO = 1_000_000_007

    fun twoSum(nums: IntArray, target: Int): IntArray {
//        for (i in nums.indices) {
//            for (j in i + 1 until nums.size) {
//                if (target == nums[i] + nums[j]) {
//                    return intArrayOf(i, j)
//                }
//            }
//        }
//        return intArrayOf(0)

//        val indexMap = HashMap<Int, Int>()
//        nums.forEachIndexed { index, i -> indexMap[i] = index }
//        nums.forEachIndexed { index1, i ->
//            val rest = target - i
//            if (indexMap.containsKey(rest) && index1 != indexMap[rest]) {
//                return intArrayOf(index1, indexMap[rest]!!)
//            }
//        }
//        return intArrayOf(0)

        val indexMap = HashMap<Int, Int>()
        nums.forEachIndexed { index, i ->
            indexMap[i] = index
            val rest = target - i
            if (indexMap.containsKey(rest) && index != indexMap[rest]) {
                return intArrayOf(index, indexMap[rest]!!)
            }
        }
        return intArrayOf(0)
    }

    fun twoSumOrdered(numbers: IntArray, target: Int): IntArray {
        var start = 0
        var end = numbers.lastIndex
        var sum: Int
        while (start < end) {
            sum = numbers[start] + numbers[end]
            when {
                sum > target -> end--
                sum < target -> start++
                else -> break
            }
        }
        return intArrayOf(start + 1, end + 1)
    }

    fun removeDuplicates(nums: IntArray): Int {
        if (nums.size <= 1) return nums.size
        var i = 0
        for (cur in 1 until nums.size) {
            if (nums[cur] != nums[i]) {
                i++
                if (i != cur) {
                    nums[i] = nums[cur]
                }
            }
        }
        return i + 1
    }

    fun removeDuplicatesLeavingAtMostTwice(nums: IntArray): Int {
        if (nums.size <= 2) return nums.size
        var i = 1
        for (cur in 2 until nums.size) {
            if (nums[cur] != nums[i - 1]) {
                // either find a new number, or same number not repeated yet
                i++
                if (i != cur) {
                    nums[i] = nums[cur]
                }
            }
        }
        return i + 1
    }

    fun removeElement(nums: IntArray, `val`: Int): Int {
//        var i = -1
//        for (cur in nums.indices) {
//            if (nums[cur] != `val`) {
//                i++
//                if (i != cur) {
//                    nums[i] = nums[cur]
//                }
//            }
//        }
//        return i + 1

        var start = 0
        var end = nums.size - 1
        while (start <= end) {
            if (nums[start] == `val`) {
                if (nums[end] != `val`) {
                    nums[start] = nums[end]
                    start++
                    end--
                } else {
                    end--
                }
            } else {
                start++
            }
        }
        println(nums.joinToString())
        return start
    }

    fun searchInsert(nums: IntArray, target: Int): Int {
        var start = 0
        var end = nums.size - 1
        while (start <= end) {
            val mid = start + (end - start) / 2
            when {
                target == nums[mid] -> return mid
                target > nums[mid] -> start = mid + 1
                else -> end = mid - 1
            }
        }
        return start
    }

    fun plusOne(digits: IntArray): IntArray {
        val n = digits.size
        for (i in n - 1 downTo 0) {
            if (digits[i] < 9) {
                digits[i]++
                return digits
            }
            digits[i] = 0
        }
        // If all digits were 9, create a new array with an additional element
        val result = IntArray(n + 1)
        result[0] = 1
        return result
    }

    fun plusOneAtIndex(digits: IntArray, index: Int): IntArray {
        if (index < 0) {
            val newArray = IntArray(digits.size + 1)
            newArray[0] = 1
            System.arraycopy(digits, 0, newArray, 1, digits.size)
            return newArray
        }
        return if (digits[index] == 9) {
            digits[index] = 0
            plusOneAtIndex(digits, index - 1)
        } else {
            digits[index]++
            digits
        }
    }

    fun merge(nums1: IntArray, m: Int, nums2: IntArray, n: Int): Unit {
        var i = m - 1
        var j = n - 1
        for (k in m + n - 1 downTo 0) {
            when {
                i < 0 || (j >= 0 && nums1[i] < nums2[j]) -> {
                    nums1[k] = nums2[j--]
                }

                else -> {
                    nums1[k] = nums1[i--]
                }
            }
        }
    }

    fun intersectionUnique(nums1: IntArray, nums2: IntArray): IntArray {
        val result = HashSet<Int>()
        nums1.sort()
        nums2.sort()
        var i = 0
        var j = 0
        while (i < nums1.size && j < nums2.size) {
            when {
                nums1[i] < nums2[j] -> {
                    i++
                }

                nums1[i] > nums2[j] -> {
                    j++
                }

                else -> {
                    result.add(nums1[i])
                    i++
                    j++
                }
            }
        }
        return result.toIntArray()
    }

    fun intersect(nums1: IntArray, nums2: IntArray): IntArray {
        val map = mutableMapOf<Int, Int>()
        val result = mutableListOf<Int>()
        for (num in nums1) {
            map[num] = map.getOrDefault(num, 0) + 1
        }
        for (num in nums2) {
            if (map.containsKey(num) && map[num]!! > 0) {
                result.add(num)
                map[num] = map[num]!! - 1
            }
        }
        return result.toIntArray()
    }

    fun getMinimumCommon(nums1: IntArray, nums2: IntArray): Int {
//        var i = 0
//        var j = 0
//        while (i < nums1.size && j < nums2.size) {
//            when {
//                nums1[i] < nums2[j] -> {
//                    i++
//                }
//                nums1[i] > nums2[j] -> {
//                    j++
//                }
//                else -> {
//                    return nums1[i]
//                }
//            }
//        }
//        return -1


        for (n in nums1) {
            val index = binarySearch(nums2, n)
            if (index != -1) {
                return n
            }
        }
        return -1
    }

    fun binarySearch(nums: IntArray, target: Int): Int {
        var start = 0
        var end = nums.size - 1
        while (start <= end) {
            val mid = start + (end - start) / 2
            when {
                target < nums[mid] -> end = mid - 1
                target > nums[mid] -> start = mid + 1
                target == nums[mid] -> return mid
            }
        }
        return -1
    }

    fun searchMatrix(matrix: Array<IntArray>, target: Int): Boolean {
        if (matrix.isEmpty() || matrix[0].isEmpty()) return false
        val cols = matrix[0].size
        var start = 0
        var end = matrix.size * cols - 1
        var mid: Int
        var i: Int
        var j: Int
        while (start <= end) {
            mid = start + (end - start) / 2
            i = mid / cols
            j = mid % cols
            when {
                target == matrix[i][j] -> return true
                target > matrix[i][j] -> start = mid + 1
                else -> end = mid - 1
            }
        }
        return false
    }

    fun totalMaxFrequencyElements(nums: IntArray): Int {
        val frequencyMap = HashMap<Int, Int>() // (value, frequency)
        val totalMap = HashMap<Int, Int>() // (frequency, total count)
        var maxFrequency = 0
        nums.forEach { n ->
            frequencyMap[n] = frequencyMap.getOrDefault(n, 0) + 1
            if (frequencyMap[n]!! > maxFrequency) {
                maxFrequency = frequencyMap[n]!!
            }
            totalMap[frequencyMap[n]!!] = totalMap.getOrDefault(frequencyMap[n], 0) + 1
            if (totalMap.containsKey(frequencyMap[n]!! - 1)) {
                totalMap[frequencyMap[n]!! - 1] = totalMap[frequencyMap[n]!! - 1]!! - 1
            }
        }
        return totalMap[maxFrequency]!! * maxFrequency

//        val frequencyMap = HashMap<Int, Int>() // (value, frequency)
//        var maxFrequency = 0
//
//        for (n in nums) {
//            frequencyMap[n] = frequencyMap.getOrDefault(n, 0) + 1
//            maxFrequency = maxOf(maxFrequency, frequencyMap[n]!!)
//        }
//
//        return frequencyMap.values.count { it == maxFrequency } * maxFrequency
    }

    fun numSubarraysWithSum(nums: IntArray, goal: Int): Int { // binary array nums
//        var result = 0
//        var sum = 0
//        for (start in nums.indices) {
//            sum = nums[start]
//            if (sum == goal) result++
//            for (end in start + 1 until nums.size) {
//                sum += nums[end]
//                if (sum > goal) break
//                else if (sum == goal) result++
//            }
//        }
//        return result

        var start = 0
        var prefixZeros = 0
        var currentSum = 0
        var totalCount = 0

        // Loop through the array using end pointer
        for (end in nums.indices) {
            // Add current element to the sum
            currentSum += nums[end]

            // Slide the window while condition is met
            while (start < end && (nums[start] == 0 || currentSum > goal)) {
                if (nums[start] == 1) {
                    prefixZeros = 0
                } else {
                    prefixZeros++
                }

                currentSum -= nums[start]
                start++
            }

            // Count subarrays when window sum matches the goal
            if (currentSum == goal) {
                totalCount += 1 + prefixZeros
            }
        }

        return totalCount
    }

    fun productExceptSelf(nums: IntArray): IntArray {
//        var totalProduct = 1
//        var countOfZero = 0
//        var exceptZeroProduct = 0
//        for (n in nums) {
//            if (n == 0) {
//                countOfZero++
//                if (countOfZero == 1) {
//                    exceptZeroProduct = totalProduct
//                    totalProduct = 0
//                } else {
//                    totalProduct = 0
//                    break
//                }
//            } else {
//                totalProduct *= n
//                exceptZeroProduct *= n
//            }
//        }
//        val result = IntArray(nums.size)
//        for (n in nums.indices) {
//            when {
//                countOfZero > 1 -> {
//                    result[n] = 0
//                }
//                nums[n] == 0 -> {
//                    result[n] = exceptZeroProduct
//                }
//                else -> {
//                    result[n] = totalProduct / nums[n]
//                }
//            }
//        }

        val n = nums.size
        val leftProduct = IntArray(n) { 1 }
        val rightProduct = IntArray(n) { 1 }
        val result = IntArray(n)

        // Calculate the product of all elements to the left of each element
        var left = 1
        for (i in 1 until n) {
            left *= nums[i - 1]
            leftProduct[i] = left
        }

        // Calculate the product of all elements to the right of each element
        var right = 1
        for (i in n - 2 downTo 0) {
            right *= nums[i + 1]
            rightProduct[i] = right
        }

        // Multiply the left and right products to get the final result
        for (i in 0 until n) {
            result[i] = leftProduct[i] * rightProduct[i]
        }

        return result
    }

    fun calculatePrefixSum(nums: IntArray): IntArray {
        val n = nums.size
        val prefixSum = IntArray(n)
        prefixSum[0] = nums[0]
        for (i in 1 until n) {
            prefixSum[i] = prefixSum[i - 1] + nums[i]
        }
        return prefixSum
    }

    fun findMaxEqualLength(nums: IntArray): Int {
//        var maxEqual = 0
//        var start = 0
//        val n = nums.size
//        val prefixSum = IntArray(n)
//        prefixSum[0] = if (nums[0] == 0) -1 else 1
//        for (i in 1 until n) {
//            prefixSum[i] = prefixSum[i - 1] + if (nums[i] == 0) -1 else 1
//        }
//        for (end in prefixSum.indices) {
//            start = 0
//            while (start < end) {
//                if (start == 0) {
//                    if (prefixSum[end] == 0) {
//                        maxEqual = max(maxEqual, end + 1)
//                        break
//                    }
//                } else if (prefixSum[end] - prefixSum[start - 1] == 0) {
//                    maxEqual = max(maxEqual, end - start + 1)
//                    break
//                }
//                start++
//            }
//        }
//        return maxEqual

        val prefixSumToIndex = HashMap<Int, Int>()
        prefixSumToIndex[0] = -1 // Initialize with 0 prefix sum at index -1
        var maxLen = 0
        var prefixSum = 0

        for (i in nums.indices) {
            // Increment 1 for 1, decrement 1 for 0
            prefixSum += if (nums[i] == 1) 1 else -1

            // If the prefix sum exists in the map, update the max length
            if (prefixSumToIndex.containsKey(prefixSum)) {
                maxLen = maxOf(maxLen, i - prefixSumToIndex[prefixSum]!!)
            } else {
                prefixSumToIndex[prefixSum] = i
            }
        }
        return maxLen
    }

    fun insert(intervals: Array<IntArray>, newInterval: IntArray): Array<IntArray> {
        val result = ArrayList<IntArray>()
        var i = 0
        val size = intervals.size

        // intervals totally before new interval
        while (i < size && intervals[i][1] < newInterval[0]) {
            result.add(intervals[i]);
            i++;
        }

        // overlapping
        while (i < size && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])
            i++;
        }
        result.add(newInterval)

        // intervals totally after new interval
        while (i < size) {
            result.add(intervals[i]);
            i++;
        }
        return result.toTypedArray()
    }

    fun merge(intervals: Array<IntArray>): Array<IntArray> {
        intervals.sortBy { it[0] }
        val list = mutableListOf<IntArray>()
        val size = intervals.size
        var pending = intervals[0]
        var index = 1
        while (index < size) {
            if (pending[1] < intervals[index][0]) {
                list.add(pending)
                pending = intervals[index]
                index++
            } else {
//                pending[0] = minOf(pending[0], intervals[index][0])
                pending[1] = maxOf(pending[1], intervals[index][1])
                index++
            }
        }
        list.add(pending)
        return list.toTypedArray()
    }

    fun findMinArrowShots(points: Array<IntArray>): Int {
//        points.sortWith(compareBy<IntArray> { it[0] }.thenBy { it[1] })
//        var end = points[0][1]
//        var arrows = 0
//        for (i in 1 until points.size) {
//            if (points[i][0] > end) {
//                arrows++
//                end = points[i][1]
//            } else {
//                end = minOf(end, points[i][1])
//            }
//        }
//        return arrows + 1
        points.sortWith { o1, o2 ->
            when {
                o1[0] < o2[0] -> -1
                o1[0] > o2[0] -> 1
                else -> {
                    when {
                        o1[1] < o2[1] -> -1
                        o1[1] > o2[1] -> 1
                        else -> 0
                    }
                }
            }
        }
        var result = 0
        var farest: Int? = null
        for (n in points.indices) {
            // already covered
            if (farest != null && points[n][0] <= farest) {
                farest = min(farest, points[n][1])
                continue
            }
            result++
            farest = points[n][1]
        }

        return result
    }

    fun findDisappearedNumbers(nums: IntArray): List<Int> {
        val result = mutableListOf<Int>()

        // problem: extra space
//        val numsSet = nums.toHashSet()
//        for (i in 1..nums.size) {
//            if (!numsSet.contains(i)) {
//                result.add(i)
//            }
//        }

        // removing takes too long
//        for (i in 1..nums.size) {
//            result.add(i)
//        }
//        for (n in nums) {
//            result.remove(n)
//        }

        // use num as index, in-place mark index-based numbers as negative
        // so those not marked are still positive, which are missing
        for (num in nums) {
            val index = abs(num) - 1
            if (nums[index] > 0) {
                nums[index] *= -1
            }
        }
        for (i in nums.indices) {
            if (nums[i] > 0) {
                result.add(i + 1)
            }
        }

        return result
    }

    fun leastInterval(tasks: CharArray, n: Int): Int {
//        val coolDownAt = HashMap<Char, Int>()
//        val taskCount = HashMap<Char, Int>()
//        for (task in tasks) {
//            taskCount[task] = taskCount.getOrDefault(task, 0) + 1
//        }
//
//        val taskQueue = taskCount.keys.sortedByDescending { taskCount[it] }.toMutableList()
//        var roundIndex = 0
//        var curr: Char? = null
//        var i = 0
//        while (taskQueue.isNotEmpty()) {
//            curr = taskQueue[i]
//            if (!coolDownAt.containsKey(curr) || roundIndex - coolDownAt[curr]!! > n) {
//                // task not in cd or cd time reached, do this task, record cd index
//                taskCount[curr] = taskCount[curr]!! - 1
//                println(curr)
//                coolDownAt[curr] = roundIndex
//                roundIndex++
//
//                if (taskCount[curr] == 0) {
//                    taskQueue.removeAt(i)
//                }
//                while (i + 1 < taskQueue.size && taskCount[taskQueue[i + 1]]!! > taskCount[taskQueue[i]]!!) {
//                    // bubble down sort this task
//                    taskQueue[i] = taskQueue.set(i + 1, taskQueue[i]) // swap
//                    i++
//                }
//                i = 0
//            } else {
//                // task in cd, check next, if reached end, round++, reset index
//                i++
//                if (i >= taskQueue.size) {
//                    println("null")
//                    roundIndex++
//                    i = 0
//                }
//            }
//        }
//        return roundIndex

        var maxFreq = 0
        val taskCount = HashMap<Char, Int>()
        val totalTasks = tasks.size
        for (task in tasks) {
            taskCount[task] = taskCount.getOrDefault(task, 0) + 1
            maxFreq = max(maxFreq, taskCount[task]!!)
        }
        val maxFreqOccur = taskCount.filter { it.value == maxFreq }.size
        return max((maxFreq - 1) * (n + 1) + maxFreqOccur, tasks.size)
    }

    fun findDuplicate(nums: IntArray): Int {
//        val map = HashSet<Int>(nums.size)
//        for (num in nums) {
//            if (map.contains(num)) {
//                return num
//            }
//            map.add(num)
//        }
//        return 0

        // Floyd's cycle detection algorithm
        // two pointers to find the meeting point
        var slow = 0
        var fast = 0
        while (true) {
            slow = nums[slow]
            fast = nums[nums[fast]]
            if (slow == fast) break
        }
        // reset slow to beginning, fast move 1 step now
        slow = 0
        while (slow != fast) {
            slow = nums[slow]
            fast = nums[fast]
        }
        // when they meet again, it's the loop point
        return slow
    }

    fun findDuplicates(nums: IntArray): List<Int> {
        val result = mutableListOf<Int>()
        var index = 0
        for (num in nums) {
            index = abs(num) - 1
            if (nums[index] > 0) {
                nums[index] *= -1
            } else {
                result.add(abs(num))
            }
        }
        return result
    }

    fun findMissingAndRepeatedValues(grid: Array<IntArray>): IntArray {
        val n = grid.size
        val result = intArrayOf(0, 0) // duplicated, missing

        for (i in 0 until n) {
            for (j in 0 until n) {
                val index = abs(grid[i][j]) - 1
                val r = index / n
                val c = index % n
                if (grid[r][c] < 0) {
                    result[0] = abs(grid[i][j])
                } else {
                    grid[r][c] *= -1
                }
            }
        }
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (grid[i][j] > 0) {
                    result[1] = i * n + j + 1
                    return result
                }
            }
        }
        return result
    }

    fun arrayStringsAreEqual(word1: Array<String>, word2: Array<String>): Boolean {
        var p1 = 0
        var p2 = 0
        var in_p1 = 0
        var in_p2 = 0
        while (p1 < word1.size && p2 < word2.size) {
            val s1 = word1[p1]
            val s2 = word2[p2]
            while (in_p1 < s1.length && in_p2 < s2.length) {
                if (s1[p1] != s2[p2]) {
                    return false
                }
                in_p1++
                in_p2++
            }

            if (in_p1 == s1.length) {
                p1++
                in_p1 = 0
            }
            if (in_p2 == s2.length) {
                p2++
                in_p2 = 0
            }
        }
        if (p1 != word1.size || p2 != word2.size) {
            return false
        }
        return true
    }

    fun numSubarrayProductLessThanK(nums: IntArray, k: Int): Int {
        // prefix product technique will potentially int overflow or consume much more space
        // use sliding window
        var result = 0
        var left = 0
        var product = 1
        for (right in nums.indices) {
            product *= nums[right]
            while (left <= right && product >= k) {
                product /= nums[left]
                left++
            }
            result += right - left + 1
        }

        return result
    }

    fun maxFreqGoodSubarrayLength(nums: IntArray, k: Int): Int {
        // good: freq less than or equal to k
        val freqMap = mutableMapOf<Int, Int>()
        var maxLength = 0
        var start = 0
        for (end in nums.indices) {
            freqMap[nums[end]] = freqMap.getOrDefault(nums[end], 0) + 1

            while (start < end && freqMap[nums[end]]!! > k) {
                freqMap[nums[start]] = freqMap.getOrDefault(nums[start], 1) - 1
                start++
            }
            maxLength = max(maxLength, end - start + 1)
        }
        return maxLength
    }

    fun countSubarraysLargestAppearsAtLeastK(nums: IntArray, k: Int): Long {
        // find the largest number in the array
        // then check the subarray which contains the largest number K times
        var maxNum = 0
        for (num in nums) {
            maxNum = maxOf(maxNum, num)
        }
        var count = 0L
        var appears = 0
        var left = 0
        for (right in nums.indices) {
            if (nums[right] == maxNum) {
                appears++
            }
            while (appears >= k) {
                if (nums[left] == maxNum) {
                    appears--
                }
                left++
            }
            count += left
        }

        // count在left移动途中累加，操作次数也是n次，效率一样
//        for (right in nums.indices) {
//            if (nums[right] == maxNum) {
//                appears++
//            }
//            while (appears >= k) {
//                count += nums.size - right
//                if (nums[left] == maxNum) {
//                    appears--
//                }
//                left++
//            }
//        }

        // 固定left，扩展right
//        var right = 0
//        for (left in nums.indices) {
//            while (right < nums.size && appears < k) {
//                if (nums[right] == maxNum) {
//                    appears++
//                }
//                right++
//            }
//            if (appears >= k) {
//                count += nums.size - right + 1
//            }
//            if (nums[left] == maxNum) {
//                appears--
//            }
//        }
        return count
    }

    fun subarraysWithAtMostKDistinct(nums: IntArray, k: Int): Int {
        // good: distinct integers number is at most k.
        if (k == 0) return 0
        val freqMap = mutableMapOf<Int, Int>()
        var result = 0
        var distinct = 0
        var start = 0
        for (end in nums.indices) {
            if (freqMap.getOrDefault(nums[end], 0) == 0) {
                distinct++
            }
            freqMap[nums[end]] = freqMap.getOrDefault(nums[end], 0) + 1

            while (distinct > k) {
                freqMap[nums[start]] = freqMap.getOrDefault(nums[start], 1) - 1
                if (freqMap.getOrDefault(nums[start], 0) == 0) {
                    distinct--
                }
                start++
            }
            result += end - start + 1
        }

        return result
    }

    fun countStudents(students: IntArray, sandwiches: IntArray): Int {
        val size = students.size
        // if the number of students who like 0 is equal to the number of sandwiches with 0, all students can get the sandwich
        val studentZeros = students.count { it == 0 }
        val sandwichZeros = sandwiches.count { it == 0 }
        if (studentZeros == sandwichZeros) return 0
        // we check which type of students can all get the sandwich
        val typeToCheck = if (studentZeros < sandwichZeros) 0 else 1
        var countToCheck = if (studentZeros < sandwichZeros) studentZeros else size - studentZeros
        sandwiches.forEachIndexed { index, sandwich ->
            // if count reaches 0, and current sandwich is the type to check, it means all rest students can't get the sandwich
            if (countToCheck == 0 && sandwich == typeToCheck) {
                return size - index
            }
            if (sandwich == typeToCheck) {
                countToCheck--
            }
        }
        return 0
    }

    fun numberOfGoodSubarraySplits(nums: IntArray): Int {
        // good if it contains exactly one element with the value 1.
        // [0,1,0,0,1] -> [0,1] [0,0,1] && [0,1,0] [0,1] && [0,1,0,0] [1]
        val modulo = (1e9 + 7).toInt()
        var count = 1L
        var lastOneIndex = -1
        nums.forEachIndexed { index, num ->
            if (num == 1) {
                if (lastOneIndex != -1) {
                    count *= index - lastOneIndex
                    if (count > modulo) {
                        count %= modulo
                    }
                }
                lastOneIndex = index
            }
        }
        if (lastOneIndex == -1) return 0
        return count.toInt()
    }

    fun timeRequiredToBuy(tickets: IntArray, k: Int): Int {
//        var count = 0
//        var i = 0
//        while (true) {
//            if (tickets[i] > 0) {
//                tickets[i]--
//                count++
//            }
//            if (tickets[k] == 0) {
//                break
//            }
//            i++
//            if (i >= tickets.size) {
//                i = 0
//            }
//        }
//        return count

        var count = 0
        for (i in tickets.indices) {
            count += if (i <= k) {
                min(tickets[i], tickets[k])
            } else {
                min(tickets[i], tickets[k] - 1)
            }
        }
        return count
    }

    fun deckRevealedIncreasing(deck: IntArray): IntArray {
        val indexList = mutableListOf<Int>()
        for (i in deck.indices) {
            indexList.add(i)
        }
        for (i in 0 until indexList.size - 1) {
            indexList.add(indexList.removeAt(i + 1))
        }
        val result = IntArray(deck.size)
        deck.sort()
        for (i in deck.indices) {
            result[indexList[i]] = deck[i]
        }
        return result
    }

    fun majorityElement(nums: IntArray): Int {
//        val freqMap = mutableMapOf<Int, Int>()
//        var mostElement = nums[0]
//        var mostFreq = 0
//        for (i in nums.indices) {
//            freqMap[nums[i]] = freqMap.getOrDefault(nums[i], 0) + 1
//            if (freqMap[nums[i]]!! > mostFreq) {
//                mostFreq = freqMap[nums[i]]!!
//                mostElement = nums[i]
//            }
//            if (freqMap[nums[i]]!! > nums.size / 2) break
//        }
//        return mostElement

        // Boyer-Moore Voting Algorithm
        var candidate = nums[0]
        var count = 0
        for (num in nums) {
            if (count == 0) {
                candidate = num
            }
            if (num == candidate) {
                count++
            } else {
                count--
            }
        }
        return candidate
    }

    fun canCompleteCircuit(gas: IntArray, cost: IntArray): Int {
        // brute force
//        val length = gas.size
//        val delta = IntArray(length) { 0 }
//        for (i in gas.indices) {
//            delta[i] = gas[i] - cost[i]
//        }
//        var result = 0
//        var index = 0
//        var lastBalance = 0
//        var valid = true
//        while (result in delta.indices) {
//            if (delta[result] < 0) {
//                result++
//                continue
//            }
//            index = result
//            lastBalance = 0
//            valid = true
//            for (i in 0 until length) {
//                lastBalance += delta[index]
//                if (lastBalance < 0) {
//                    valid = false
//                    break
//                }
//                index++
//                if (index == length) {
//                    index = 0
//                }
//            }
//            if (!valid) {
//                result++
//            } else {
//                break
//            }
//        }
//        return if (result == length) -1 else result

        var totalBalance = 0
        var currentBalance = 0
        var startIndex = 0
        for (i in gas.indices) {
            val delta = gas[i] - cost[i]
            totalBalance += delta
            currentBalance += delta

            if (currentBalance < 0) {
                currentBalance = 0
                startIndex = i + 1
            }
        }
        return if (totalBalance < 0) -1 else startIndex % gas.size
    }

    fun maxProfit(prices: IntArray): Int {
        var maxProfit = 0
        var minPrice = prices[0]
        for (i in 1 until prices.size) {
            maxProfit = max(prices[i] - minPrice, maxProfit)
            minPrice = min(prices[i], minPrice)
        }
        return maxProfit
    }

    fun maxProfitMultiGreedy(prices: IntArray): Int {
        var totalProfit = 0
        var minPrice = prices[0]
        var maxProfit = 0
        var newProfit = 0
        for (i in 1 until prices.size) {
            minPrice = min(prices[i], minPrice)
            newProfit = prices[i] - minPrice
            if (newProfit > maxProfit) {
                maxProfit = newProfit
            } else {
                totalProfit += maxProfit
                minPrice = prices[i]
                maxProfit = 0
            }
        }
        if (maxProfit != 0) {
            totalProfit += maxProfit
        }
        return totalProfit
    }

    fun maxProfitMultiDP(prices: IntArray): Int {
        var noStockProfit = 0        // 不持有股票时的利润，最开始未进行任何交易
        var holdProfit = -prices[0]  // 持有股票时的利润，最开始买入第一天的股票

        // 状态转移
        for (i in 1 until prices.size) {
            // 计算新的不持有股票的最大利润（卖出股票或者保持不持有状态）
            noStockProfit = maxOf(noStockProfit, holdProfit + prices[i])
            // 计算新的持有股票的最大利润（买入股票或者保持持有状态）
            holdProfit = maxOf(holdProfit, noStockProfit - prices[i])
        }

        return noStockProfit
    }

    fun rob(nums: IntArray): Int {
        if (nums.size == 1) return nums[0]
        // keep max rob at n - 2 and n - 1
        val previous = intArrayOf(nums[0], maxOf(nums[1], nums[0]))
        var current = 0
        for (i in 2 until nums.size) {
            // comparing n-1 and n-2 plus current
            current = maxOf(previous[0] + nums[i], previous[1])
            previous[0] = previous[1]
            previous[1] = current
        }
        return previous[1]
    }

    class RandomizedSet() {
        val list = ArrayList<Int>()
        val map = HashMap<Int, Int>()

        fun insert(`val`: Int): Boolean {
            if (map.containsKey(`val`)) return false
            list.add(`val`)
            map[`val`] = list.size - 1
            return true
        }

        fun remove(`val`: Int): Boolean {
            if (map.containsKey(`val`).not()) return false
            val index = map[`val`]!!
            list[index] = list.last()
            map.remove(`val`)
            map[list[index]] = index
            list.removeLast()
            return true
        }

        fun getRandom(): Int {
            return list[(0..list.lastIndex).random()]
        }
    }

    fun numRescueBoats(people: IntArray, limit: Int): Int {
        var boat = 0
        people.sortDescending()
        var start = 0
        var end = people.size - 1
        while (start < end) {
            if (people[start] + people[end] <= limit) {
                boat++
                start++
                end--
            } else {
                boat++
                start++
            }
        }
        if (start == end) boat++
        return boat
    }

    fun canJump(nums: IntArray): Boolean {
        // bfs
//        val lastIndex = nums.lastIndex
//        if (lastIndex == 0) return true
//        val queue = ArrayDeque<Int>()
//        queue.add(0)
//        var currentIndex = 0
//        var nextIndex = 0
//        while (queue.isNotEmpty()) {
//            currentIndex = queue.removeFirst()
//            if (nums[currentIndex] == 0) continue
//            (1..abs(nums[currentIndex])).forEach { step ->
//                nextIndex = currentIndex + step
//                if (nextIndex == lastIndex) return true
//                if (nums[nextIndex] > 0) {
//                    queue.addLast(nextIndex)
//                    nums[nextIndex] *= -1
//                }
//            }
//        }
//        return false
        if (nums.lastIndex == 0) return true
        var farthest = 0
        for (i in nums.indices) {
            if (farthest == i && nums[i] == 0) return false
            farthest = maxOf(farthest, i + nums[i])
            if (farthest >= nums.lastIndex) return true
        }
        return false
    }

    fun jumpMinToEnd(nums: IntArray): Int {
//        if (nums.lastIndex == 0) return 0
//        val stepsToReach = IntArray(nums.size) { Int.MAX_VALUE }
//        stepsToReach[0] = 0
//        for (i in 0 until nums.lastIndex) {
//            val canGo = nums[i]
//            (1..canGo).forEach { step ->
//                // update when smaller
//                stepsToReach[i + step] = minOf(stepsToReach[i + step], stepsToReach[i] + 1)
//                if (i + step == stepsToReach.lastIndex) return stepsToReach.last()
//            }
//        }
//        return stepsToReach.last()

        if (nums.lastIndex == 0) return 0
        var farthest = 0
        var jumps = 0
        var currentEnd = 0 // the farthest index that can be reached in the current number of jumps
        for (i in 0 until nums.lastIndex) {
            farthest = maxOf(farthest, i + nums[i])
            if (farthest >= nums.lastIndex) {
                jumps++
                break
            }
            if (i == currentEnd) {
                jumps++
                currentEnd = farthest
            }
        }
        return jumps
    }

    fun coinChange(coins: IntArray, amount: Int): Int {
        // bfs
//        if (amount == 0) return 0
//        val amountCoinMap = mutableMapOf<Int, Int>()
//        val queue = ArrayDeque<Int>()
//        queue.add(amount)
//        amountCoinMap[amount] = 0
//        var restAmount: Int
//        while (queue.isNotEmpty()) {
//            restAmount = queue.removeFirst()
//            for (coin in coins) {
//                if (restAmount < coin) continue
//                val newRest = restAmount - coin
//                if (newRest == 0) return amountCoinMap[restAmount]!! + 1
//                if (!amountCoinMap.containsKey(newRest)) {
//                    amountCoinMap[newRest] = amountCoinMap[restAmount]!! + 1
//                    queue.addLast(newRest)
//                }
//            }
//        }
//        return -1

        // minimum number of coins needed to make up the amount i
        val dp = IntArray(amount + 1) { amount + 1 }
        dp[0] = 0
        for (i in 1..amount) {
            for (coin in coins) {
                if (coin <= i) {
                    dp[i] = minOf(dp[i], dp[i - coin] + 1)
                }
            }
        }
        return if (dp[amount] > amount) -1 else dp[amount]
    }

    fun minimumAddedCoins(coins: IntArray, target: Int): Int {
//        val targets = BooleanArray(target + 1)
//        targets[0] = true
//        for (coin in coins) {
//            for (i in target downTo coin) {
//                if (targets[i - coin]) {
//                    targets[i] = true
//                }
//            }
//        }
//
//        var additionalCoins = 0
//        for (i in 1..target) {
//            if (!targets[i]) {
//                additionalCoins++
//                for (j in target downTo i) {
//                    if (targets[j - i]) {
//                        targets[j] = true
//                    }
//                }
//            }
//        }
//        return additionalCoins
        coins.sort()
        var maxReach = 0
        var additionalCoins = 0

        for (coin in coins) {
            while (coin > maxReach + 1 && maxReach < target) {
                additionalCoins++
                maxReach += maxReach + 1
            }
            if (maxReach >= target) break
            maxReach += coin
        }

        while (maxReach < target) {
            additionalCoins++
            maxReach += maxReach + 1
        }

        return additionalCoins
    }

    fun lengthOfLIS(nums: IntArray): Int {
        // Longest Increasing Subsequence
        // keep the end value of each length
        val dp = IntArray(nums.size + 1) { Int.MIN_VALUE }
        var max = 0
        for (num in nums) {
            if (num > dp[max]) {
                max++
                dp[max] = num
            } else {
                var i = dp.binarySearch(num, 0, max)
                if (i < 0) {
                    i = -(i + 1) // find the insert point
                }
                dp[i] = num
//
//                for (j in max - 1 downTo 0) {
//                    if (num > dp[j]) {
//                        dp[j + 1] = num
//                        break
//                    }
//                }
            }
        }
        return max
    }

    fun findRelativeRanks(score: IntArray): Array<String> {
        // indexOf
//        val result = Array(score.size) { "" }
//        val sortedScore = score.clone()
//        sortedScore.sortDescending()
//        for (i in score.indices) {
//            sortedScore.indexOf(score[i]).let {
//                result[i] = when (it) {
//                    0 -> "Gold Medal"
//                    1 -> "Silver Medal"
//                    2 -> "Bronze Medal"
//                    else -> (it + 1).toString()
//                }
//            }
//        }
//        return result

        // map to store index
        val result = Array(score.size) { "" }
        val indexMap = mutableMapOf<Int, Int>()
        for (i in score.indices) {
            indexMap[score[i]] = i
        }
        score.sortDescending()
        for (i in score.indices) {
            result[indexMap[score[i]]!!] = when (i) {
                0 -> "Gold Medal"
                1 -> "Silver Medal"
                2 -> "Bronze Medal"
                else -> (i + 1).toString()
            }
        }
        return result
    }

    fun maximumHappinessSum(happiness: IntArray, k: Int): Long {
        happiness.sortDescending()
        var result = 0L
        for (i in 0 until k) {
            result += maxOf(happiness[i] - i, 0)
        }
        return result
    }

    fun kthSmallestPrimeFraction(arr: IntArray, k: Int): IntArray {
        val heap = PriorityQueue<IntArray> { o1, o2 ->
            (o2[0].toDouble() / o2[1]).compareTo(o1[0].toDouble() / o1[1])
        }
        for (i in arr.indices) {
            for (j in i + 1 until arr.size) {
                if (heap.size < k) {
                    heap.offer(intArrayOf(arr[i], arr[j]))
                } else if (heap.peek()[0].toDouble() / heap.peek()[1] > arr[i].toDouble() / arr[j]) {
                    heap.poll()
                    heap.offer(intArrayOf(arr[i], arr[j]))
                }
            }
        }
        return heap.peek()
    }

    fun containsNearbyDuplicate(nums: IntArray, k: Int): Boolean {
        val valueIndexMap = mutableMapOf<Int, Int>() // last index is enough
        for (i in nums.indices) {
            if (valueIndexMap.containsKey(nums[i])) {
                if (i - valueIndexMap[nums[i]]!! <= k) return true
                else valueIndexMap[nums[i]] = i
            } else {
                valueIndexMap[nums[i]] = i
            }
        }
        return false
    }

    fun subsetXORSum(nums: IntArray): Int {
        // 所有子集的XOR和之和等价于对每一位单独计算贡献，然后累加。（XOR操作的分配性和交换性）
        // 每个元素出现在所有子集中的概率是1/2
        // 按位OR操作将所有数组中的1位都合并起来
        var orResult = 0
        for (num in nums) {
            orResult = orResult or num
        }
        return orResult shl (nums.size - 1)
    }

    fun subsets(nums: IntArray): List<List<Int>> {
        val n = nums.size
        val result = mutableListOf<MutableList<Int>>()
        for (i in 0 until (1 shl n)) {
            val currentList = mutableListOf<Int>()
            for (j in 0 until n) {
                if ((i and (1 shl j)) != 0) {
                    currentList.add(nums[j])
                }
            }
            result.add(currentList)
        }
        return result
    }

    fun subsetsRecursively(nums: IntArray): List<List<Int>> {
        val result = mutableListOf<MutableList<Int>>()
        val currentList = mutableListOf<Int>()

        fun dfs(index: Int) {
            if (index == nums.size) {
                result.add(ArrayList(currentList)) // prevent modifying
                return
            }

            // exclude current integer
            dfs(index + 1)

            // include current integer
            currentList.add(nums[index])
            dfs(index + 1)
            currentList.removeLast() // backtracking
        }

        dfs(0)
        return result
    }

    fun beautifulSubsetsCount(nums: IntArray, k: Int): Int {
        // not contain two integers with an absolute difference equal to k
        var count = 0
        val currentList = mutableListOf<Int>()

        fun dfs(index: Int) {
            if (index == nums.size) {
                count++
                return
            }
            // maybe include current integer
            val current = nums[index]
            if (!(currentList.contains(current - k) || currentList.contains(current + k))) {
                currentList.add(current)
                dfs(index + 1)
                // backtracking
                currentList.remove(current)
            }

            // exclude current integer
            dfs(index + 1)
        }

        dfs(0)
        return count - 1
    }

    fun maxScoreWords(words: Array<String>, letters: CharArray, score: IntArray): Int {
        val letterRemains = IntArray(26) { 0 }
        for (letter in letters) {
            letterRemains[letter - 'a']++
        }
        val wordsSize = words.size
        val wordLetterFreq = Array(wordsSize) { IntArray(26) }
        val wordScore = IntArray(wordsSize)
        for (i in words.indices) {
            wordScore[i] = words[i].sumOf { score[it - 'a'] }
            for (letter in words[i]) {
                wordLetterFreq[i][letter - 'a']++
            }
        }

        var currentScore = 0
        var max = 0

        fun dfs(index: Int) {
            if (index == wordsSize) {
                max = maxOf(max, currentScore)
                return
            }

            // case: maybe include this word
            var canAdd = true
            for (i in wordLetterFreq[index].indices) {
                if (wordLetterFreq[index][i] > letterRemains[i]) {
                    canAdd = false
                    break
                }
            }
            if (canAdd) {
                for (i in wordLetterFreq[index].indices) {
                    letterRemains[i] -= wordLetterFreq[index][i]
                }
                currentScore += wordScore[index]
                dfs(index + 1)
                // backtrack
                for (i in wordLetterFreq[index].indices) {
                    letterRemains[i] += wordLetterFreq[index][i]
                }
                currentScore -= wordScore[index]
            }

            // case: don't include this word
            dfs(index + 1)
        }

        dfs(0)
        return max
    }

    fun specialArray(nums: IntArray): Int {
        // there are x elements with value >= x; x is unique.
        nums.sortDescending()
        var valueToTest = -1
        for (i in nums.indices) {
            valueToTest = i + 1
            if (nums[i] >= valueToTest && (i == nums.lastIndex || nums[i + 1] < valueToTest)) {
                return valueToTest
            }
        }
        return -1
    }

    fun minimumTotalTriangle(triangle: List<List<Int>>): Int {
        val height = triangle.size
        val dp = Array(height) { IntArray(triangle[height - 1].size) }
        dp[0][0] = triangle[0][0]

        for (i in 1 until height) {
            for (j in 0..triangle[i].lastIndex) {
                dp[i][j] = when (j) {
                    0 -> dp[i - 1][j]
                    triangle[i].lastIndex -> dp[i - 1][j - 1]
                    else -> minOf(dp[i - 1][j], dp[i - 1][j - 1])
                } + triangle[i][j]
            }
        }
        return dp[height - 1].minOf { it }
    }

    fun countTriplets(arr: IntArray): Int {
        val n = arr.size
        val xorMap = Array(n) { IntArray(n) }
        for (i in 0 until n) {
            for (j in i until n) {
                if (j == i) {
                    xorMap[i][j] = arr[j]
                } else {
                    xorMap[i][j] = xorMap[i][j - 1] xor arr[j]
                }
            }
        }

        var count = 0
        val dp = IntArray(n)
        var currentCount: Int
        for (i in 0 until n) {
            dp[i] = 0
            for (k in i + 1 until n) {
                currentCount = 0
                for (j in i + 1..k) {
                    if (xorMap[i][j - 1] == xorMap[j][k]) {
                        currentCount++
                    }
                }
                dp[k] = dp[k - 1] + currentCount
                if (k == n - 1) {
                    count += dp[k]
                }
            }
        }
        return count
    }

    fun countTripletsPrefix(arr: IntArray): Int {
        val n = arr.size
        val prefixXor = IntArray(n + 1) // 前n个数字的XOR结果
        for (i in 0 until n) {
            prefixXor[i + 1] = prefixXor[i] xor arr[i]
        }

        var count = 0
        for (i in 0 until n) {
            for (k in i + 1 until n) {
                if (prefixXor[i] == prefixXor[k + 1]) {
                    // i-1到k之间所有j都满足两部分
                    count += k - i
                }
            }
        }
        return count
    }

    fun summaryRanges(nums: IntArray): List<String> {
        val size = nums.size
        if (size == 0) return listOf()
        val result = mutableListOf<String>()

        var start = 0
        for (end in 1 until size) {
            if (nums[end] != nums[end - 1] + 1) {
                if (end - 1 != start) {
                    result.add("${nums[start]}->${nums[end - 1]}")
                } else {
                    result.add("${nums[start]}")
                }
                start = end
            }
        }
        if (start != size - 1) {
            result.add("${nums[start]}->${nums[size - 1]}")
        } else {
            result.add("${nums[start]}")
        }
        return result
    }

    fun rotate(nums: IntArray, k: Int): Unit {
        val n = nums.size
        val tmp = IntArray(nums.size)
        var j: Int
        var i = 0
        val modK = k % n
        while (i < modK) {
            j = (i + modK) % n
            tmp[j] = nums[j]
            nums[j] = if (i < modK) nums[i] else tmp[i]
            i++
        }
        while (i < n) {
            j = (i + modK) % n
            tmp[j] = nums[j]
            nums[j] = tmp[i]
            i++
        }

        //        // 辅助函数：翻转数组的某个区间
        //        fun reverse(start: Int, end: Int) {
        //            var i = start
        //            var j = end
        //            while (i < j) {
        //                val tmp = nums[i]
        //                nums[i] = nums[j]
        //                nums[j] = tmp
        //                i++
        //                j--
        //            }
        //        }
        //
        //        // 第一步：翻转整个数组
        //        reverse(0, n - 1)
        //        // 第二步：翻转前 k 个元素
        //        reverse(0, modK - 1)
        //        // 第三步：翻转剩余的元素
        //        reverse(modK, n - 1)
    }

    fun checkSubarraySumIsMultiple(nums: IntArray, k: Int): Boolean {
        val prefixSumToIndex = mutableMapOf<Int, Int>()
        prefixSumToIndex[0] = -1
        var sum = 0
        for (i in nums.indices) {
            sum += nums[i]
            if (prefixSumToIndex.containsKey(sum % k)) {
                if (i - prefixSumToIndex[sum % k]!! > 1) {
                    return true
                }
            } else {
                prefixSumToIndex[sum % k] = i
            }
        }
        return false
    }

    fun subarraysDivByK(nums: IntArray, k: Int): Int {
        val modCount = IntArray(10001)
        modCount[0] = 1
        var sum = 0
        var result = 0
        for (num in nums) {
            sum += num
            val rest = (sum % k + k) % k // 优化模运算，确保结果为非负数
            result += modCount[rest]
            modCount[rest]++
        }
        return result
    }

    fun isNStraightHand(hand: IntArray, groupSize: Int): Boolean {
        if (hand.size % groupSize != 0) return false
        val cardCounts = mutableMapOf<Int, Int>()
        for (card in hand) {
            cardCounts[card] = cardCounts.getOrDefault(card, 0) + 1
        }
        val sortedCards = cardCounts.keys.sorted()
        var newCard: Int
        for (card in sortedCards) {
            val count = cardCounts[card] ?: continue
            if (count > 0) {
                for (i in 1 until groupSize) {
                    newCard = card + i
                    if (cardCounts.getOrDefault(newCard, 0) < count) {
                        return false
                    }
                    cardCounts[newCard] = cardCounts[newCard]!! - count
                }
            }
        }
        return true
    }

    fun heightChecker(heights: IntArray): Int {
        val sorted = heights.sortedArray()
        var count = 0
        for (i in sorted.indices) {
            count += if (sorted[i] != heights[i]) 1 else 0
        }
        return count
    }

    fun relativeSortArray(arr1: IntArray, arr2: IntArray): IntArray {
        val indexMap = arr2.withIndex().associate { it.value to it.index }
//        val sortedGroup = Array(arr2.size) { mutableListOf<Int>() }
//        val pq = PriorityQueue<Int>()
//        arr1.forEach { num ->
//            indexMap[num]?.let { index ->
//                sortedGroup[index].add(num)
//            } ?: pq.add(num)
//        }
//        val result = mutableListOf<Int>()
//        result.addAll(sortedGroup.flatMap { it })
//        while (pq.isNotEmpty()) {
//            result.add(pq.poll())
//        }
//        return result.toIntArray()
        val maxIndex = arr2.size
        return arr1.sortedWith { a, b ->
            val indexA = indexMap[a] ?: maxIndex
            val indexB = indexMap[b] ?: maxIndex
            if (indexA != maxIndex || indexB != maxIndex) {
                indexA.compareTo(indexB)
            } else {
                a.compareTo(b)
            }
        }.toIntArray()
    }

    fun sortColors(nums: IntArray): Unit {
        val colorCount = IntArray(3)
        nums.forEach { color ->
            colorCount[color]++
        }
        var index = 0
        for (i in 0..2) {
            repeat(colorCount[i]) {
                nums[index++] = i
            }
        }
    }

    fun minMovesToSeat(seats: IntArray, students: IntArray): Int {
        seats.sort()
        students.sort()
        var result = 0
        for (i in seats.indices) {
            result += abs(seats[i] - students[i])
        }
        return result
    }

    fun minIncrementForUnique(nums: IntArray): Int {
//        nums.sort()
//        var nextIndex = -1
//        var nextValue = -1
//        var result = 0
//        for (i in 1 until nums.size) {
//            if (nums[i] == nums[i - 1]) {
//                if (nums[i] > nextValue) {
//                    nextIndex = i + 1
//                    nextValue = nums[i] + 1
//                }
//                while (nextIndex < nums.size) {
//                    if (nums[nextIndex] < nextValue) {
//                        nextIndex++
//                    } else if (nums[nextIndex] == nextValue) {
//                        nextValue++
//                        nextIndex++
//                    } else {
//                        break
//                    }
//                }
//                result += nextValue - nums[i]
//                nextValue++
//            }
//        }
//        return result
        nums.sort()
        var result = 0
        var nextValue = nums[0]
        for (i in nums.indices) {
            if (nums[i] < nextValue) {
                result += nextValue - nums[i]
            }
            nextValue = maxOf(nextValue, nums[i]) + 1
        }
        return result
    }

    fun maxProfitAssignment(difficulty: IntArray, profit: IntArray, worker: IntArray): Int {
        val jobs = difficulty.indices
            .map { i -> Pair(difficulty[i], profit[i]) }
            .sortedBy { it.first }
        worker.sort()
        var result = 0
        var currentMaxProfit = 0
        var i = 0
        for (workerAbility in worker) {
            while (i < jobs.size && workerAbility < jobs[i].first) {
                currentMaxProfit = maxOf(currentMaxProfit, jobs[i].second)
                i++
            }
            result += currentMaxProfit
        }
        return result
    }

    fun bouquetsMinDays(bloomDay: IntArray, m: Int, k: Int): Int {
        if (bloomDay.size < m * k) return -1

        fun countBouquets(day: Int): Int {
            var count = 0
            var flowers = 0

            for (bloom in bloomDay) {
                if (bloom <= day) {
                    flowers++
                    if (flowers == k) {
                        count++
                        flowers = 0
                    }
                } else {
                    flowers = 0
                }
            }
            return count
        }

        var start = bloomDay.minOrNull() ?: 0
        var end = bloomDay.maxOrNull() ?: 0
        var minDays = -1
        while (start <= end) {
            val mid = start + (end - start) / 2
            val bouquets = countBouquets(mid)
            if (m <= bouquets) {
                minDays = mid
                end = mid - 1
            } else {
                start = mid + 1
            }
        }
        return minDays
    }

    fun maxDistance(position: IntArray, m: Int): Int {
        position.sort()

        fun canPlace(distance: Int): Boolean {
            var placed = 1
            var lastPosition = position[0]

            for (i in 1 until position.size) {
                if (position[i] - lastPosition >= distance) {
                    placed++
                    lastPosition = position[i]
                }
                if (placed == m) return true
            }
            return false
        }

        var start = 1
        var end = position.last() - position.first()
        var result = 0
        while (start <= end) {
            val mid = start + (end - start) / 2
            if (canPlace(mid)) {
                result = mid
                start = mid + 1
            } else {
                end = mid - 1
            }
        }
        return result
    }

    fun maxSatisfied(customers: IntArray, grumpy: IntArray, minutes: Int): Int {
        var result = 0
        for (i in customers.indices) {
            if (grumpy[i] == 0) {
                result += customers[i]
            }
        }
        var start = 0
        var sum = 0
        var maxSum = 0
        for (end in grumpy.indices) {
            if (grumpy[end] == 1) {
                sum += customers[end]
            }
            if (end >= minutes) {
                if (grumpy[start] == 1) {
                    sum -= customers[start]
                }
                start++
            }
            maxSum = maxOf(maxSum, sum)
        }
        return result + maxSum
    }

    fun numberOfSubarraysOddCount(nums: IntArray, k: Int): Int {
        var oddCount = 0
        var start = 0
        var prefixEvens = 0
        var result = 0
        for (end in nums.indices) {
            oddCount += if (nums[end] % 2 == 0) 0 else 1
            while (start <= end && (nums[start] % 2 == 0 || oddCount > k)) {
                if (nums[start] % 2 != 0) {
                    prefixEvens = 0
                } else {
                    prefixEvens++
                }

                oddCount -= if (nums[start] % 2 == 0) 0 else 1
                start++
            }
            if (oddCount == k) {
                result += 1 + prefixEvens
            }
        }
        return result
    }

    fun longestSubarrayPQ(nums: IntArray, limit: Int): Int {
        val pqMax = PriorityQueue<Int>(reverseOrder())
        val pqMin = PriorityQueue<Int>()
        var start = 0
        var maxLength = 0
        for (end in nums.indices) {
            pqMax.offer(nums[end])
            pqMin.offer(nums[end])
            while (pqMax.peek() - pqMin.peek() > limit) {
                pqMax.remove(nums[start])
                pqMin.remove(nums[start])
                start++
            }
            maxLength = maxOf(maxLength, end - start + 1)
        }
        return maxLength
    }

    fun longestSubarrayDeque(nums: IntArray, limit: Int): Int {
        val maxDeque = LinkedList<Int>()
        val minDeque = LinkedList<Int>()
        var start = 0
        var result = 0

        for (end in nums.indices) {
            while (!maxDeque.isEmpty() && maxDeque.last < nums[end]) {
                maxDeque.removeLast()
            }
            while (!minDeque.isEmpty() && minDeque.last > nums[end]) {
                minDeque.removeLast()
            }
            maxDeque.addLast(nums[end])
            minDeque.addLast(nums[end])

            while (maxDeque.first - minDeque.first > limit) {
                if (nums[start] == maxDeque.first) {
                    maxDeque.removeFirst()
                }
                if (nums[start] == minDeque.first) {
                    minDeque.removeFirst()
                }
                start++
            }

            result = maxOf(result, end - start + 1)
        }

        return result
    }

    fun convertZigzag(s: String, numRows: Int): String {
        if (numRows == 1 || s.length <= numRows) return s
        val result = StringBuilder()
        val distance = 2 * numRows - 2
        for (i in 0 until numRows) {
            var j = 0
            while (j + i < s.length) {
                result.append(s[j + i])
                // not first row, not last row
                if (i != 0 && i != numRows - 1 &&
                    j + distance - i < s.length
                ) {
                    result.append(s[j + distance - i]) // 锁进距离等于行数
                }
                j += distance
            }
        }
        return result.toString()

//        if (numRows == 1 || s.length <= numRows) return s
//        val n = (s.length / (numRows + numRows - 2) + 1) * (numRows - 1)
//        val matrix = Array(numRows) { CharArray(n) { ' ' } }
//        var i = 0
//        var j = 0
//        var back = false
//        s.forEach { char ->
//            matrix[i][j] = char
//            if (!back && i != numRows - 1) {
//                i++
//            } else if (back) {
//                i--
//                j++
//            }
//            if (i == numRows - 1) {
//                back = true
//            } else if (i == 0) {
//                back = false
//            }
//        }
//        val result = StringBuilder()
//        for (u in 0 until numRows) {
//            for (v in 0 until n) {
//                if (matrix[u][v] != ' ') {
//                    result.append(matrix[u][v])
//                }
//            }
//        }
//        return result.toString()
    }

    fun threeSum(nums: IntArray): List<List<Int>> {
        val indexMap = nums.withIndex().associate { it.value to it.index }
        val resultSet = mutableSetOf<List<Int>>()
        for (i in nums.indices) {
            val target = -nums[i]
            for (j in i + 1 until nums.size) {
                if (indexMap.containsKey(target - nums[j]) && indexMap[target - nums[j]]!! > j) {
                    resultSet.add(listOf(nums[i], nums[j], target - nums[j]).sorted())
                }
            }
        }
        return resultSet.toList()
    }

    fun threeSumTwoPointers(nums: IntArray): List<List<Int>> {
        nums.sort()
        val result = mutableListOf<List<Int>>()
        for (i in nums.indices) {
            if (i > 0 && nums[i] == nums[i - 1]) continue
            var j = i + 1
            var k = nums.size - 1
            while (j < k) {
                when {
                    nums[i] + nums[j] + nums[k] < 0 -> j++
                    nums[i] + nums[j] + nums[k] > 0 -> k--
                    else -> {
                        result.add(listOf(nums[i], nums[j], nums[k]))
                        while (j < k && nums[j] == nums[j + 1]) j++
                        while (j < k && nums[k] == nums[k - 1]) k--
                        j++
                        k--
                    }
                }
            }
        }
        return result
    }

    fun minDifference(nums: IntArray): Int {
//        if (nums.size <= 4) return 0
//        nums.sort()
//        val j = nums.lastIndex
//        return minOf(
//            nums[j - 3] - nums[0],
//            nums[j - 2] - nums[1],
//            nums[j - 1] - nums[2],
//            nums[j] - nums[3]
//        )

        if (nums.size <= 4) return 0
        val minNums = IntArray(4) { Int.MAX_VALUE }
        val maxNums = IntArray(4) { Int.MIN_VALUE }
        for (num in nums) {
            for (i in 0..3) {
                if (num < minNums[i]) {
                    for (j in 3 downTo i + 1) {
                        minNums[j] = minNums[j - 1]
                    }
                    minNums[i] = num
                    break
                }
            }
            for (i in 0..3) {
                if (num > maxNums[i]) {
                    for (j in 3 downTo i + 1) {
                        maxNums[j] = maxNums[j - 1]
                    }
                    maxNums[i] = num
                    break
                }
            }
        }
        return minOf(
            maxNums[0] - minNums[3],
            maxNums[1] - minNums[2],
            maxNums[2] - minNums[1],
            maxNums[3] - minNums[0]
        )
    }

    fun averageWaitingTime(customers: Array<IntArray>): Double {
        var currentTime = 0L
        var totalWaitTime = 0L

        for (customer in customers) {
            if (currentTime < customer[0]) {
                currentTime = customer[0].toLong()
            }

            currentTime += customer[1]
            totalWaitTime += currentTime - customer[0]
        }

        return totalWaitTime.toDouble() / customers.size
    }

    fun minOperations(logs: Array<String>): Int {
        var depth = 0
        for (log in logs) {
            when (log) {
                "../" -> depth = maxOf(0, depth - 1)
                "./" -> {}
                else -> depth++
            }
        }
        return depth
    }

    fun hIndex(citations: IntArray): Int {
//        citations.sortDescending()
//        var count = 0
//        for ((index, cited) in citations.withIndex()) {
//            if (cited >= index + 1) {
//                count++
//            } else {
//                break
//            }
//        }
//        return count
        val cited = IntArray(1001) { 0 }
        citations.forEach {
            cited[it]++
        }
        var h = 0
        for (i in 1000 downTo 1) {
            h += cited[i]
            if (h >= i) return i
        }
        return 0
    }

    fun sortPeople(names: Array<String>, heights: IntArray): Array<String> {
        // sort
//        val heightOrders = heights.withIndex().sortedByDescending { it.value }
//        val result = Array(names.size) { "" }
//        for (i in names.indices) {
//            result[i] = names[heightOrders[i].index]
//        }
//        return result

        val n = names.size
        val pq = PriorityQueue<Int> { o1, o2 -> heights[o2] - heights[o1] }
        for (i in 0 until n) {
            pq.offer(i)
        }
        val result = Array(n) { "" }
        for (i in 0 until n) {
            result[i] = names[pq.poll()!!]
        }
        return result
    }

    fun frequencySort(nums: IntArray): IntArray {
//        val freq = mutableMapOf<Int, Int>()
//        nums.forEach { num ->
//            freq[num] = freq.getOrDefault(num, 0) + 1
//        }
//        return nums.sortedWith { o1, o2 ->
//            if (freq[o1] == freq[o2]) {
//                o2 - o1
//            } else {
//                freq[o1]!! - freq[o2]!!
//            }
//        }.toIntArray()
        val freq = IntArray(201)
        for (num in nums) {
            freq[num + 100]++
        }
        val pq = PriorityQueue<Int>(compareBy<Int> { freq[it + 100] }.thenByDescending { it })
        for (num in nums) {
            pq.offer(num)
        }
        val result = IntArray(nums.size)
        for (i in result.indices) {
            result[i] = pq.poll()!!
        }
        return result
    }

//    fun sortJumbled(mapping: IntArray, nums: IntArray): IntArray {
//        fun mappingValue(origin: Int): Int {
//            if (origin == 0) {
//                return mapping[0]
//            }
//            var current = origin
//            var result = 0
//            var multiplier = 1
//            while (current != 0) {
//                result += mapping[current % 10] * multiplier
//                multiplier *= 10
//                current /= 10
//            }
//            return result
//        }
//
//        return nums.sortedWith { o1, o2 ->
//            val p1 = mappingValue(o1)
//            val p2 = mappingValue(o2)
//            p1 - p2
//        }.toIntArray()
//    }

    fun sortJumbled(mapping: IntArray, nums: IntArray): IntArray {
        var maxNumber = nums.maxOrNull() ?: 0
        var digitCount = 0
        while (maxNumber != 0) {
            digitCount++
            maxNumber /= 10
        }

        fun getMappedDigit(value: Int, digitPosition: Int): Int {
            if (value == 0) return mapping[value]
            var scale = 1
            repeat(digitPosition - 1) {
                scale *= 10
            }
            val digit = value / scale
            return if (digit == 0) 0 else mapping[digit % 10]
        }

        fun radixSort(input: IntArray): IntArray {
            var source = input
            var destination = IntArray(source.size)
            val bucketSize = 10
            val bucket = IntArray(bucketSize)
            var count: Int
            var temp: Int

            for (position in 1..digitCount) {
                bucket.fill(0)
                source.forEach { number ->
                    val digit = getMappedDigit(number, position)
                    bucket[digit]++
                }
                count = 0
                for (i in 0 until bucketSize) {
                    temp = bucket[i]
                    bucket[i] = count
                    count += temp
                }
                source.forEach { number ->
                    val digit = getMappedDigit(number, position)
                    destination[bucket[digit]] = number
                    bucket[digit]++
                }
                val tempArray = source
                source = destination
                destination = tempArray
            }
            return source
        }

        return radixSort(nums)
    }

    fun findTheCityWithSmallestReachable(
        n: Int,
        edges: Array<IntArray>,
        distanceThreshold: Int
    ): Int {
        val matrix = Array(n) { IntArray(n) { Int.MAX_VALUE } }
        for (i in 0 until n) {
            matrix[i][i] = 0
        }
        for ((from, to, weight) in edges) {
            matrix[from][to] = weight
            matrix[to][from] = weight
        }
        //  Floyd-Warshall
        for (k in 0 until n) {
            for (i in 0 until n) {
                for (j in 0 until n) {
                    if (matrix[i][k] != Int.MAX_VALUE && matrix[k][j] != Int.MAX_VALUE) {
                        matrix[i][j] = minOf(matrix[i][j], matrix[i][k] + matrix[k][j])
                    }
                }
            }
        }
        var smallest = n
        var result = -1
        for (i in 0 until n) {
            val count = matrix[i].count { it <= distanceThreshold }
            if (count <= smallest) {
                smallest = count
                result = i
            }
        }
        return result
    }

    fun findTheCityWithSmallestReachableDijkstra(
        n: Int,
        edges: Array<IntArray>,
        distanceThreshold: Int
    ): Int {
        val graph = Array(n) { mutableListOf<Pair<Int, Int>>() }
        for ((from, to, weight) in edges) {
            graph[from].add(Pair(to, weight))
            graph[to].add(Pair(from, weight))
        }

        fun dijkstra(n: Int, src: Int, distanceThreshold: Int): IntArray {
            val dist = IntArray(n) { Int.MAX_VALUE }
            dist[src] = 0
            val pq = PriorityQueue<Pair<Int, Int>>(compareBy { it.second })
            pq.add(Pair(src, 0))

            while (pq.isNotEmpty()) {
                val (u, currentDist) = pq.poll()
                if (currentDist > dist[u]) continue

                for ((v, weight) in graph[u]) {
                    val newDist = dist[u] + weight
                    if (newDist < dist[v] && newDist <= distanceThreshold) {
                        dist[v] = newDist
                        pq.add(Pair(v, newDist))
                    }
                }
            }

            return dist
        }

        var smallest = n
        var result = -1

        for (i in 0 until n) {
            val dist = dijkstra(n, i, distanceThreshold)
            val count = dist.count { it <= distanceThreshold }
            if (count <= smallest) {
                smallest = count
                result = i
            }
        }

        return result
    }

    fun minimumCost(
        source: String,
        target: String,
        original: CharArray,
        changed: CharArray,
        cost: IntArray
    ): Long {
        val matrix = Array(26) { LongArray(26) { Long.MAX_VALUE } }
        for (i in 0 until 26) {
            matrix[i][i] = 0
        }
        for (i in original.indices) {
            val u = original[i] - 'a'
            val v = changed[i] - 'a'
            matrix[u][v] = minOf(matrix[u][v], cost[i].toLong())
        }
        //  Floyd-Warshall
        for (k in 0 until 26) {
            for (i in 0 until 26) {
                for (j in 0 until 26) {
                    if (matrix[i][k] != Long.MAX_VALUE && matrix[k][j] != Long.MAX_VALUE) {
                        val potentialSum = try {
                            Math.addExact(matrix[i][k], matrix[k][j])
                        } catch (e: ArithmeticException) {
                            Long.MAX_VALUE
                        }
                        if (potentialSum < Long.MAX_VALUE) {
                            matrix[i][j] = minOf(matrix[i][j], potentialSum)
                        }
                    }
                }
            }
        }
        var result = 0L
        for (i in source.indices) {
            if (matrix[source[i] - 'a'][target[i] - 'a'] == Long.MAX_VALUE) return -1
            result += matrix[source[i] - 'a'][target[i] - 'a']
        }
        return result
    }

    fun numTeams(rating: IntArray): Int {
        val n = rating.size
        if (n < 3) return 0
        var result = 0

//        val dp = Array(n) { Array(2) { IntArray(2) { 0 } } }
//        for (i in 1 until n) {
//            for (j in 0 until i) {
//                if (rating[i] > rating[j]) {
//                    dp[i][0][0] += 1
//                    dp[i][0][1] += dp[j][0][0]
//                } else if (rating[i] < rating[j]) {
//                    dp[i][1][0] += 1
//                    dp[i][1][1] += dp[j][1][0]
//                }
//            }
//            result += dp[i][0][1] + dp[i][1][1]
//        }

        for (j in 1 until n - 1) {
            var lessBefore = 0
            var moreBefore = 0
            var lessAfter = 0
            var moreAfter = 0
            for (i in 0 until j) {
                if (rating[i] < rating[j]) lessBefore++
                if (rating[i] > rating[j]) moreBefore++
            }
            for (k in j + 1 until n) {
                if (rating[k] < rating[j]) lessAfter++
                if (rating[k] > rating[j]) moreAfter++
            }
            result += lessBefore * moreAfter + moreBefore * lessAfter
        }

        return result
    }

    fun minSubArrayLen(target: Int, nums: IntArray): Int {
        // sum equal or greater than target
        var start = 0
        var minLength = Int.MAX_VALUE
        var sum = 0
        for (end in nums.indices) {
            sum += nums[end]
            if (sum >= target) {
                while (sum >= target) {
                    sum -= nums[start]
                    start++
                }
                minLength = minOf(minLength, end - start + 2)
            }
        }
        return if (minLength == Int.MAX_VALUE) 0 else minLength
    }

    fun minHeightShelves(books: Array<IntArray>, shelfWidth: Int): Int {
        // book [width, height]
        val n = books.size
        val dp = IntArray(n + 1) { Int.MAX_VALUE }
        dp[0] = 0  // height before the first book

        for (i in 1..n) {
            var currentWidth = 0
            var currentHeight = 0
            for (j in i downTo 1) {
                currentWidth += books[j - 1][0]
                if (currentWidth > shelfWidth) break
                currentHeight = maxOf(currentHeight, books[j - 1][1])
                dp[i] = minOf(dp[i], dp[j - 1] + currentHeight)
            }
        }
        return dp[n]
    }

    fun minDistanceToConvert(word1: String, word2: String): Int {
        val dp = Array(word1.length + 1) { IntArray(word2.length + 1) }
        for (i in 0..word1.length) {
            for (j in 0..word2.length) {
                when {
                    i == 0 -> {
                        dp[i][j] = j
                    }

                    j == 0 -> {
                        dp[i][j] = i
                    }

                    word1[i - 1] == word2[j - 1] -> {
                        dp[i][j] = dp[i - 1][j - 1]
                    }

                    else -> {
                        dp[i][j] = minOf(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + 1)
                    }
                }
            }
        }
        return dp[word1.length][word2.length]
    }

    fun minSwapsToGroupBinary(nums: IntArray): Int {
        // [0,1,0,1,1,0,0]
        val length = nums.size
        if (length == 1 || length == 2 || length == 3) return 0
        val totalOnes = nums.sum()
        if (totalOnes == 0 || totalOnes == 1 || length - totalOnes == 0 || length - totalOnes == 1) return 0
        val newArray = nums + nums.sliceArray(IntRange(0, totalOnes - 1))
        var start = 0
        var countZeros = 0
        var maxZeros = Int.MAX_VALUE
        for (end in newArray.indices) {
            if (newArray[end] == 0) {
                countZeros++
            }
            if (end - start + 1 == totalOnes) {
                maxZeros = minOf(maxZeros, countZeros)
                if (newArray[start] == 0) {
                    countZeros--
                }
                start++
            }
        }
        return maxZeros
    }

    fun canBeEqual(target: IntArray, arr: IntArray): Boolean {
        val freq = mutableMapOf<Int, Int>()
        target.forEach { to ->
            freq[to] = freq.getOrDefault(to, 0) + 1
        }
        arr.forEach { from ->
            if (!freq.containsKey(from) || freq[from] == 0) return false
            freq[from] = freq[from]!! - 1
        }
        freq.values.forEach { value ->
            if (value != 0) return false
        }
        return true
    }

    fun rangeSum(nums: IntArray, n: Int, left: Int, right: Int): Int {
        val MOD = 1000000007
        val prefixSums = IntArray(n + 1)
        for (i in 1..n) {
            prefixSums[i] = (prefixSums[i - 1] + nums[i - 1]) % MOD
        }
        val newSums = IntArray(n * (n + 1) / 2)
        var index = 0
        for (i in prefixSums.indices) {
            for (j in i + 1 until prefixSums.size) {
                newSums[index++] = prefixSums[j] - prefixSums[i]
            }
        }
        newSums.sort()
        var result = 0
        for (i in left - 1 until right) {
            result = (result + newSums[i]) % MOD
        }
        return result
    }

    fun kthDistinct(arr: Array<String>, k: Int): String {
        val lastPresent = mutableMapOf<String, Int>()
        arr.forEachIndexed { i, str ->
            if (lastPresent.containsKey(str)) {
                arr[lastPresent[str]!!] = "."
                arr[i] = "."
            }
            lastPresent[str] = i
        }
        var count = 0
        for (i in arr.indices) {
            if (arr[i] != ".") {
                count++
            }
            if (count == k) {
                return arr[i]
            }
        }
        return ""
    }

    fun combine(n: Int, k: Int): List<List<Int>> {
        // all possible combinations of k numbers chosen from the range [1, n]
        val result = mutableListOf<List<Int>>()
        val currentList = mutableListOf<Int>()

        fun dfs(num: Int) {
            if (currentList.size == k) {
                result.add(currentList.toList())
                return
            }
            if (num > n) return

            currentList.add(num)
            dfs(num + 1)
            currentList.removeLast()

            dfs(num + 1)
        }
        dfs(1)
        return result
    }

    fun permute(nums: IntArray): List<List<Int>> {
        // all the possible permutations
        val size = nums.size
        val result = mutableListOf<List<Int>>()
        val currentList = mutableListOf<Int>()
        val visited = mutableSetOf<Int>()

        fun dfs() {
            if (currentList.size == size) {
                result.add(currentList.toList())
                return
            }
            for (num in nums) {
                if (num !in visited) {
                    currentList.add(num)
                    visited.add(num)
                    dfs()
                    currentList.removeLast()
                    visited.remove(num)
                }
            }
        }
        dfs()
        return result
    }

    fun minimumPushes(word: String): Int {
        val charFreq = IntArray(26)
        word.forEach { char ->
            charFreq[char - 'a']++
        }
        charFreq.sortDescending()
        var result = 0
        for (i in charFreq.indices) {
            result += charFreq[i] * (i / 8 + 1)
        }
        return result
    }

    fun groupAnagrams(strs: Array<String>): List<List<String>> {

        fun toId(original: String): Int {
            val freq = IntArray(26)
            for (char in original) {
                freq[char - 'a']++
            }
            return freq.contentHashCode()
        }

        return strs.groupBy { toId(it) }.values.toList()
    }

    fun combinationSum(candidates: IntArray, target: Int): List<List<Int>> {
        val n = candidates.size
        val result = mutableListOf<List<Int>>()
        val currentList = mutableListOf<Int>()

        fun dfs(index: Int, currentSum: Int) {
            if (currentSum == target) {
                result.add(currentList.toList())
                return
            }
            if (index >= n || currentSum > target) return

            val newSum = currentSum + candidates[index]
            if (newSum <= target) {
                currentList.add(candidates[index])
                dfs(index, newSum)
                currentList.removeLast()
            }
            dfs(index + 1, currentSum)
//            for (i in index until n) {
//                currentList.add(candidates[i])
//                dfs(i, currentSum + candidates[i])
//                currentList.removeLast()
//            }
        }

        dfs(0, 0)
        return result
    }

    fun combinationSum2(candidates: IntArray, target: Int): List<List<Int>> {
//        val freq = IntArray(51)
//        candidates.forEach {
//            freq[it]++
//        }
//        val result = mutableListOf<List<Int>>()
//        val queue = LinkedList<Triple<Int, Int, MutableList<Int>>>()
//        queue.add(Triple(1, 0, mutableListOf()))
//        while (queue.isNotEmpty()) {
//            var (num, currentSum, currentList) = queue.poll()!!
//            if (currentSum == target) {
//                result.add(currentList)
//                continue
//            } else if (num >= freq.size || currentSum > target) {
//                continue
//            }
//            var next = num + 1
//            while (next < freq.size && freq[next] == 0) {
//                next++
//            }
//            queue.add(Triple(next, currentSum, currentList))
//            for (i in 0 until freq[num]) {
//                currentSum += num
//                val newList = mutableListOf<Int>()
//                newList.addAll(currentList)
//                repeat(i + 1) {
//                    newList.add(num)
//                }
//                queue.add(Triple(next, currentSum, newList))
//            }
//        }
//        return result

        candidates.sort()
        val result = mutableListOf<List<Int>>()
        val currentList = mutableListOf<Int>()

        fun dfs(index: Int, currentSum: Int) {
            if (currentSum == target) {
                result.add(currentList.toList())
                return
            }
            if (index >= candidates.size || currentSum > target) return

            // to avoid duplication: when consider a num, just consider;
            // when not consider a num, we have to skip all duplications too
            val currentNum = candidates[index]
            if (currentSum + currentNum <= target) {
                currentList.add(currentNum)
                dfs(index + 1, currentSum + currentNum)
                currentList.removeLast()
            }
            var next = index + 1
            while (next < candidates.size && candidates[next] == currentNum) {
                next++
            }
            dfs(next, currentSum)
        }
        dfs(0, 0)
        return result
    }

    fun findKthLargest(nums: IntArray, k: Int): Int {
        val pq = PriorityQueue<Int>()
        nums.forEach { num ->
            if (pq.size < k) {
                pq.offer(num)
            } else if (pq.peek() < num) {
                pq.poll()
                pq.offer(num)
            }
        }
//        nums.forEach { num ->
//            pq.offer(num)
//            if (pq.size > k) {
//                pq.poll()
//            }
//        }
        return pq.peek()
    }

    class KthLargest(val k: Int, nums: IntArray) {
        private val pq = PriorityQueue<Int>()

        init {
            nums.forEach { add(it) }
        }

        fun add(`val`: Int): Int {
            if (pq.size < k) {
                pq.offer(`val`)
            } else if (pq.peek()!! < `val`) {
                pq.poll()
                pq.offer(`val`)
            }
            return pq.peek()!!
        }
    }

    fun maxDistance(arrays: List<List<Int>>): Int {
        //        var smallest = arrays[0][0]
        //        var biggest = arrays[0].last()
        //        var maxDistance = 0
        //
        //        for (i in 1 until arrays.size) {
        //            maxDistance = maxOf(maxDistance, abs(arrays[i].last() - smallest), abs(biggest - arrays[i][0]))
        //            smallest = minOf(smallest, arrays[i][0])
        //            biggest = maxOf(biggest, arrays[i].last())
        //        }
        //
        //        return maxDistance
        var firstMax = Pair(Int.MIN_VALUE, -1)
        var secondMax = Pair(Int.MIN_VALUE, -1)
        var firstMin = Pair(Int.MAX_VALUE, -1)
        var secondMin = Pair(Int.MAX_VALUE, -1)

        for (i in arrays.indices) {
            val biggest = arrays[i].last()
            if (biggest > firstMax.first) {
                secondMax = firstMax
                firstMax = Pair(biggest, i)
            } else if (biggest > secondMax.first) {
                secondMax = Pair(biggest, i)
            }
            val smallest = arrays[i][0]
            if (smallest < firstMin.first) {
                secondMin = firstMin
                firstMin = Pair(smallest, i)
            } else if (smallest < secondMin.first) {
                secondMin = Pair(smallest, i)
            }
        }
        if (firstMax.second != firstMin.second) {
            return firstMax.first - firstMin.first
        } else {
            return maxOf(firstMax.first - secondMin.first, secondMax.first - firstMin.first)
        }
    }

    fun maxPointsPruning(points: Array<IntArray>): Long {
        val rows = points.size
        val cols = points[0].size
        var prev = LongArray(cols)
        var curr = LongArray(cols)
        var tmp: LongArray
        for (i in 0 until cols) {
            prev[i] = points[0][i].toLong()
        }
        for (row in 1 until rows) {
            for (i in 0 until cols) {
                for (j in i + 1 until cols) {
                    val delta = prev[i] - j + i
                    if (delta < 0) {
                        break
                    }
                    curr[j] = maxOf(curr[j], delta)
                }
                for (j in i downTo 0) {
                    val delta = prev[i] - i + j
                    if (delta < 0) {
                        break
                    }
                    curr[j] = maxOf(curr[j], delta)
                }
            }
            for (i in 0 until cols) {
                curr[i] = curr[i] + points[row][i]
            }
            tmp = prev
            prev = curr
            curr = tmp
        }
        return prev.maxOf { it }
    }

    fun maxPoints(points: Array<IntArray>): Long {
        val rows = points.size
        val cols = points[0].size
        var prev = LongArray(cols)
        var curr = LongArray(cols)
        var tmp: LongArray
        for (i in 0 until cols) {
            prev[i] = points[0][i].toLong()
        }
        for (row in 1 until rows) {
            val leftMax = LongArray(cols)
            val rightMax = LongArray(cols)
            leftMax[0] = prev[0]
            for (i in 1 until cols) {
                leftMax[i] = maxOf(leftMax[i - 1], prev[i] + i) // 从左边能获得最大贡献
            }
            rightMax[cols - 1] = prev[cols - 1] - (cols - 1)
            for (i in cols - 2 downTo 0) {
                rightMax[i] = maxOf(rightMax[i + 1], prev[i] - i) // 从右边能获得的最大贡献
            }

            for (i in 0 until cols) {
                curr[i] = maxOf(leftMax[i] - i, rightMax[i] + i) + points[row][i]
            }

            tmp = prev
            prev = curr
            curr = tmp
        }
        return prev.maxOf { it }
    }

    fun nthUglyNumber(n: Int): Int {
//        if (n == 1) return 1
//        val factors = intArrayOf(2, 3, 5)
//        val minHeap = PriorityQueue<Int>()
//        val seen = mutableSetOf(1)
//        minHeap.offer(1)
//        var ugly = 1
//        repeat(n) {
//            ugly = minHeap.poll()
//            for (factor in factors) {
//                val newUgly = ugly * factor
//                if (newUgly < 0) break
//                if (newUgly !in seen) {
//                    seen.add(newUgly)
//                    minHeap.offer(newUgly)
//                }
//            }
//        }
//        return ugly

        val arr = IntArray(n)
        arr[0] = 1
        var i = 1
        var i2 = 0 // where needs to * 2
        var i3 = 0 // where needs to * 3
        var i5 = 0 // where needs to * 5
        while (i < n) {
            val a = arr[i2] * 2
            val b = arr[i3] * 3
            val c = arr[i5] * 5
            arr[i] = minOf(a, b, c)

            if (arr[i] == a) i2++
            if (arr[i] == b) i3++
            if (arr[i] == c) i5++
            i++
        }
        return arr.last()
    }

    fun minSteps(n: Int): Int { // 2 keys keyboard
        // 'A' -> n*'A'
//        val dp = IntArray(n + 1)
//        dp[1] = 0
//        for (i in 2..n) {
//            var div = 2
//            while (i % div != 0) {
//                div++
//            }
//            dp[i] = dp[i / div] + div
//        }
//        return dp.last()

        // break down n into its prime factors, add up each factor
        var result = 0
        var num = n
        var factor = 2
        while (num > 1) {
            while (num % factor == 0) {
                result += factor
                num /= factor
            }
            factor++
        }
        return result
    }

    fun findPeakElement(nums: IntArray): Int {
//        if (nums.size == 1) return 0
//        var n = nums.lastIndex
//        if (nums[0] > nums[1]) return 0
//        if (nums[n] > nums[n - 1]) return n
//        var start = 0
//        var end = n
//        var mid: Int
//        while (start < end) {
//            mid = start + (end - start) / 2
//            if (mid == 0 && nums[0] < nums[1]) {
//                return 1
//            } else if (mid == n && nums[n] < nums[n - 1]) {
//                return n - 1
//            } else if (nums[mid] > nums[mid - 1] && nums[mid] > nums[mid + 1]) {
//                return mid
//            } else if (nums[mid] < nums[mid - 1]) {
//                end = mid
//            } else {
//                start = mid
//            }
//        }
//        return start
        var left = 0
        var right = nums.size - 1
        while (left < right) {
            val mid = left + (right - left) / 2
            if (nums[mid] > nums[mid + 1]) { // 此时peak一定是在左边或者就是mid，所以right移到mid
                right = mid
            } else { // 此时peak一定在右边，所以left移动到mid+1
                left = mid + 1
            }
        }
        return left
    }

    fun construct2DArray(original: IntArray, m: Int, n: Int): Array<IntArray> {
        if (original.size != m * n) return arrayOf()
        return Array(m) { row ->
            IntArray(n) { col ->
                original[row * n + col]
            }
        }
    }

    fun matrixReshape(mat: Array<IntArray>, r: Int, c: Int): Array<IntArray> {
        val m = mat.size
        val n = mat[0].size
        if (m * n != r * c) return mat
        return Array(r) { row ->
            IntArray(c) { col ->
                val index = row * c + col
                mat[index / n][index % n]
            }
        }
    }

    fun gameOfLife(board: Array<IntArray>): Unit {
        val neighbourDelta = arrayOf(
            intArrayOf(-1, -1), intArrayOf(-1, 0), intArrayOf(-1, 1),
            intArrayOf(0, -1), intArrayOf(0, 1),
            intArrayOf(1, -1), intArrayOf(1, 0), intArrayOf(1, 1)
        )
        val m = board.size
        val n = board[0].size
        for (i in 0 until m) {
            for (j in 0 until n) {
                var liveCount = 0
                for ((dx, dy) in neighbourDelta) {
                    val nI = i + dx
                    val nJ = j + dy
                    if (nI in 0 until m && nJ in 0 until n) {
                        liveCount += if (board[nI][nJ] == 1 || board[nI][nJ] == 2) 1 else 0
                    }
                    if (liveCount > 3) break
                }
                if (board[i][j] == 1) {
                    if (liveCount < 2 || liveCount > 3) {
                        board[i][j] = 2 // will die
                    }
                } else if (board[i][j] == 0) {
                    if (liveCount == 3) {
                        board[i][j] = -1 // will live
                    }
                }
            }
        }
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (board[i][j] == 2) {
                    board[i][j] = 0
                } else if (board[i][j] == -1) {
                    board[i][j] = 1
                }
            }
        }
    }

    fun triangularSum(nums: IntArray): Int {
        for (i in nums.size - 1 downTo 1) {
            for (j in 0 until i) {
                nums[j] = (nums[j] + nums[j + 1]) % 10
            }
        }
        return nums[0]
    }

    fun robotSim(commands: IntArray, obstacles: Array<IntArray>): Int {
        val DELTA =
            arrayOf(
                intArrayOf(1, 0),
                intArrayOf(0, -1),
                intArrayOf(-1, 0),
                intArrayOf(0, 1)
            ) // East, South, West, North
        var direction = 3
        fun updateDirection(command: Int) {
            if (command == -2) {
                direction--
                if (direction == -1) direction = 3
            } else if (command == -1) {
                direction++
                if (direction == 4) direction = 0
            }
        }

        var x = 0
        var y = 0
        var maxDistance = 0
        val obstaclesSet = obstacles.map { it[0] to it[1] }.toHashSet()
        for (cmd in commands) {
            if (cmd == -1 || cmd == -2) {
                updateDirection(cmd)
            } else {
                for (i in 0 until cmd) {
                    val nextX = x + DELTA[direction][0]
                    val nextY = y + DELTA[direction][1]
                    if (nextX to nextY in obstaclesSet) {
                        break
                    }
                    x = nextX
                    y = nextY
                }
                maxDistance = maxOf(maxDistance, x * x + y * y)
            }
        }
        return maxDistance
    }

    class Robot(val width: Int, val height: Int) {
        val DIRECTIONS = arrayOf("North", "East", "South", "West")
        val DELTA =
            arrayOf(intArrayOf(0, 1), intArrayOf(1, 0), intArrayOf(0, -1), intArrayOf(-1, 0))
        var x = 0
        var y = 0
        var direction = 1
        val modulo = 2 * (width - 1) + 2 * (height - 1)

        fun step(num: Int) {
            val curr = if (num > modulo) num % modulo else num
            if (num != 0 && curr == 0) {
                if (x == 0 && y == 0 && direction == 1) direction = 2
                else if (x == width - 1 && y == 0 && direction == 0) direction = 1
                else if (x == width - 1 && y == height - 1 && direction == 3) direction = 0
                else if (x == 0 && y == height - 1 && direction == 2) direction = 3
                return
            }
            val nextX = x + DELTA[direction][0] * curr
            val nextY = y + DELTA[direction][1] * curr
            val moved: Int
            if (nextX >= width) {
                direction = 0
                moved = width - 1 - x
                x = width - 1
            } else if (nextX < 0) {
                direction = 2
                moved = x
                x = 0
            } else if (nextY >= height) {
                direction = 3
                moved = height - 1 - y
                y = height - 1
            } else if (nextY < 0) {
                direction = 1
                moved = y
                y = 0
            } else {
                moved = curr
                x = nextX
                y = nextY
            }
            if (curr != moved) {
                step(curr - moved)
            }
        }

        fun getPos(): IntArray {
            return intArrayOf(x, y)
        }

        fun getDir(): String {
            return DIRECTIONS[direction]
        }

    }

    fun missingRolls(rolls: IntArray, mean: Int, n: Int): IntArray {
        val m = rolls.size
        val total = mean * (m + n)
        val target = total - rolls.sumOf { it }
        val averages = IntArray(6) { (it + 1) * n }
        if (target < averages[0] || target > averages[5]) return intArrayOf()
        var index = averages.binarySearch(target)
        if (index >= 0) {
            return IntArray(n) { index + 1 }
        } else {
            index = -(index + 1)
            val result = IntArray(n) { index }
            for (i in 0 until target - averages[index - 1]) {
                result[i]++
            }
            return result
        }
    }

    fun numRollsToTarget(n: Int, k: Int, target: Int): Int {
        if (target < n || target > n * k) return 0
        val MODULO = 1000000007
        var curr = IntArray(target + 1)
        var prev = IntArray(target + 1)
        prev[0] = 1
        for (i in 1..n) {
            for (j in 0..target) {
                for (x in 1..k) {
                    if (j - x >= 0) {
                        // 新一颗骰子骰出x，次数就加上少一颗的时候能投出j-x的组合数
                        curr[j] = (curr[j] + prev[j - x]) % MODULO
                    }
                }
            }
            val tmp = prev
            prev = curr
            curr = tmp.apply { fill(0) }
        }
        return prev[target]
    }

    class MinStack() {
        private val stack = LinkedList<Pair<Int, Int>>()
        fun push(`val`: Int) {
            val top = stack.peek()
            if (top == null) {
                stack.push(`val` to `val`)
            } else {
                stack.push(`val` to minOf(top.second, `val`))
            }
        }

        fun pop() {
            stack.pop()
        }

        fun top(): Int {
            return stack.peek()!!.first
        }

        fun getMin(): Int {
            return stack.peek()!!.second
        }
    }

    fun searchRotated(nums: IntArray, target: Int): Int {
        //        val n = nums.size
        //        var left = 0
        //        var right = n - 1
        //        while (left <= right) {
        //            val mid = left + ((right - left) shr 1)
        //            if (nums[mid] < nums[0]) {
        //                right = mid - 1
        //            } else {
        //                left = mid + 1
        //            }
        //        }
        //        val pivot = left
        //        left = 0
        //        right = n - 1
        //        while (left <= right) {
        //            val mid = left + ((right - left) shr 1)
        //            val realMid = (mid + pivot) % n
        //            when {
        //                nums[realMid] == target -> return realMid
        //                nums[realMid] < target -> left = mid + 1
        //                else -> right = mid - 1
        //            }
        //        }
        //        return -1
        val size = nums.size
        var start = 0
        var end = size - 1
        while (start <= end) {
            if (nums[start] == target) return start
            if (nums[end] == target) return end
            val mid = start + (end - start) / 2
            if (nums[mid] == target) return mid
            if (nums[start] <= nums[mid]) { // first half is sorted. mid can be start, use "<="
                if (target >= nums[start] && target < nums[mid]) {
                    end = mid - 1
                } else {
                    start = mid + 1
                }
            } else { // last half is sorted
                if (target > nums[mid] && target <= nums[end]) {
                    start = mid + 1
                } else {
                    end = mid - 1
                }
            }
        }
        return -1
    }

    fun peopleIndexes(favoriteCompanies: List<List<String>>): List<Int> {
        val companies = Array(favoriteCompanies.size) {
            favoriteCompanies[it].toHashSet()
        }
        val result = mutableListOf<Int>()
//        for (i in companies.indices) {
//            val current = companies[i]
//            var found = false
//            for (another in companies) {
//                if (current != another && another.containsAll(current)) {
//                    found = true
//                    break
//                }
//            }
//            if (!found) {
//                result.add(i)
//            }
//        }
//        return result

        // union-find
        val parent = IntArray(favoriteCompanies.size) { it }
        fun find(x: Int): Int {
            if (parent[x] != x) {
                parent[x] = find(parent[x])
            }
            return parent[x]
        }

        fun union(x: Int, y: Int) {
            val rootX = find(x)
            val rootY = find(y)
            if (rootX != rootY) {
                if (companies[rootX].containsAll(companies[rootY])) parent[rootY] = rootX
                else if (companies[rootY].containsAll(companies[rootX])) parent[rootX] = rootY
            }
        }

        for (i in companies.indices) {
            for (j in i + 1 until companies.size) {
                union(i, j)
            }
        }
        for (i in companies.indices) {
            if (find(i) == i) {
                result.add(i)
            }
        }
        return result

    }

    fun countMaxOrSubsets(nums: IntArray): Int {
        val size = nums.size
        var finalSum = 0
        for (num in nums) {
            finalSum = finalSum or num
        }
        var result = 0

        fun dfs(index: Int, currentSum: Int) {
            if (index >= size) return
            val newSum = currentSum or nums[index]
            if (newSum == finalSum) result++
            dfs(index + 1, newSum)
            dfs(index + 1, currentSum)
        }

        dfs(0, 0)
        return result
    }

    fun smallestCountSubarraysWithMaximumOr(nums: IntArray): IntArray {
//        val total = IntArray(32)
//        var target = 0
//        for (num in nums) {
//            var current = num
//            var index = 0
//            while (current != 0) {
//                if (current and 1 == 1) {
//                    if (total[index]++ == 0) {
//                        target++
//                    }
//                }
//                current = current shr 1
//                index++
//            }
//        }
//
//        val bitCount = IntArray(32)
//        var distinct = 0
//
//        fun addNum(num: Int) {
//            var current = num
//            var index = 0
//            while (current != 0) {
//                if (current and 1 == 1) {
//                    if (bitCount[index]++ == 0) {
//                        distinct++
//                    }
//                }
//                index++
//                current = current shr 1
//            }
//        }
//
//        fun removeNum(num: Int) {
//            var current = num
//            var index = 0
//            while (current != 0) {
//                if (current and 1 == 1) {
//                    if (--bitCount[index] == 0) {
//                        distinct--
//                    }
//                    if (--total[index] == 0) {
//                        target--
//                    }
//                }
//                index++
//                current = current shr 1
//            }
//        }
//
//        val result = IntArray(nums.size)
//        var left = 0
//        for (right in nums.indices) {
//            addNum(nums[right])
//            while (left <= right && distinct == target) {
//                result[left] = right - left + 1
//                removeNum(nums[left])
//                left++
//            }
//        }
//        return result
        val size = nums.size
        val result = IntArray(size)
        val lastSeen = IntArray(30) { -1 } // 记录每个位最后出现的位置
        for (i in size - 1 downTo 0) {
            for (bit in 0 until 30) {
                if ((nums[i] and (1 shl bit)) != 0) {
                    lastSeen[bit] = i
                }
            }
            var peakPoint = i
            for (bit in 0 until 30) {
                peakPoint = maxOf(peakPoint, lastSeen[bit])
            }
            result[i] = peakPoint - i + 1
        }
        return result
    }

    fun findMinDifference(timePoints: List<String>): Int {

        fun parse(t: String): Int {
            val split = t.split(":")
            return split[0].toInt() * 60 + split[1].toInt()
        }

        val size = timePoints.size
        val sorted = timePoints.map { parse(it) }.sorted()
        var minDiff = sorted[0] + 1440 - sorted[size - 1]
        for (i in 0 until size - 1) {
            minDiff = minOf(minDiff, sorted[i + 1] - sorted[i])
        }
        return minDiff
    }

    fun nextPermutation(nums: IntArray): Unit {
        val n = nums.size
        var target = -1
        for (i in n - 2 downTo 0) {
            if (nums[i] < nums[i + 1]) {
                target = i
                break
            }
        }

        fun swap(i: Int, j: Int) {
            val tmp = nums[i]
            nums[i] = nums[j]
            nums[j] = tmp
        }

        if (target >= 0) {
            // binary search for the right-most number bigger than target
            var left = target + 1
            var right = n - 1
            while (left <= right) {
                val mid = left + ((right - left) shr 1)
                if (nums[mid] > nums[target]) {
                    left = mid + 1
                } else {
                    right = mid - 1
                }
            }
            swap(target, right)
        }

        // reverse all starting from start + 1
        var left = target + 1
        var right = n - 1
        while (left <= right) {
            swap(left, right)
            left++
            right--
        }
    }

    fun orderOfLargestPlusSign(n: Int, mines: Array<IntArray>): Int {
        val graph = Array(n) { BooleanArray(n) { true } }
        mines.forEach { mine ->
            graph[mine[0]][mine[1]] = false
        }
        val maxOrder = (n + 1) / 2
        var previous = Array(n) { BooleanArray(n) }
        var current = Array(n) { BooleanArray(n) }
        var found = false
        for (i in 0 until n) {
            for (j in 0 until n) {
                previous[i][j] = graph[i][j]
                found = found || previous[i][j]
            }
        }
        if (!found) {
            return 0
        }
        for (arm in 1 until maxOrder) {
            found = false
            for (i in 0 until n) {
                for (j in 0 until n) {
                    current[i][j] = previous[i][j]
                            && (if (i + arm < n) graph[i + arm][j] else false)
                            && (if (i - arm >= 0) graph[i - arm][j] else false)
                            && (if (j + arm < n) graph[i][j + arm] else false)
                            && (if (j - arm >= 0) graph[i][j - arm] else false)
                    found = found || current[i][j]
                }
            }
            if (!found) {
                return arm
            }
            val tmp = previous
            previous = current
            current = tmp
        }
        return maxOrder
    }

    fun minExtraChar(s: String, dictionary: Array<String>): Int {
        val n = s.length
        val dp = IntArray(n + 1) { it }

        //        val wordSet = dictionary.toSet()
//        for (i in 1..n) {
//            for (j in 0 until i) {
//                if (s.substring(j, i) in wordSet) {
//                    dp[i] = minOf(dp[i], dp[j])
//                } else {
//                    dp[i] = minOf(dp[i], dp[j] + i - j)
//                }
//            }
//        }
//        return dp[n]
        class TrieNode {
            val children = mutableMapOf<Char, TrieNode>()
            var isWord = false
        }

        fun buildTrie(dictionary: Array<String>): TrieNode {
            val root = TrieNode()
            for (word in dictionary) {
                var node = root
                for (char in word) {
                    node = node.children.getOrPut(char) { TrieNode() }
                }
                node.isWord = true
            }
            return root
        }

        val root = buildTrie(dictionary)
        for (i in 0 until n) {
            var node = root
            for (j in i until n) {
                node = node.children[s[j]] ?: break
                if (node.isWord) {
                    dp[j + 1] = minOf(dp[j + 1], dp[i])
                }
            }
            dp[i + 1] = minOf(dp[i + 1], dp[i] + 1)
        }
        return dp[n]
    }

    fun longestCommonPrefix(arr1: IntArray, arr2: IntArray): Int {
        class TrieNode {
            val children = mutableMapOf<Char, TrieNode>()
        }

        fun buildTrie(arr: IntArray): TrieNode {
            val root = TrieNode()
            for (num in arr) {
                var node = root
                val text = num.toString()
                for (char in text) {
                    node = node.children.getOrPut(char) { TrieNode() }
                }
            }
            return root
        }

        fun findPrefixLength(num: Int, root: TrieNode): Int {
            var result = 0
            val text = num.toString()
            var node = root
            for (char in text) {
                node = node.children[char] ?: break
                result++
            }
            return result
        }

        val root = buildTrie(arr1)
        var longest = 0
        for (num in arr2) {
            longest = maxOf(longest, findPrefixLength(num, root))
        }
        return longest
    }

    fun longestCommonPrefixTrie(strs: Array<String>): String {
        class TrieNode {
            val children = mutableMapOf<Char, TrieNode>()
            var isEnd = false
        }

        fun buildTrie(): TrieNode {
            val root = TrieNode()
            for (str in strs) {
                var node = root
                if (str.isEmpty()) {
                    node.children.clear()
                    break
                }
                for (char in str) {
                    node = node.children.getOrPut(char) { TrieNode() }
                }
                node.isEnd = true
            }
            return root
        }

        var node = buildTrie()
        val sb = StringBuilder()
        while (!node.isEnd && node.children.keys.size == 1) {
            for (key in node.children.keys) {
                sb.append(key)
                node = node.children[key]!!
            }
        }
        return sb.toString()
    }

    fun kSmallestPairs(nums1: IntArray, nums2: IntArray, k: Int): List<List<Int>> {
        val result = mutableListOf<List<Int>>()
        if (nums1.isEmpty() || nums2.isEmpty() || k == 0) return result

        val pq = PriorityQueue(compareBy<Pair<Int, Int>> { nums1[it.first] + nums2[it.second] })
        for (i in 0 until minOf(k, nums1.size)) {
            pq.offer(Pair(i, 0))
        }

        while (result.size < k && pq.isNotEmpty()) {
            val (i, j) = pq.poll()!!
            result.add(listOf(nums1[i], nums2[j]))
            if (j + 1 < nums2.size) {
                pq.offer(Pair(i, j + 1))
            }
        }

        return result
    }

    class MyCalendar() {
//        val starts = mutableListOf<Int>()
//        val ends = mutableListOf<Int>()
//
//        fun book(start: Int, end: Int): Boolean {
//            var i = starts.binarySearch(start)
//            if (i >= 0) return false
//            i = - (i + 1)
//            if (i - 1 >= 0 && ends[i - 1] > start) return false
//            if (i < starts.size && starts[i] < end) return false
//            starts.add(i, start)
//            ends.add(i, end)
//            return true
//        }

        data class Event(val time: Int, val type: Int)

        val events = mutableListOf<Event>()

        fun book(start: Int, end: Int): Boolean {
            var i = events.binarySearch(Event(start, 1), compareBy { it.time })
            if (i >= 0) {
                if (events[i].type == 1) return false
                if (i + 1 < events.size && events[i + 1].time < end) return false
                i++
            } else {
                i = -(i + 1)
                if (i < events.size) {
                    if (events[i].type == -1 || events[i].time < end) return false
                }
            }
            events.add(i, Event(end, -1))
            events.add(i, Event(start, 1))
            return true
        }
    }

    class MyCalendarTwo() {
        data class Event(val time: Int, val type: Int)

        val events = mutableListOf<Event>()

        fun book(start: Int, end: Int): Boolean {
            val e1 = Event(start, 1)
            val e2 = Event(end, -1)

            // 添加、排序，不符合再删除
//            events.add(e1)
//            events.add(e2)
//            events.sortWith(compareBy<Event> { it.time }.thenBy { it.type })
//            var overlap = 0
//            for (event in events) {
//                overlap += event.type
//                if (overlap > 2) {
//                    events.remove(e1)
//                    events.remove(e2)
//                    return false
//                }
//            }
//            return true

            // 查找插入点，直接计算范围内重叠值
            val startPos = events.binarySearch(e1, compareBy<Event> { it.time }.thenBy { it.type })
            val endPos = events.binarySearch(e2, compareBy<Event> { it.time }.thenBy { it.type })
            val startInsertPos = if (startPos < 0) -(startPos + 1) else startPos
            val endInsertPos = if (endPos < 0) -(endPos + 1) else endPos

            var overlap = 0
            for (i in 0 until events.size) {
                if (i in startInsertPos..endInsertPos && overlap >= 2) {
                    return false // 如果在插入范围内overlap超过2，无法插入
                }
                overlap += events[i].type
            }

            events.add(startInsertPos, e1)
            events.add(endInsertPos + 1, e2)
            return true
        }
    }

    class MyCalendarTwoNoLineSweep() {
        private val doubleOverlapped = mutableListOf<Pair<Int, Int>>()
        private val allBookings = mutableListOf<Pair<Int, Int>>()

        fun doesOverlap(s1: Int, e1: Int, s2: Int, e2: Int): Boolean {
            return maxOf(s1, s2) < minOf(e1, e2)
        }

        fun getOverlappedInterval(s1: Int, e1: Int, s2: Int, e2: Int): Pair<Int, Int> {
            return maxOf(s1, s2) to minOf(e1, e2)
        }

        fun book(start: Int, end: Int): Boolean {
            // if current interval overlaps with already double booked interval
            doubleOverlapped.forEach { (s1, e1) ->
                if (doesOverlap(s1, e1, start, end)) {
                    return false
                }
            }
            for ((s1, e1) in allBookings) {
                if (doesOverlap(s1, e1, start, end)) {
                    val res = getOverlappedInterval(s1, e1, start, end)
                    doubleOverlapped.add(res)
                }
            }
            allBookings.add(start to end)
            return true
        }

    }

    class MyCircularDeque(val k: Int) {
        val list = IntArray(k)
        var frontIndex = 1
        var rearIndex = 0
        var count = 0

        fun insertFront(value: Int): Boolean {
            if (isFull()) return false
            if (frontIndex == 0) {
                frontIndex = k - 1
            } else {
                frontIndex--
            }
            list[frontIndex] = value
            count++
            return true
        }

        fun insertLast(value: Int): Boolean {
            if (isFull()) return false
            if (rearIndex == k - 1) {
                rearIndex = 0
            } else {
                rearIndex++
            }
            list[rearIndex] = value
            count++
            return true
        }

        fun deleteFront(): Boolean {
            if (isEmpty()) return false
            if (frontIndex == k - 1) {
                frontIndex = 0
            } else {
                frontIndex++
            }
            count--
            return true
        }

        fun deleteLast(): Boolean {
            if (isEmpty()) return false
            if (rearIndex == 0) {
                rearIndex = k - 1
            } else {
                rearIndex--
            }
            count--
            return true
        }

        fun getFront(): Int {
            if (count == 0) return -1
            return list[frontIndex]
        }

        fun getRear(): Int {
            if (count == 0) return -1
            return list[rearIndex]
        }

        fun isEmpty(): Boolean {
            return count == 0
        }

        fun isFull(): Boolean {
            return count == k
        }
    }

    fun canArrange(arr: IntArray, k: Int): Boolean {
        val numFreq = IntArray(k)
        for (num in arr) {
            var mod = num % k
            if (mod < 0) {
                mod += k
            }
            numFreq[mod]++
        }
        if (numFreq[0] % 2 != 0) return false
        for (i in 1..k / 2) {
            if (numFreq[i] != numFreq[k - i]) return false
        }
        return true
    }

    fun arrayRankTransform(arr: IntArray): IntArray {
        val pq = PriorityQueue<Int>() { o1, o2 ->
            arr[o1] - arr[o2]
        }
        for (i in arr.indices) {
            pq.offer(i)
        }
        val result = IntArray(arr.size)
        var rank = 1
        while (pq.isNotEmpty()) {
            val index = pq.poll()!!
            result[index] = rank
            if (pq.peek() != null && arr[pq.peek()] != arr[index]) {
                rank++
            }
        }
        return result
    }

    fun searchRange(nums: IntArray, target: Int): IntArray {
        val n = nums.size
        var left = 0
        var right = n - 1
        val result = intArrayOf(-1, -1)
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (nums[mid] >= target) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        if (left < 0 || left >= n || nums[left] != target) return result
        result[0] = left
        left = 0
        right = n - 1
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (nums[mid] <= target) {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        result[1] = right
        return result
    }

    fun minSubarray(nums: IntArray, p: Int): Int {
        var rest = 0
        for (num in nums) {
            rest = (rest + num % p) % p
        }
        if (rest == 0) return 0

        val prefixSumMap = mutableMapOf(0 to -1)
        var sum = 0
        var result = nums.size
        for (i in nums.indices) {
            sum = (sum + nums[i] % p) % p
            val target = (sum - rest + p) % p
            if (prefixSumMap.containsKey(target)) {
                val index = prefixSumMap[target]!!
                result = minOf(result, i - index)
            }
            prefixSumMap[sum] = i
        }
        return if (result == nums.size) -1 else result
    }

    fun dividePlayers(skill: IntArray): Long {
        skill.sort()
        var i = 0
        var j = skill.lastIndex
        val target = skill[i] + skill[j]
        var product = 1L * skill[i++] * skill[j--]
        while (i < j) {
            if (skill[i] + skill[j] != target) return -1L
            product += skill[i++] * skill[j--]
        }
        return product
    }

    fun maxOperationsToPairTarget(nums: IntArray, k: Int): Int {
        nums.sort()
        var i = 0
        var j = nums.lastIndex
        var count = 0
        while (i < j) {
            val sum = nums[i] + nums[j]
            if (sum == k) {
                count++
                i++
                j--
            } else if (sum < k) {
                i++
            } else {
                j--
            }
        }
        return count
    }

    fun maxWidthRamp(nums: IntArray): Int {
//        val pq = PriorityQueue<Int>(compareBy<Int> { nums[it] }.thenBy { it })
//        for (i in nums.indices) {
//            pq.offer(i)
//        }
//        var minIndex = Int.MAX_VALUE
//        var maxWidth = 0
//        while (pq.isNotEmpty()) {
//            val index = pq.poll()!!
//            maxWidth = maxOf(maxWidth, index - minIndex)
//            minIndex = minOf(minIndex, index)
//        }
//        return maxWidth

        // monotonic stack
        val stack = LinkedList<Int>()
        // Step 1: Build a decreasing stack of indices
        for (i in nums.indices) {
            if (stack.isEmpty() || nums[i] < nums[stack.peek()!!]) {
                stack.push(i)
            }
        }
        var maxWidth = 0
        // Step 2: Traverse from the end to the beginning
        var i = nums.size - 1
        while (i >= 0 && stack.isNotEmpty()) {
            if (nums[i] >= nums[stack.peek()!!]) {
                maxWidth = maxOf(maxWidth, i - stack.pop())
            } else {
                i--
            }
        }
        return maxWidth
    }

    fun smallestChair(times: Array<IntArray>, targetFriend: Int): Int {

        data class Chair(val index: Int, var emptyAt: Int)

        val occupied = PriorityQueue<Chair>(compareBy { it.emptyAt })
        val free = PriorityQueue<Chair>(compareBy { it.index })
        val sorted = times.withIndex().sortedBy { it.value[0] }
        for ((originalIndex, time) in sorted) {
            // Free up chairs for friends that have already left
            while (occupied.peek() != null && occupied.peek()!!.emptyAt <= time[0]) {
                free.offer(occupied.poll()!!)
            }
            // Assign the next available chair
            val chair = if (free.isNotEmpty()) {
                free.poll()!!
            } else {
                Chair(occupied.size, 0)
            }
            if (originalIndex == targetFriend) {
                return chair.index
            } else {
                chair.emptyAt = time[1]
                occupied.offer(chair)
            }
        }
        return 0
    }

    fun minGroups(intervals: Array<IntArray>): Int {
//        intervals.sortWith (compareBy<IntArray> { it[0] }.thenBy { it[1] })
//        val pq = PriorityQueue<Int>() // end
//        for ((left, right) in intervals) {
//            if (pq.isNotEmpty() && pq.peek()!! < left) {
//                pq.poll()
//            }
//            pq.offer(right)
//        }
//        return pq.size

        val prefixSum = IntArray(1000002) { 0 }
        var count = 0
        for ((start, end) in intervals) {
            prefixSum[start]++
            prefixSum[end + 1]--
        }

        for (i in 1 until prefixSum.size) {
            prefixSum[i] += prefixSum[i - 1]
            count = maxOf(count, prefixSum[i])
        }
        return count
    }

    fun maxArea(height: IntArray): Int {
        var left = 0
        var right = height.lastIndex
        var maxArea = 0
        while (left < right) {
            if (height[left] <= height[right]) {
                maxArea = maxOf(maxArea, (right - left) * height[left])
                left++
            } else {
                maxArea = maxOf(maxArea, (right - left) * height[right])
                right--
            }
        }
        return maxArea
    }

    fun snakesAndLadders(board: Array<IntArray>): Int {
        val n = board.size
        val maxIndex = n * n
        val mustJump = IntArray(maxIndex + 1)
        var index = 1
        var backFlag = false
        for (i in n - 1 downTo 0) {
            if (backFlag) {
                for (j in n - 1 downTo 0) {
                    mustJump[index++] = board[i][j]
                }
            } else {
                for (j in 0 until n) {
                    mustJump[index++] = board[i][j]
                }
            }
            backFlag = !backFlag
        }
        // bfs
//        val visited = mutableSetOf<Int>()
//        val queue = LinkedList<Pair<Int, Int>>()
//        queue.offer(1 to 0)
//        while (queue.isNotEmpty()) {
//            val (curr, count) = queue.poll()!!
//            if (curr in visited) {
//                continue
//            } else {
//                visited.add(curr)
//            }
//            for (step in 1..6) {
//                val next = minOf(curr + step, maxIndex)
//                if (next == maxIndex) return count + 1
//                if (mustJump[next] == -1) {
//                    queue.offer(next to count + 1)
//                } else {
//                    queue.offer(mustJump[next] to count + 1)
//                }
//            }
//        }
//        return -1

        // dp with relaxation
        val dp = IntArray(maxIndex + 1) { Int.MAX_VALUE }
        dp[1] = 0
        var updated = true
        while (updated) {
            updated = false
            for (i in 1 until maxIndex) {
                if (dp[i] == Int.MAX_VALUE) continue
                for (step in 1..6) {
                    val next = i + step
                    if (next > maxIndex) break
                    val destination = if (mustJump[next] == -1) next else mustJump[next]
                    if (dp[destination] > dp[i] + 1) {
                        dp[destination] = dp[i] + 1
                        updated = true // flag that we've made an update
                    }
                }
            }
        }
        return if (dp[maxIndex] == Int.MAX_VALUE) -1 else dp[maxIndex]
    }

    fun maxKElements(nums: IntArray, k: Int): Long {
        // pick biggest nums[i], then place with ceil(nums[i] / 3)
        val pq = PriorityQueue<Int>() { o1, o2 -> o2 - o1 }
        for (num in nums) {
            pq.offer(num)
        }
        var times = 0
        var sum = 0L
        while (times != k) {
            val current = pq.poll()!!
            sum += current
            pq.offer(if (current % 3 == 0) current / 3 else current / 3 + 1)
            times++
        }
        return sum
    }

    // Kadane's Algorithm
    fun maxSumSubArray(nums: IntArray): Int {
        var maxSum = nums[0]
        var currentSum = nums[0]

        for (i in 1 until nums.size) {
//            if (currentSum > 0) {
//                currentSum += nums[i]
//            } else {
//                currentSum = nums[i]
//            }
            currentSum = maxOf(nums[i], currentSum + nums[i])
            maxSum = maxOf(maxSum, currentSum)
        }

        return maxSum
    }

    // Kadane's Algorithm check max and min
    fun maxSubarraySumCircular(nums: IntArray): Int {
        var maxSum = Int.MIN_VALUE
        var minSum = Int.MAX_VALUE
        var currentSum = 0
        var currentMinSum = 0
        var totalSum = 0
        for (num in nums) {
            currentSum = maxOf(currentSum + num, num)
            maxSum = maxOf(maxSum, currentSum)

            currentMinSum = minOf(currentMinSum + num, num)
            minSum = minOf(minSum, currentMinSum)

            totalSum += num
        }
        if (minSum == totalSum) {
            return maxSum
        }
        return maxOf(maxSum, totalSum - minSum)
    }

    fun maxAbsoluteSum(nums: IntArray): Int {
        //        var max = abs(nums[0])
        //        var currMin = nums[0]
        //        var currMax = nums[0]
        //        for(i in 1 until nums.size){
        //            if(currMin > 0){
        //                currMin = 0
        //            }
        //            if(currMax < 0){
        //                currMax = 0
        //            }
        //            currMin = currMin + nums[i]
        //            currMax = currMax + nums[i]
        //            max = maxOf(max, -currMin, currMax)
        //        }
        //        return max
        var currentSum = 0
        var negativeSum = 0
        var maxSum = 0
        for (num in nums) {
            currentSum = maxOf(currentSum + num, num)
            negativeSum = minOf(negativeSum + num, 0)
            maxSum = maxOf(maxSum, currentSum, -negativeSum)
        }
        return maxSum
    }

    fun removeSubfolders(folder: Array<String>): List<String> {
        //    folder.sort()
        //    val result = mutableListOf<String>()
        //    for (f in folder) {
        //        if (result.isEmpty() || !f.startsWith(result.last() + "/")) {
        //            result.add(f)
        //        }
        //    }
        //    return result

        class TrieNode {
            val children = hashMapOf<String, TrieNode>()
            var isEnd = false
        }

        fun buildTrie(): TrieNode {
            val root = TrieNode()
            for (str in folder) {
                var node = root
                var start = 1
                for (end in 1..str.length) {
                    if (end == str.length || str[end] == '/') {
                        val dir = str.substring(start, end)
                        node = node.children.getOrPut(dir) { TrieNode() }
                        if (node.isEnd) break
                        start = end + 1
                    }
                }
                node.isEnd = true
                if (node.children.keys.isNotEmpty()) {
                    node.children.clear()
                }
            }
            return root
        }

        val root = buildTrie()
        val result = mutableListOf<String>()
        val sb = StringBuilder()

        fun dfs(node: TrieNode) {
            if (node.isEnd) {
                result.add(sb.toString())
                return
            }
            for ((key, next) in node.children) {
                val length = sb.length
                sb.append("/$key")
                dfs(next)
                sb.setLength(length)
            }
        }

        dfs(root)
        return result
    }

    fun maxMoves(grid: Array<IntArray>): Int {
        val m = grid.size
        val n = grid[0].size
        val dp = IntArray(m)
        var above = 0 // for rolling storage
        var max = 0
        var allNegative = true // Flag to check if all dp values are -1
        for (col in 1 until n) {
            allNegative = true
            for (row in grid.indices) {
                val tmp = dp[row] // as "above" for next col
                var current = -1
                if (row - 1 >= 0 && above != -1 && grid[row][col] > grid[row - 1][col - 1]) {
                    current = maxOf(current, above + 1)
                }
                if (dp[row] != -1 && grid[row][col] > grid[row][col - 1]) {
                    current = maxOf(current, dp[row] + 1)
                }
                if (row + 1 < m && dp[row + 1] != -1 && grid[row][col] > grid[row + 1][col - 1]) {
                    current = maxOf(current, dp[row + 1] + 1)
                }
                // update current dp
                dp[row] = current
                if (dp[row] != -1) {
                    allNegative = false // valid path exists
                }
                // prepare next above
                above = tmp
                // update max moves
                max = maxOf(max, dp[row])
            }
            if (allNegative) return max // if no valid path, we can return
        }

        return max
    }

    fun canSortArray(nums: IntArray): Boolean {
        val n = nums.size
        val bitCount = IntArray(257) { -1 }
        for (i in 0 until n) {
            if (bitCount[nums[i]] != -1) continue
            var curr = nums[i]
            while (curr != 0) {
                bitCount[nums[i]] += curr % 2
                curr /= 2
            }
        }
        for (i in 0 until n - 1) {
            for (j in 0 until n - 1 - i) {
                if (nums[j] > nums[j + 1]) {
                    if (bitCount[nums[j]] != bitCount[nums[j + 1]]) return false
                    val temp = nums[j + 1]
                    nums[j + 1] = nums[j]
                    nums[j] = temp
                }
            }
        }
        return true
    }

    fun largestCombination(candidates: IntArray): Int {
        val bitCount = IntArray(24)
        for (num in candidates) {
            var current = num
            var bitIndex = 0
            while (current != 0) {
                bitCount[bitIndex++] += current % 2
                current /= 2
            }
        }
        return bitCount.maxOf { it }
    }

    fun primeSubOperation(nums: IntArray): Boolean {
        val primes = MathCode.generatePrimes(1000)

        // binary search smallest between [atLeast, atMost]
        fun findSmallestPrime(atLeast: Int, atMost: Int): Int {
            var left = 0
            var right = primes.lastIndex
            while (left <= right) {
                val mid = left + (right - left) / 2
                when {
                    primes[mid] == atLeast -> return if (primes[mid] <= atMost) primes[mid] else -1
                    primes[mid] < atLeast -> left = mid + 1
                    else -> right = mid - 1
                }
            }
            if (left < primes.size && primes[left] <= atMost) {
                return primes[left]
            }
            return -1
        }

        val n = nums.size
        for (i in n - 2 downTo 0) {
            if (nums[i] >= nums[i + 1]) {
                // find smallest prime to subtract
                val prime = findSmallestPrime(nums[i] - nums[i + 1] + 1, nums[i] - 1)
                if (prime == -1) return false
                nums[i] -= prime
            }
        }
        return true
    }

    fun maximumBeauty(items: Array<IntArray>, queries: IntArray): IntArray {
        items.sortWith(compareBy<IntArray> { it[0] }.thenBy { it[1] })
        val sortedQueries = queries.withIndex().sortedByDescending { it.value }
        val n = items.size
        val result = IntArray(queries.size)

        val prefixMax = IntArray(n)
        prefixMax[0] = items[0][1]
        for (i in 1 until n) {
            prefixMax[i] = maxOf(prefixMax[i - 1], items[i][1])
        }

        fun findIndex(query: Int): Int {
            var left = 0
            var right = n - 1
            while (left <= right) {
                val mid = left + (right - left) / 2
                if (items[mid][0] <= query) {
                    left = mid + 1 // 找右侧边界
                } else {
                    right = mid - 1
                }
            }
            return if (right in items.indices && items[right][0] <= query) right else -1
        }

        var lastIndex = n - 1
        for ((originalIndex, query) in sortedQueries) {
            val right = findIndex(query)
            if (right == -1 || right < lastIndex) {
                lastIndex = right
            }
            if (lastIndex == -1) break
            result[originalIndex] = prefixMax[lastIndex]
        }

        return result
    }

    fun countFairPairs(nums: IntArray, lower: Int, upper: Int): Long {
//        nums.sort()
//        var count = 0L
//
//        fun findLeftMostIndex(target: Int, left: Int): Int {
//            var left = left
//            var right = nums.size - 1
//            while (left <= right) {
//                val mid = left + (right - left) / 2
//                if (nums[mid] >= target) {
//                    right = mid - 1
//                } else {
//                    left = mid + 1
//                }
//            }
//            return left
//        }
//
//        fun findRightMostIndex(target: Int, left: Int): Int {
//            var left = left
//            var right = nums.size - 1
//            while (left <= right) {
//                val mid = left + (right - left) / 2
//                if (nums[mid] <= target) {
//                    left = mid + 1
//                } else {
//                    right = mid - 1
//                }
//            }
//            return right
//        }
//
//        for (i in nums.indices) {
//            val left = findLeftMostIndex(lower - nums[i], i + 1)
//            val right = findRightMostIndex(upper - nums[i], i + 1)
//            count += right - left + 1
//        }
//        return count

        nums.sort()
        val n = nums.size

        fun countPairsLessThanOrEqual(target: Int): Long {
            var count = 0L
            var left = 0
            var right = n - 1
            while (left < right) {
                if (nums[left] + nums[right] <= target) {
                    count += (right - left).toLong()
                    left++
                } else {
                    right--
                }
            }
            return count
        }
        return countPairsLessThanOrEqual(upper) - countPairsLessThanOrEqual(lower - 1)
    }

    fun minimizedMaximum(n: Int, quantities: IntArray): Int {
        fun canDistribute(k: Int): Boolean {
            var count = 0
            for (quantity in quantities) {
                count += (quantity + k - 1) / k // ceil
            }
            return count <= n
        }

        var left = 1
        var right = 1
        for (quantity in quantities) {
            right = maxOf(right, quantity)
        }

        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (canDistribute(mid)) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }

        return left
    }

    fun findLengthOfShortestSubarray(arr: IntArray): Int {
        val n = arr.size
        var prefix = 0
        for (i in 1 until n) {
            if (arr[i] >= arr[prefix]) {
                prefix++
            } else {
                break
            }
        }
        if (prefix == n - 1) return 0
        var suffix = n - 1
        for (i in n - 2 downTo 0) {
            if (arr[i] <= arr[suffix]) {
                suffix--
            } else {
                break
            }
        }

        var result = n
        var current = prefix
        while (current >= 0) {
            var left = suffix
            var right = n - 1
            while (left <= right) {
                val mid = left + ((right - left) shr 1)
                if (arr[mid] >= arr[current]) {
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            }
            result = minOf(result, left - 1 - current)
            if (left == suffix) break
            current--
        }
        current = suffix
        while (current < n) {
            var left = 0
            var right = prefix
            while (left <= right) {
                val mid = left + ((right - left) shr 1)
                if (arr[mid] <= arr[current]) {
                    left = mid + 1
                } else {
                    right = mid - 1
                }
            }
            result = minOf(result, current - left)
            if (left == prefix) break
            current++
        }
        return result
    }

    fun decryptBomb(code: IntArray, k: Int): IntArray {
        val n = code.size
        val result = IntArray(n)
        if (k == 0) return result
        var sum = 0
        if (k > 0) {
            for (i in 0 until k) {
                sum += code[i]
            }
            for (i in 0 until n) {
                sum = sum + code[(i + k) % n] - code[i]
                result[i] = sum
            }
        } else {
            for (i in n - 1 downTo n + k) {
                sum += code[i]
            }
            for (i in n - 1 downTo 0) {
                sum = sum + code[(i + k + n) % n] - code[i]
                result[i] = sum
            }
        }
        return result
    }

    fun maximumSubarraySum(nums: IntArray, k: Int): Long {
        var maxSum = 0L
        var sum = 0L
        var start = 0
        val seen = BooleanArray(100001)
        for (end in nums.indices) {
            val current = nums[end]
            sum += current
            while (seen[current]) {
                val left = nums[start]
                seen[left] = false
                sum -= left
                start++
                if (left == current) break
            }
            seen[current] = true
            if (end - start + 1 == k) {
                maxSum = maxOf(maxSum, sum)
                seen[nums[start]] = false
                sum -= nums[start]
                start++
            }
        }
        return maxSum
    }

    fun rotateTheBox(box: Array<CharArray>): Array<CharArray> {
        val m = box.size
        val n = box[0].size
        val result = Array(n) { CharArray(m) { '.' } }
        for (rowIndex in 0 until m) {
            var stopAt = n - 1
            for (colIndex in n - 1 downTo 0) {
                when (box[rowIndex][colIndex]) {
                    '*' -> {
                        result[colIndex][m - 1 - rowIndex] = '*'
                        stopAt = colIndex - 1
                    }

                    '#' -> {
                        result[stopAt][m - 1 - rowIndex] = '#'
                        stopAt--
                    }
                }
            }
        }
        return result
    }

    fun maxCount(banned: IntArray, n: Int, maxSum: Int): Int {
//        val bannedSet = banned.toSet()
//        var sum = 0
//        var count = 0
//        for (i in 1..n) {
//            if (i in bannedSet) {
//                continue
//            }
//            sum += i
//            if (sum <= maxSum) {
//                count++
//            } else {
//                break // !!! need this break to quit loop early
//            }
//        }
//        return count

        // prefix
        fun sumUp(from: Int, to: Int): Int {
            return (from + to) * (to - from + 1) shr 1
        }
        banned.sort()
        val bannedPrefix = IntArray(n + 1)
        val bannedCount = IntArray(n + 1)
        var lastBanNum = 0
        for (banNum in banned) {
            if (banNum > n) break
            if (banNum != lastBanNum) {
                bannedPrefix[banNum] = bannedPrefix[lastBanNum] + banNum
                bannedCount[banNum] = bannedCount[lastBanNum] + 1
                lastBanNum = banNum
            }
        }
        for (i in 1..n) {
            if (bannedPrefix[i] == 0) {
                bannedPrefix[i] = bannedPrefix[i - 1]
                bannedCount[i] = bannedCount[i - 1]
            }
        }
        var left = 1
        var right = n
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            val testSum = sumUp(1, mid) - bannedPrefix[mid]
            if (testSum <= maxSum) {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return right - bannedCount[right]
    }

    fun minimumSize(nums: IntArray, maxOperations: Int): Int {
        val totalBags = nums.size + maxOperations

        fun countBags(maxEach: Int): Int {
            var count = 0
            for (num in nums) {
                count += num / maxEach + if (num % maxEach == 0) 0 else 1
            }
            return count
        }

        var left = 1
        var right = nums.maxOf { it }
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            val bags = countBags(mid)
            if (bags <= totalBags) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        return left
    }

    fun maxTwoEvents(events: Array<IntArray>): Int {
        val startPQ = PriorityQueue<IntArray>(compareBy { it[0] })
        val endPQ = PriorityQueue<IntArray>(compareBy { it[1] })
        for (event in events) {
            startPQ.offer(event)
        }
        var currentMax = 0
        var result = 0
        while (startPQ.isNotEmpty()) {
            val currentEvent = startPQ.poll()!!
            while (endPQ.isNotEmpty() && endPQ.peek()!![1] < currentEvent[0]) {
                currentMax = maxOf(currentMax, endPQ.poll()!![2])
            }
            result = maxOf(result, currentMax + currentEvent[2])
            if (currentEvent[2] > currentMax) {
                endPQ.offer(currentEvent)
            }
        }
        return result
    }

    fun isArrayAdjacentDifferentParity(nums: IntArray, queries: Array<IntArray>): BooleanArray {
        val n = nums.size
        val prefixCount = IntArray(n)
        var previousEven = nums[0] % 2 == 0
        for (i in 1 until n) {
            val currentEven = nums[i] % 2 == 0
            prefixCount[i] = prefixCount[i - 1] + if (previousEven xor currentEven) 1 else 0
            previousEven = currentEven
        }
        val result = BooleanArray(queries.size)
        for (i in queries.indices) {
            result[i] =
                prefixCount[queries[i][1]] - prefixCount[queries[i][0]] == queries[i][1] - queries[i][0]
        }
        return result
    }

    fun maximumOverlappingSubsequence(nums: IntArray, k: Int): Int {  // +-k
        nums.sort()
        var start = 0
        var maxCount = 0
        for (end in nums.indices) {
            while (nums[start] + k < nums[end] - k) {
                start++
            }
            maxCount = maxOf(maxCount, end - start + 1)
        }
        return maxCount
    }

    fun findScore(nums: IntArray): Long {
        //        // find descending segments
        //        val n = nums.size
        //        var result = 0L
        //        var start = 0
        //        while (start < n) {
        //            var end = start
        //            while (end < n - 1 && nums[end] > nums[end + 1]) {
        //                end++
        //            }
        //            for (i in end downTo start step 2) {
        //                result += nums[i]
        //            }
        //            start = end + 2
        //        }
        //        return result

        // find ascending segments
        var result = 0L
        var end = nums.size - 1
        while (end >= 0) {
            var i = end
            while (i - 1 >= 0 && nums[i] >= nums[i - 1]) {
                i--
            }
            for (j in i..end step 2) {
                result += nums[j]
            }
            end = i - 2
        }
        return result
    }

    fun continuousSubarrays(nums: IntArray): Long {
        val maxDeque = LinkedList<Int>()
        val minDeque = LinkedList<Int>()
        var start = 0
        var result = 0L

        for (end in nums.indices) {
            while (!maxDeque.isEmpty() && maxDeque.last < nums[end]) {
                maxDeque.removeLast()
            }
            while (!minDeque.isEmpty() && minDeque.last > nums[end]) {
                minDeque.removeLast()
            }
            maxDeque.addLast(nums[end])
            minDeque.addLast(nums[end])

            while (start < end && maxDeque.first - minDeque.first > 2) {
                if (nums[start] == maxDeque.first) {
                    maxDeque.removeFirst()
                }
                if (nums[start] == minDeque.first) {
                    minDeque.removeFirst()
                }
                start++
            }

            if (end >= start) {
                result += end - start + 1
            }
        }

        return result
    }

    fun maxAverageRatio(classes: Array<IntArray>, extraStudents: Int): Double {

        class Student(var pass: Double, var total: Double) : Comparable<Student> {
            var ratio = pass / total
            var prio = (pass + 1) / (total + 1) - ratio

            fun inc() {
                ratio = ++pass / ++total
                prio = (pass + 1) / (total + 1) - ratio
            }

            override fun compareTo(other: Student) = other.prio.compareTo(prio)
        }

        val pq = PriorityQueue<Student>()
        for (student in classes) {
            pq.offer(Student(student[0].toDouble(), student[1].toDouble()))
        }
        for (i in 0 until extraStudents) {
            val student = pq.poll()!!
            student.inc()
            pq.offer(student)
        }
        var sum = 0.0
        for (student in pq) {
            sum += student.ratio
        }
        return sum / classes.size
    }

    fun finalPrices(prices: IntArray): IntArray {
        val result = IntArray(prices.size)
        val stack = LinkedList<Int>()
        for (i in prices.indices.reversed()) {
            while (stack.isNotEmpty() && stack.peek()!! > prices[i]) {
                stack.poll()
            }
            result[i] = prices[i] - (stack.peek() ?: 0)
            stack.push(prices[i])
        }
        return result
    }

    fun findTargetSumWays(nums: IntArray, target: Int): Int {
        // target sum
        val delta = nums.sum() - target
        if (delta < 0 || delta % 2 == 1) return 0

        val half = delta shr 1
        val dp = IntArray(half + 1)
        dp[0] = 1
        for (num in nums) {
            for (i in half downTo num) {
                dp[i] += dp[i - num]
            }
        }

        return dp[half]
    }

    fun maxScoreSightseeingPair(values: IntArray): Int {
        var result = 0
        var maxValue = values[0]
        for (j in 1 until values.size) {
            result = maxOf(result, maxValue + values[j] - j)
            maxValue = maxOf(maxValue, values[j] + j)
        }
        return result
    }

    fun minCostTickets(days: IntArray, costs: IntArray): Int {
        val lastDay = days[days.size - 1]
        var dayIndex = 0
        val dp = IntArray(lastDay + 1)
        val minCost = costs.min()
        for (i in 1..lastDay) {
            if (i != days[dayIndex]) {
                dp[i] = dp[i - 1]
            } else {
                dp[i] = minOf(
                    dp[i - 1] + minCost,
                    (if (i >= 7) dp[i - 7] else 0) + costs[1],
                    (if (i >= 30) dp[i - 30] else 0) + costs[2]
                )
                dayIndex++
            }
        }
        return dp[lastDay]
    }

    fun countServers(grid: Array<IntArray>): Int {
        val m = grid.size
        val n = grid[0].size
        val colsCount = IntArray(n)
        val colsExtra = IntArray(n)
        var result = 0
        for (i in 0 until m) {
            var rowCount = 0
            for (j in 0 until n) {
                if (grid[i][j] == 1) {
                    rowCount++
                    colsCount[j]++
                }
            }
            if (rowCount > 1) {
                result += rowCount
                for (j in 0 until n) {
                    if (grid[i][j] == 1) {
                        colsExtra[j]++
                    }
                }
            }
        }
        for (j in 0 until n) {
            if (colsCount[j] > 1) {
                result += colsCount[j] - colsExtra[j]
            }
        }
        return result
    }

    fun lexicographicallySmallestArray(nums: IntArray, limit: Int): IntArray {
        val sorted = nums.withIndex().sortedBy { it.value }
        val indices = PriorityQueue<Int>()
        var rangeMax = 0
        val result = IntArray(nums.size)
        var j = 0
        for (i in sorted.indices) {
            val (index, value) = sorted[i]
            if (value > rangeMax) {
                while (j < i) {
                    result[indices.poll()!!] = sorted[j++].value
                }
            }
            indices.offer(index)
            rangeMax = maxOf(rangeMax, value + limit)
        }
        while (indices.isNotEmpty()) {
            result[indices.poll()!!] = sorted[j++].value
        }
        return result
    }

    fun checkIfPrerequisiteDFS(
        numCourses: Int,
        prerequisites: Array<IntArray>,
        queries: Array<IntArray>
    ): List<Boolean> {
        val graph = Array(numCourses) { mutableListOf<Int>() }
        for ((u, v) in prerequisites) {
            graph[u].add(v)
        }
        val dependencies = Array(numCourses) { BooleanArray(numCourses) }

        fun dfs(start: Int, current: Int) {
            for (neighbour in graph[current]) {
                if (!dependencies[start][neighbour]) {
                    dependencies[start][neighbour] = true
                    dfs(start, neighbour)
                }
            }
        }

        for (i in 0 until numCourses) {
            dfs(i, i)
        }

        val result = mutableListOf<Boolean>()
        for ((u, v) in queries) {
            result.add(dependencies[u][v])
        }
        return result
    }

    fun checkIfPrerequisite(
        numCourses: Int,
        prerequisites: Array<IntArray>,
        queries: Array<IntArray>
    ): List<Boolean> {
        val dependencies = Array(numCourses) { BooleanArray(numCourses) }
        for ((u, v) in prerequisites) {
            dependencies[u][v] = true
        }
        // Floyd-Warshall
        for (k in 0 until numCourses) {
            for (u in 0 until numCourses) {
                for (v in 0 until numCourses) {
                    if (dependencies[u][k] && dependencies[k][v]) {
                        dependencies[u][v] = true
                    }
                }
            }
        }

        val result = mutableListOf<Boolean>()
        for ((u, v) in queries) {
            result.add(dependencies[u][v])
        }
        return result
    }

    fun longestMonotonicSubarray(nums: IntArray): Int {
        var maxLength = 1
        var currAsc = 1
        var currDes = 1
        for (i in 1 until nums.size) {
            if (nums[i] > nums[i - 1]) {
                currAsc++
                currDes = 1
            } else if (nums[i] < nums[i - 1]) {
                currDes++
                currAsc = 1
            } else {
                currDes = 1
                currAsc = 1
            }
            maxLength = maxOf(maxLength, currAsc, currDes)
        }
        return maxLength
    }

    fun assignElements(groups: IntArray, elements: IntArray): IntArray {
        val result = IntArray(groups.size) { -1 }
        var maxGroup = 0
        for (group in groups) {
            maxGroup = maxOf(maxGroup, group)
        }
        val minPos = IntArray(maxGroup + 1) { -1 }
        for (i in elements.indices) {
            val element = elements[i]
            if (element <= maxGroup && minPos[element] == -1) {
                minPos[element] = i
            }
        }
        val memo = mutableMapOf<Int, Int>()
        for (i in groups.indices) {
            if (memo.contains(groups[i])) {
                result[i] = memo[groups[i]]!!
                continue
            }
            val pq = PriorityQueue<Int>()
            val sqrt = Math.sqrt(groups[i].toDouble()).toInt()
            for (factor in 1..sqrt) {
                if (groups[i] % factor == 0) {
                    if (minPos[factor] != -1) {
                        pq.offer(minPos[factor])
                    }
                    val other = groups[i] / factor
                    if (other != factor && minPos[other] != -1) {
                        pq.offer(minPos[other])
                    }
                }
            }
            result[i] = if (pq.isNotEmpty()) {
                pq.peek()!!
            } else -1
            memo[groups[i]] = result[i]
        }
        return result
    }

    fun clearDigits(s: String): String {
        val array = CharArray(s.length) { ' ' }
        var index = 0
        for (c in s) {
            if (c in 'a'..'z') {
                array[index++] = c
            } else {
                array[--index] = ' '
            }
        }
        val sb = StringBuilder()
        for (i in array.indices) {
            if (array[i] != ' ') {
                sb.append(array[i])
            }
        }
        return sb.toString()
    }

    fun constructDistancedSequence(n: Int): IntArray {
        // backtracking
        val size = 2 * n - 1
        val array = IntArray(size)
        var used = 0

        fun dfs(index: Int): Boolean {
            if (index >= size) return true
            if (array[index] != 0) return dfs(index + 1)
            for (num in n downTo 1) {
                val bit = 1 shl num
                if (used and bit == 0) {
                    if (num == 1) {
                        used = used xor bit
                        array[index] = num
                        if (dfs(index + 1)) return true
                        used = used xor bit
                        array[index] = 0
                    } else if (index + num < size && array[index + num] == 0) {
                        used = used xor bit
                        array[index] = num
                        array[index + num] = num
                        if (dfs(index + 1)) return true
                        used = used xor bit
                        array[index] = 0
                        array[index + num] = 0
                    }
                }
            }
            return false
        }

        dfs(0)
        return array
    }

    fun findDifferentBinaryString(nums: Array<String>): String {
        //        val n = nums.size
        //        val array = CharArray(n)
        //        for (i in 0 until n) {
        //            if (nums[i][i] == '1') {
        //                array[i] = '0'
        //            } else {
        //                array[i] = '1'
        //            }
        //        }
        //        return String(array)
        val n = nums.size
        val seen = BooleanArray(1 shl n)
        for (numString in nums) {
            val numInt = numString.toInt(2)
            seen[numInt] = true
        }
        var current = 0
        var result = 0

        fun dfs(index: Int): Boolean {
            if (index == n) {
                if (!seen[current]) {
                    result = current
                    return false
                }
                return true
            }
            current = current shl 1
            var keep = dfs(index + 1)
            if (keep) {
                current += 1
                keep = dfs(index + 1)
            }
            current = current shr 1
            return keep
        }

        dfs(0)
        return result.toString(2)
    }

    fun mostProfitablePath(edges: Array<IntArray>, bob: Int, amount: IntArray): Int {
        var n = 1
        val edgesMap = mutableMapOf<Int, MutableList<Int>>()
        for ((a, b) in edges) {
            n++
            edgesMap.getOrPut(a) { mutableListOf() }.add(b)
            edgesMap.getOrPut(b) { mutableListOf() }.add(a)
        }
        val bPath = LinkedList<Int>()
        val currentPath = mutableListOf(bob)

        fun dfs(index: Int, previous: Int): Boolean {
            if (index == 0) {
                bPath.addAll(currentPath)
                return false
            }
            var shouldContinue = true
            for (next in edgesMap[index]!!) {
                if (next == previous) continue
                currentPath.add(next)
                shouldContinue = dfs(next, index)
                currentPath.removeAt(currentPath.size - 1)
                if (!shouldContinue) break
            }
            return shouldContinue
        }

        dfs(bob, -1)
        amount[bPath.removeFirst()] = 0
        var maxProfit = Int.MIN_VALUE
        val queue = LinkedList<Pair<Int, Int>>()
        queue.offer(0 to amount[0])
        val visited = BooleanArray(n)
        visited[0] = true
        while (queue.isNotEmpty()) {
            val bobIndex = if (bPath.isNotEmpty()) bPath.removeFirst() else -1
            val levelSize = queue.size
            repeat(levelSize) {
                val (node, profit) = queue.poll()!!
                if (edgesMap[node]!!.size == 1 && visited[edgesMap[node]!![0]]) {
                    maxProfit = maxOf(maxProfit, profit)
                } else {
                    for (next in edgesMap[node]!!) {
                        if (visited[next]) continue
                        val newProfit = if (next == bobIndex) {
                            profit + (amount[next] shr 1)
                        } else {
                            profit + amount[next]
                        }
                        queue.offer(next to newProfit)
                        visited[next] = true
                    }
                }
            }
            if (bobIndex != -1) {
                amount[bobIndex] = 0
            }
        }
        return maxProfit
    }

    fun numOfSubarrays(arr: IntArray): Int {
        var result = 0
        var oddCount = 0
        var prefix = 0
        for (i in arr.indices) {
            prefix = (prefix + arr[i]) % 2
            if (prefix == 0) {
                result = (result + oddCount) % MODULO
            } else {
                result = (result + 1 + i - oddCount) % MODULO
                oddCount++
            }
        }
        return result
    }

    fun lenLongestFibSubseq(arr: IntArray): Int {
        val n = arr.size
        val map = mutableMapOf<Int, Int>()
        for (i in arr.indices) {
            map[arr[i]] = i
        }
        var maxLength = 0
        //        fun dfs(index: Int, previous: Int, length: Int) {
        //            if (arr[previous] < Int.MAX_VALUE - arr[index]) {
        //                val next = map.getOrDefault(arr[previous] + arr[index], -1)
        //                if (next != -1) {
        //                    dfs(next, index, length + 1)
        //                }
        //            }
        //            if (length >= 3) {
        //                maxLength = maxOf(maxLength, length)
        //            }
        //        }
        //
        //        for (i in 0 until n) {
        //            for (j in i + 1 until n) {
        //                dfs(j, i, 2)
        //            }
        //        }
        val dp = Array(n) { IntArray(n) { 2 } }
        for (j in 0 until n) {
            for (i in 0 until j) {
                val k = map.getOrDefault(arr[j] - arr[i], -1)
                if (k in 0 until i) {
                    dp[i][j] = dp[k][i] + 1
                    if (dp[i][j] >= 2) {
                        maxLength = maxOf(maxLength, dp[i][j])
                    }
                }
            }
        }
        return maxLength
    }

    fun rearrangeArray(nums: IntArray): IntArray {
        val n = nums.size
        val result = IntArray(n)
        var index = 0
        var negIndex = 0
        var posIndex = 0
        var positive = true
        while (index < n) {
            if (positive) {
                while (nums[posIndex] < 0) posIndex++
                result[index++] = nums[posIndex++]
            } else {
                while (nums[negIndex] > 0) negIndex++
                result[index++] = nums[negIndex++]
            }
            positive = !positive
        }
        return result
    }

    fun closestPrimes(left: Int, right: Int): IntArray {
        val limit = Math.sqrt(right.toDouble()).toInt()
        val isSmallPrime = BooleanArray(limit + 1) { true }
        isSmallPrime[0] = false
        isSmallPrime[1] = false
        for (i in 2..limit) {
            if (isSmallPrime[i]) {
                for (j in i * i..limit) {
                    isSmallPrime[j] = true
                }
            }
        }
        val isPrime = BooleanArray(right - left + 1) { true }
        if (left == 1) {
            isPrime[0] = false
        }
        for (i in 2..limit) {
            if (isSmallPrime[i]) {
                val start = maxOf(i * i, (left + i - 1) / i * i)
                for (j in start..right step i) {
                    isPrime[j - left] = false
                }
            }
        }
        var diff = Int.MAX_VALUE
        val result = intArrayOf(-1, -1)
        var previous = -1
        for (i in left..right) {
            if (isPrime[i - left]) {
                if (previous != -1 && i - previous < diff) {
                    diff = i - previous
                    result[0] = previous
                    result[1] = i
                }
                previous = i
            }
        }
        return result
    }

    fun numberOfAlternatingGroups(colors: IntArray, k: Int): Int {
        val n = colors.size
        var count = 0
        var last = -1
        var start = 0
        for (end in 0 until n + k - 1) {
            if (colors[end % n] == last) {
                start = end
            }
            if (end - start + 1 > k) {
                start++
            }
            if (end - start + 1 == k) {
                count++
            }
            last = colors[end % n]
        }
        return count
    }

    fun countOfSubstrings(word: String, k: Int): Long {
        // a, e, i, o, u at least 1 each; consonants count k
        val vowelFreq = IntArray(5)
        var consonantCount = 0
        var vowelCount = 0
        var leadings = 0

        fun isVowel(letter: Char): Boolean {
            return letter == 'a' || letter == 'e' || letter == 'i' || letter == 'o' || letter == 'u'
        }

        fun addLetter(letter: Char) {
            when (letter) {
                'a' -> if (vowelFreq[0]++ == 0) vowelCount++
                'e' -> if (vowelFreq[1]++ == 0) vowelCount++
                'i' -> if (vowelFreq[2]++ == 0) vowelCount++
                'o' -> if (vowelFreq[3]++ == 0) vowelCount++
                'u' -> if (vowelFreq[4]++ == 0) vowelCount++
                else -> consonantCount++
            }
        }

        fun removeLetter(letter: Char) {
            when (letter) {
                'a' -> if (--vowelFreq[0] == 0) vowelCount--
                'e' -> if (--vowelFreq[1] == 0) vowelCount--
                'i' -> if (--vowelFreq[2] == 0) vowelCount--
                'o' -> if (--vowelFreq[3] == 0) vowelCount--
                'u' -> if (--vowelFreq[4] == 0) vowelCount--
                else -> consonantCount--
            }
        }

        var result = 0L
        var start = 0
        for (end in word.indices) {
            addLetter(word[end])
            while (consonantCount > k) {
                removeLetter(word[start++])
                leadings = 0
            }
            while (vowelCount == 5 && isVowel(word[start])) {
                removeLetter(word[start++])
                leadings++
                if (vowelCount != 5) {
                    addLetter(word[--start])
                    leadings--
                    break
                }
            }
            if (vowelCount == 5 && consonantCount == k) {
                result += 1 + leadings
            }
        }
        return result
    }

    fun numberOfSubstrings(s: String): Int {
        // a, b, c at least 1 each
        //        val lastSeen = IntArray(3) { -1 }
        //        var result = 0
        //        for (i in s.indices) {
        //            lastSeen[s[i] - 'a'] = i
        //            result += minOf(lastSeen[0], lastSeen[1], lastSeen[2]) + 1
        //        }
        //        return result
        val n = s.length
        val freq = IntArray(3)
        var result = 0
        var start = 0
        for (end in s.indices) {
            freq[s[end] - 'a']++
            while (freq[0] > 0 && freq[1] > 0 && freq[2] > 0) {
                result += n - end
                freq[s[start++] - 'a']--
            }
        }
        return result
    }

    fun maximumCount(nums: IntArray): Int {
        // sorted array, find max between pos count and neg count
        val n = nums.size
        var left = 0
        var right = n - 1
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (nums[mid] >= 0) { // shrink right to find left-most non negative
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        val negCount = left
        left = 0
        right = n - 1
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (nums[mid] > 0) { // shrink right to find left-most positive
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        val posCount = n - left
        return maxOf(negCount, posCount)
    }

    fun countNegatives(grid: Array<IntArray>): Int {
        // staircase search
        //        var count = 0
        //        var j = 0
        //        for (i in m - 1 downTo 0) {
        //            while (j < n && grid[i][j] >= 0) {
        //                j++
        //            }
        //            if (j == n) {
        //                break
        //            }
        //            count += n - j
        //        }
        //        return count
        val m = grid.size
        val n = grid[0].size
        var count = 0
        var left = 0
        var right = n - 1
        for (i in 0 until m) {
            while (left <= right) {
                val mid = (left + right) shr 1
                if (grid[i][mid] < 0) {
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            }
            count += n - left
            left = 0
        }
        return count
    }

    fun minZeroArray(nums: IntArray, queries: Array<IntArray>): Int {
//        val n = nums.size
//        val changes = IntArray(n)
//        var current = 0
//        var index = 0
//        while (index < n && nums[index] == 0) {
//            index++
//        }
//        if (index == n) {
//            return 0
//        }
//        for (i in queries.indices) {
//            val (l, r, v) = queries[i]
//            if (index > r) continue
//            changes[maxOf(l, index)] += v
//            if (r != n - 1) changes[r + 1] -= v
//            while (index < n && nums[index] <= current + changes[index]) {
//                current += changes[index]
//                index++
//            }
//            if (index == n) {
//                return i + 1
//            }
//        }
//        return -1

        val n = nums.size
        val changes = IntArray(n)
        var current = 0
        var k = 0
        for (i in nums.indices) {
            while (current + changes[i] < nums[i]) {
                if (k == queries.size) return -1
                val (l, r, v) = queries[k++]
                if (i > r) continue
                changes[maxOf(l, i)] += v
                if (r != n - 1) changes[r + 1] -= v
            }
            current += changes[i]
        }
        return k
    }

    fun minEatingSpeed(piles: IntArray, h: Int): Int {

        fun canFinish(k: Int): Boolean {
            var time = 0
            for (pile in piles) {
                time += (pile + k - 1) / k
                if (time > h) return false
            }
            return true
        }

        var left = 1
        var right = 0
        for (pile in piles) {
            right = maxOf(right, pile)
        }
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (canFinish(mid)) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        return left
    }

    fun minCapability(nums: IntArray, k: Int): Int {
        val n = nums.size

        //        val dp = Array(n + 1) { IntArray(k + 1) { Int.MAX_VALUE } }
//        for (i in 0..n) {
//            dp[i][0] = 0
//        }
//        for (i in 1..n) {
//            for (j in 1..minOf(k, i)) {
//                dp[i][j] =
//                    minOf(dp[i - 1][j], maxOf(nums[i - 1], if (i > 1) dp[i - 2][j - 1] else 0))
//            }
//        }
//        return dp[nums.size][k]
        fun canRob(take: Int): Boolean {
            var count = 0
            var start = -1
            for (i in 0..n) {
                if (i == n || nums[i] > take) {
                    if (start != -1) {
                        count += (i - start + 1) shr 1
                        start = -1
                    }
                } else {
                    if (start == -1) {
                        start = i
                    }
                }
                if (count >= k) return true
            }
            return false
        }

        var left = Int.MAX_VALUE
        var right = 0
        for (num in nums) {
            left = minOf(left, num)
            right = maxOf(right, num)
        }
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (canRob(mid)) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        return left
    }

    fun maxJump(stones: IntArray): Int {
        //        var result = stones[1] - stones[0]
        //        for (i in 2 until stones.size) {
        //            result = maxOf(result, stones[i] - stones[i - 2])
        //        }
        //        return result
        val n = stones.size
        val taken = BooleanArray(n)

        fun findRightMost(target: Int): Int {
            var left = 0
            var right = n - 1
            while (left <= right) {
                val mid = left + ((right - left) shr 1)
                if (stones[mid] <= target) {
                    left = mid + 1
                } else {
                    right = mid - 1
                }
            }
            return right
        }

        fun findLeftMost(target: Int): Int {
            var left = 0
            var right = n - 1
            while (left <= right) {
                val mid = left + ((right - left) shr 1)
                if (stones[mid] >= target) {
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            }
            return left
        }

        fun canJump(index: Int, maxCost: Int, back: Boolean): Boolean {
            if (index <= 0 && back) return true
            var valid = false
            if (back) {
                val farest = findLeftMost(stones[index] - maxCost)
                for (i in farest until index) {
                    if (taken[i]) continue
                    taken[i] = true
                    valid = canJump(i, maxCost, back)
                    taken[i] = false
                    break
                }
            } else {
                val farest = findRightMost(stones[index] + maxCost)
                for (i in farest downTo index + 1) {
                    if (farest == n - 1) {
                        valid = canJump(i, maxCost, true)
                        break
                    }
                    if (taken[i]) continue
                    taken[i] = true
                    valid = canJump(i, maxCost, back)
                    taken[i] = false
                    break
                }
            }
            return valid
        }

        var left = Int.MAX_VALUE
        for (i in 1 until n) {
            left = maxOf(stones[i] - stones[i - 1])
        }
        var right = stones[n - 1] - stones[0]
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (canJump(0, mid, false)) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        return left
    }

    fun divideArrayToPairs(nums: IntArray): Boolean {
        val seen = BooleanArray(501)
        for (num in nums) {
            seen[num] = !seen[num]
        }
        for (num in nums) {
            if (seen[num]) return false
        }
        return true
    }

    fun splitArrayMinimumSum(nums: IntArray, k: Int): Int {

        fun canGroup(maxSum: Int): Boolean {
            var count = 0
            var currentSum = 0
            for (num in nums) {
                if (currentSum + num > maxSum) {
                    count++
                    currentSum = 0
                }
                currentSum += num
                if (count > k) return false
            }
            count++
            return count <= k
        }

        var left = 0
        var right = 0
        for (num in nums) {
            left = maxOf(left, num)
            right += num
        }
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (canGroup(mid)) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        return left
    }

    fun countPairsSumPowersOfTwo(deliciousness: IntArray): Int {
        val freq = mutableMapOf<Int, Int>()
        var maxNum = 0
        for (num in deliciousness) {
            freq[num] = freq.getOrDefault(num, 0) + 1
            maxNum = maxOf(maxNum, num)
        }
        var maxTarget = 2 shl 21
        while (maxTarget shr 1 > maxNum) {
            maxTarget = maxTarget shr 1
        }
        var result = 0
        for (num in freq.keys) {
            var target = maxTarget
            while (target != 0) {
                val other = target - num
                if (other >= num && freq.containsKey(other)) {
                    val ways = if (other == num) {
                        1L * freq[num]!! * (freq[num]!! - 1) / 2
                    } else {
                        1L * freq[num]!! * freq[other]!!
                    }
                    result = (result + (ways % MODULO).toInt()) % MODULO
                }
                target = target shr 1
            }
        }
        return result
    }

    fun countCompleteComponents(n: Int, edges: Array<IntArray>): Int {
        class UnionFind(n: Int) {
            val parent = IntArray(n) { it }
            val rank = IntArray(n)

            fun find(x: Int): Int {
                if (parent[x] != x) {
                    parent[x] = find(parent[x])
                }
                return parent[x]
            }

            fun union(x: Int, y: Int) {
                val rootX = find(x)
                val rootY = find(y)
                if (rootX != rootY) {
                    if (rank[rootX] > rank[rootY]) {
                        parent[rootY] = rootX
                    } else if (rank[rootX] < rank[rootY]) {
                        parent[rootX] = rootY
                    } else {
                        parent[rootY] = rootX
                        rank[rootX]++
                    }
                }
            }
        }

        val uf = UnionFind(n)
        val degree = IntArray(n)
        for ((u, v) in edges) {
            uf.union(u, v)
            degree[u]++
            degree[v]++
        }
        val groups = Array(n) { mutableListOf<Int>() }
        for (i in 0 until n) {
            val root = uf.find(i)
            groups[root].add(i)
        }
        var result = 0
        for (nodes in groups) {
            val nodeCount = nodes.size
            if (nodeCount == 0) continue
            var complete = true
            for (node in nodes) {
                if (degree[node] != nodeCount - 1) {
                    complete = false
                    break
                }
            }
            if (complete) {
                result++
            }
        }
        return result
    }

    fun checkValidCuts(n: Int, rectangles: Array<IntArray>): Boolean {
        val horizontal = rectangles.sortedBy { it[0] }
        var lastEnd = -1
        var cut = 0
        for ((start, _, end, _) in horizontal) {
            if (lastEnd != -1 && start >= lastEnd) {
                cut++
            }
            lastEnd = maxOf(lastEnd, end)
            if (cut == 2) return true
        }
        val vertical = rectangles.sortedBy { it[1] }
        lastEnd = -1
        cut = 0
        for ((_, start, _, end) in vertical) {
            if (lastEnd != -1 && start >= lastEnd) {
                cut++
            }
            lastEnd = maxOf(lastEnd, end)
            if (cut == 2) return true
        }
        return false
    }

    fun minOperationsUniValue(grid: Array<IntArray>, x: Int): Int {
        val m = grid.size
        val n = grid[0].size
        val gridSize = m * n
        val nums = IntArray(gridSize) { index ->
            grid[index / n][index % n]
        }
        nums.sort()
        val target = nums[gridSize / 2]
        var result = 0
        for (num in nums) {
            val diff = Math.abs(num - target)
            if (diff % x != 0) return -1
            result += diff / x
        }
        return result
    }

    fun minimumIndexToSplitDominant(nums: List<Int>): Int {
        // Boyer-Moore Voting
        val n = nums.size
        var candidate = -1
        var count = 0
        for (num in nums) {
            if (count == 0) {
                candidate = num
            }
            if (num == candidate) {
                count++
            } else {
                count--
            }
        }
        var total = 0
        for (num in nums) {
            if (num == candidate) total++
        }
        if (total <= n / 2) return -1
        count = 0
        for (i in 0 until n) {
            if (nums[i] == candidate) {
                count++
                if (count > (i + 1) / 2 && total - count > (n - 1 - i) / 2) return i
            }
        }
        return -1
    }

    fun partitionLabels(s: String): List<Int> {
        val ends = IntArray(26) { -1 }
        for (i in s.indices) {
            ends[s[i] - 'a'] = i
        }
        val result = mutableListOf<Int>()
        var start = 0
        var end = -1
        for (i in s.indices) {
            end = maxOf(end, ends[s[i] - 'a'])
            if (end == i) {
                result.add(end - start + 1)
                start = end + 1
            }
        }
        return result
    }

    fun mostPoints(questions: Array<IntArray>): Long {
        val n = questions.size
        val dp = LongArray(n)
        dp[n - 1] = questions[n - 1][0].toLong()
        for (i in n - 2 downTo 0) {
            val (point, power) = questions[i]
            dp[i] = maxOf(dp[i + 1], point + if (i + power + 1 < n) dp[i + power + 1] else 0)
        }
        return dp[0]
    }

    fun maximumTripletValue(nums: IntArray): Long {
        var result = 0L
        var maxSoFar = 0
        var maxDiff = 0
        for (num in nums) {
            result = maxOf(result, maxDiff.toLong() * num)
            maxDiff = maxOf(maxDiff, maxSoFar - num)
            maxSoFar = maxOf(maxSoFar, num)
        }
        return result
    }

    fun largestDivisibleSubset(nums: IntArray): List<Int> {
        val n = nums.size
        nums.sort()
        var maxIndex = -1
        val prev = IntArray(n) { -1 }
        val dp = IntArray(n) { 1 }
        for (i in 0 until n) {
            for (j in i - 1 downTo 0) {
                if (dp[i] < dp[j] + 1 && nums[i] % nums[j] == 0) {
                    dp[i] = dp[j] + 1
                    prev[i] = j
                }
            }
            if (maxIndex == -1 || dp[i] > dp[maxIndex]) {
                maxIndex = i
            }
        }
        val result = mutableListOf<Int>()
        while (maxIndex != -1) {
            result.add(nums[maxIndex])
            maxIndex = prev[maxIndex]
        }
        return result
    }

    fun canPartitionEqualSubsets(nums: IntArray): Boolean {
        // 0/1 pack
        var total = 0
        for (num in nums) {
            total += num
        }
        if (total % 2 != 0) return false
        val target = total shr 1
        val dp = BooleanArray(target + 1)
        dp[0] = true
        for (num in nums) {
            for (j in target downTo num) {
                dp[j] = dp[j] || dp[j - num]
            }
        }
        return dp[target]
    }

    fun minimumOperationsMakeDistinct(nums: IntArray): Int {
        // everytime remove 3 first num
        val seen = BooleanArray(101)
        var index = nums.size - 1
        while (index >= 0) {
            if (seen[nums[index]]) break
            seen[nums[index--]] = true
        }
        return (index + 3) / 3
    }

    fun countGoodTriplets(arr: IntArray, a: Int, b: Int, c: Int): Int {
        // |arr[i] - arr[j]| <= a
        // |arr[j] - arr[k]| <= b
        // |arr[i] - arr[k]| <= c
        val n = arr.size
        if (n < 3) return 0

        var maxVal = 0
        for (num in arr) {
            maxVal = maxOf(maxVal, num)
        }
        val freq = IntArray(maxVal + 1)
        for (k in 2 until n) {
            freq[arr[k]]++
        }

        var count = 0
        for (j in 1 until n - 1) {
            val prefix = IntArray(maxVal + 1)
            prefix[0] = freq[0]
            for (x in 1..maxVal) {
                prefix[x] = prefix[x - 1] + freq[x]
            }

            for (i in 0 until j) {
                if (abs(arr[i] - arr[j]) > a) continue

                val lower = maxOf(arr[j] - b, arr[i] - c)
                val upper = minOf(arr[j] + b, arr[i] + c)
                if (lower > upper) continue

                val L = maxOf(0, lower)
                val R = minOf(maxVal, upper)
                val segCount = if (L > 0) prefix[R] - prefix[L - 1] else prefix[R]
                count += segCount
            }
            if (j + 1 < n) {
                freq[arr[j + 1]]--
            }
        }
        return count
    }

    class FenwickTree(private val n: Int) {
        private val tree = IntArray(n + 1)

        // 单点更新：在 index 处增加 value（index 从 1 开始）
        fun update(index: Int, value: Int) {
            var i = index
            while (i <= n) {
                tree[i] += value
                i += i and -i  // 利用二进制位，跳转到下一个需要更新的位置
            }
        }

        // 前缀和查询：计算从 1 到 index 的累积和
        fun query(index: Int): Int {
            var i = index
            var sum = 0
            while (i > 0) {
                sum += tree[i]
                i -= i and -i  // 回退到上一个贡献值的位置
            }
            return sum
        }

        // 区间查询：计算 [l, r] 的区间和
        fun query(l: Int, r: Int): Int {
            return query(r) - query(l - 1)
        }
    }

    fun countEqualAtLeastKPairs(nums: IntArray, k: Int): Long {
        val freq = mutableMapOf<Int, Int>()
        var end = 0
        var pairs = 0
        var count = 0L
        for (start in nums.indices) {
            while (end < nums.size && pairs < k) {
                pairs += freq[nums[end]] ?: 0
                freq[nums[end]] = freq.getOrDefault(nums[end], 0) + 1
                end++
            }
            if (pairs >= k) {
                count += nums.size - end + 1
            }

            freq[nums[start]] = freq[nums[start]]!! - 1
            pairs -= freq[nums[start]] ?: 0
        }
        return count
    }

    fun bruteBuild(n: Int, maxValue: Int) {

        fun dfs(current: MutableList<Int>) {
            if (current.size == n) {
                println(current.joinToString())
                return
            }
            for (num in 1..maxValue) {
                if (current.isEmpty() || num % current.last() == 0) {
                    current.add(num)
                    dfs(current)
                    current.removeAt(current.lastIndex)
                }
            }
        }

        dfs(mutableListOf())
    }

    fun countInterestingSubarrays(nums: List<Int>, modulo: Int, k: Int): Long {
        var result = 0L
        val freq = mutableMapOf<Int, Long>()
        freq[0] = 1L
        var current = 0
        for (num in nums) {
            current = (current + if (num % modulo == k) 1 else 0) % modulo
            val target = (current - k + modulo) % modulo
            result += freq.getOrDefault(target, 0L)
            freq[current] = freq.getOrDefault(current, 0L) + 1
        }
        return result
    }

    fun pushDominoes(dominoes: String): String {
        val status = dominoes.toCharArray()
        var last = -1
        var pendingRight = false
        for (i in 0..status.size) {
            if (i == status.size || status[i] == 'R') {
                if (pendingRight) {
                    for (j in last + 1 until i) {
                        status[j] = 'R'
                    }
                }
                last = i
                pendingRight = true
            } else if (status[i] == 'L') {
                if (!pendingRight) {
                    for (j in last + 1 until i) {
                        status[j] = 'L'
                    }
                } else {
                    val distance = (i - last - 1) shr 1
                    for (step in 1..distance) {
                        status[last + step] = 'R'
                        status[i - step] = 'L'
                    }
                }
                last = i
                pendingRight = false
            }
        }
        return String(status)
    }

    fun minDominoRotations(tops: IntArray, bottoms: IntArray): Int {
        val topFreq = IntArray(7)
        val botFreq = IntArray(7)
        val same = IntArray(7)
        val n = tops.size
        for (i in 0 until n) {
            if (tops[i] == bottoms[i]) {
                same[tops[i]]++
            } else {
                topFreq[tops[i]]++
                botFreq[bottoms[i]]++
            }
        }
        var result = n
        var num = tops[0]
        if (topFreq[num] + botFreq[num] + same[num] == n) {
            result = minOf(result, topFreq[num], botFreq[num])
        }
        num = bottoms[0]
        if (topFreq[num] + botFreq[num] + same[num] == n) {
            result = minOf(result, topFreq[num], botFreq[num])
        }
        return if (result == n) -1 else result
    }

    fun numTilings(n: Int): Int {
        //        val dp = LongArray(n + 1)
        //        dp[0] = 1L
        //        for (i in 1..n) {
        //            dp[i] = if (i == 1) 1L
        //            else if (i == 2) 2L
        //            else (dp[i - 1] * 2 + dp[i - 3]) % MODULO
        //        }
        //        return dp[n].toInt()

        //  0  1  2
        // oo ox oo
        // ox oo oo
        val dp = Array(n + 1) { LongArray(3) }
        dp[0][2] = 1L
        dp[1][2] = 1L
        for (i in 2..n) {
            dp[i][0] = (dp[i - 1][1] + dp[i - 2][2]) % MODULO
            dp[i][1] = (dp[i - 1][0] + dp[i - 2][2]) % MODULO
            dp[i][2] = (dp[i - 1][2] + dp[i - 2][2] + dp[i - 1][0] + dp[i - 1][1]) % MODULO
        }
        return dp[n][2].toInt()
    }

    fun buildArrayInPlace(nums: IntArray): IntArray {
//        var tmp: Int
//        var next: Int
//        for (i in nums.indices) {
//            next = i
//            val first = nums[next]
//            while (nums[next] >= 0) {
//                tmp = nums[next]
//                if (nums[nums[next]] < 0) {
//                    nums[next] = -first - 1
//                    break
//                }
//                nums[next] = -nums[nums[next]] - 1
//                next = tmp
//            }
//        }
//        for (i in nums.indices) {
//            nums[i] = -(nums[i] + 1)
//        }
//        return nums
        val n = nums.size
        for (i in 0 until n) {
            nums[i] += (nums[nums[i] % n] % n) * n
        }
        for (i in 0 until n) {
            nums[i] /= n
        }
        return nums
    }

    fun minTimeToReach(moveTime: Array<IntArray>): Int {
        val DIRECTIONS = arrayOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)
        val m = moveTime.size
        val n = moveTime[0].size
        val pq = PriorityQueue<Triple<Int, Int, Int>>(compareBy { it.third })
        val minTime = Array(m) { IntArray(n) { Int.MAX_VALUE } }
        pq.offer(Triple(0, 0, 0))
        minTime[0][0] = 0 // 或者使用visited，在入队就标记。不能在出队才标记。
        while (pq.isNotEmpty()) {
            val (x, y, time) = pq.poll()!!
            if (x == m - 1 && y == n - 1) return time
            for ((dx, dy) in DIRECTIONS) {
                val newX = x + dx
                val newY = y + dy
                if (newX in 0 until m && newY in 0 until n) {
                    val newTime = maxOf(time + 1, moveTime[newX][newY] + 1)
                    if (newTime < minTime[newX][newY]) {
                        minTime[newX][newY] = newTime
                        pq.offer(Triple(newX, newY, newTime))
                    }
                }
            }
        }
        return 0
    }

    fun minTimeToReachShiftCost(moveTime: Array<IntArray>): Int {
        // cost shifting between 1 and 2
        val DIRECTIONS = arrayOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)

        data class Node(val x: Int, val y: Int, val time: Int, val cost: Int)

        val m = moveTime.size
        val n = moveTime[0].size
        val pq = PriorityQueue<Node>(compareBy { it.time })
        val visited = Array(m) { BooleanArray(n) }
        pq.offer(Node(0, 0, 0, 1))
        visited[0][0] = true
        while (pq.isNotEmpty()) {
            val node = pq.poll()!!
            if (node.x == m - 1 && node.y == n - 1) return node.time
            for ((dx, dy) in DIRECTIONS) {
                val newX = node.x + dx
                val newY = node.y + dy
                val newCost = if (node.cost == 1) 2 else 1
                if (newX in 0 until m && newY in 0 until n && !visited[newX][newY]) {
                    visited[newX][newY] = true
                    pq.offer(
                        Node(
                            newX,
                            newY,
                            maxOf(node.time, moveTime[newX][newY]) + node.cost,
                            newCost
                        )
                    )
                }
            }
        }
        return 0
    }

    fun minOperations(nums: IntArray): Int {
        val map = TreeMap<Int, LinkedList<Int>>()
        for (i in nums.indices) {
            map.getOrPut(nums[i]) { LinkedList() }.addFirst(i + 1) // 1..n
        }
        val splits = TreeSet<Int>().apply {
            add(0)
            add(nums.size + 1)
        }
        var result = 0
        while (map.isNotEmpty()) {
            val (num, ids) = map.pollFirstEntry()!!
            var lastStart = -1
            for (id in ids) {
                val start = splits.floor(id)
                if (start != lastStart) {
                    lastStart = start
                    result++
                }
                splits.add(id)
            }
        }
        return result
    }

    fun getWordsInLongestSubsequence(words: Array<String>, groups: IntArray): List<String> {

        fun isValid(word1: String, word2: String): Boolean {
            if (word1.length != word2.length) return false
            var distance = 0
            for (i in word1.indices) {
                if (word1[i] != word2[i]) {
                    if (++distance > 1) return false
                }
            }
            return true
        }

        val n = words.size
        val count = IntArray(n) { 1 }
        val previous = IntArray(n) { it }
        var maxCount = 1
        var tail = 0
        for (i in 1 until n) {
            for (j in 0 until i) {
                if (groups[j] != groups[i] && isValid(words[j], words[i])) {
                    if (count[j] + 1 > count[i]) {
                        count[i] = count[j] + 1
                        previous[i] = j
                    }
                }
            }
            if (count[i] > maxCount) {
                maxCount = count[i]
                tail = i
            }
        }
        val result = mutableListOf<String>()
        while (previous[tail] != tail) {
            result.add(0, words[tail])
            tail = previous[tail]
        }
        result.add(0, words[tail])
        return result
    }

    fun minZeroArrayApplyExactlyToSubset(nums: IntArray, queries: Array<IntArray>): Int {
        // queries: [[0,2,1],[0,2,1],[1,1,3]]
        var result = -1
        for (i in nums.indices) {
            val target = nums[i]
            if (target == 0) {
                result = maxOf(result, 0)
                continue
            }
            val canReach = BooleanArray(target + 1)
            canReach[0] = true
            for (index in queries.indices) {
                val (l, r, v) = queries[index]
                if (i !in l..r) continue
                if (target >= v) {
                    canReach[target] = canReach[target - v]
                }
                if (canReach[target]) {
                    result = maxOf(result, index + 1)
                    break
                } else {
                    for (num in target - 1 downTo v) {
                        if (!canReach[num]) {
                            canReach[num] = canReach[num - v]
                        }
                    }
                }
            }
            if (!canReach[target]) {
                return -1
            }
        }
        return result
    }

    fun maxRemovalToMakeZero(nums: IntArray, queries: Array<IntArray>): Int {
        // 3362. zero-array-transformation-iii
        val leftPQ = PriorityQueue<IntArray>(compareBy { it[0] })
        for (query in queries) {
            leftPQ.offer(query)
        }
        val rightPQ = PriorityQueue<IntArray>(compareByDescending { it[1] })
        var skipped = 0
        val diff = IntArray(nums.size + 1)
        var prefix = 0
        for (i in nums.indices) {
            while (leftPQ.isNotEmpty() && leftPQ.peek()!![0] <= i) {
                rightPQ.offer(leftPQ.poll()!!)
            }
            prefix += diff[i]
            var target = nums[i] - prefix
            while (target > 0) {
                if (rightPQ.isEmpty()) return -1
                val (_, r) = rightPQ.poll()!!
                if (r < i) {
                    skipped++
                } else {
                    diff[r + 1]--
                    target--
                    prefix++
                }
            }
        }
        return rightPQ.size + skipped
    }

    fun longestPalindrome(words: Array<String>): Int {
        // 2131.Longest Palindrome by Concatenating Two Letter Words
        val freq = IntArray(26 * 26)
        var doubleCount = 0
        var result = 0
        for (word in words) {
            val code = (word[0] - 'a') * 26 + (word[1] - 'a')
            val reverse = (word[1] - 'a') * 26 + (word[0] - 'a')
            if (freq[reverse] > 0) {
                result += 4
                freq[reverse]--
                if (code == reverse) {
                    doubleCount--
                }
            } else {
                freq[code]++
                if (code == reverse) {
                    doubleCount++
                }
            }
        }
        return if (doubleCount > 0) {
            result + 2
        } else {
            result
        }
    }

    fun maxTargetNodes(edges1: Array<IntArray>, edges2: Array<IntArray>, k: Int): IntArray {
        // 3372 maximize-the-number-of-target-nodes-after-connecting-trees-i
        val n = edges1.size + 1
        val graph1 = Array(n) { mutableListOf<Int>() }
        for ((u, v) in edges1) {
            graph1[u].add(v)
            graph1[v].add(u)
        }
        val m = edges2.size + 1
        val graph2 = Array(m) { mutableListOf<Int>() }
        for ((u, v) in edges2) {
            graph2[u].add(v)
            graph2[v].add(u)
        }

        fun dfs(
            index: Int,
            parent: Int,
            graph: Array<MutableList<Int>>,
            depth: Int,
            target: Int
        ): Int {
            if (depth > target) {
                return 0
            } else if (depth == target) {
                return 1
            } else {
                var count = 1
                for (next in graph[index]) {
                    if (next != parent) {
                        count += dfs(next, index, graph, depth + 1, target)
                    }
                }
                return count
            }
        }

        var maxCount2 = 0
        for (i in 0 until m) {
            maxCount2 = maxOf(maxCount2, dfs(i, -1, graph2, 0, k - 1))
        }
        return IntArray(n) {
            dfs(it, -1, graph1, 0, k) + maxCount2
        }
    }

    fun maxTargetNodes(edges1: Array<IntArray>, edges2: Array<IntArray>): IntArray {
        // 3373 maximize-the-number-of-target-nodes-after-connecting-trees-ii
        val n = edges1.size + 1
        val graph1 = Array(n) { mutableListOf<Int>() }
        for ((u, v) in edges1) {
            graph1[u].add(v)
            graph1[v].add(u)
        }
        val m = edges2.size + 1
        val graph2 = Array(m) { mutableListOf<Int>() }
        for ((u, v) in edges2) {
            graph2[u].add(v)
            graph2[v].add(u)
        }
        val queue = LinkedList<Pair<Int, Int>>()
        queue.offer(0 to -1)
        var depth = 0
        var oddSum = 0
        var evenSum = 0
        while (queue.isNotEmpty()) {
            val levelSize = queue.size
            if (depth % 2 == 0) {
                evenSum += levelSize
            } else {
                oddSum += levelSize
            }
            repeat(levelSize) {
                val (node, previous) = queue.poll()!!
                for (next in graph2[node]) {
                    if (next == previous) continue
                    queue.offer(next to node)
                }
            }
            depth++
        }
        val maxGain2 = maxOf(evenSum, oddSum)

        val result = IntArray(n)
        val depthEven = BooleanArray(n)
        queue.offer(0 to -1)
        depth = 0
        evenSum = 0
        oddSum = 0
        while (queue.isNotEmpty()) {
            val levelSize = queue.size
            val currentDepthEven = depth % 2 == 0
            if (currentDepthEven) {
                evenSum += levelSize
            } else {
                oddSum += levelSize
            }
            repeat(levelSize) {
                val (node, previous) = queue.poll()!!
                depthEven[node] = currentDepthEven
                for (next in graph1[node]) {
                    if (next == previous) continue
                    queue.offer(next to node)
                }
            }
            depth++
        }

        for (i in 0 until n) {
            result[i] = if (depthEven[i]) {
                evenSum + maxGain2
            } else {
                oddSum + maxGain2
            }
        }
        return result
    }

    fun closestMeetingNode(edges: IntArray, node1: Int, node2: Int): Int {
        val n = edges.size
        val dist1 = IntArray(n) { -1 }
        var distance = 0
        var current = node1
        while (current != -1 && dist1[current] == -1) {
            dist1[current] = distance
            current = edges[current]
            distance++
        }

        val dist2 = IntArray(n) { -1 }
        distance = 0
        current = node2
        while (current != -1 && dist2[current] == -1) {
            dist2[current] = distance
            current = edges[current]
            distance++
        }

        var result = -1
        var minDist = n
        for (i in 0 until n) {
            if (dist1[i] == -1 || dist2[i] == -1) continue
            val bigger = maxOf(dist1[i], dist2[i])
            if (bigger < minDist) {
                minDist = bigger
                result = i
            }
        }
        return result
    }

    fun minimizeMaxDiffInPairs(nums: IntArray, p: Int): Int {
        val n = nums.size
        if (n == 1 || p == 0) return 0
        nums.sort()

        fun valid(maxDiff: Int): Boolean {
            var validCount = 0
            var index = 1
            while (index < n) {
                if (nums[index] - nums[index - 1] <= maxDiff) {
                    if (++validCount == p) return true
                    index += 2
                } else {
                    index++
                }
            }
            return false
        }

        var left = 0
        var right = 1_000_000_000
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (valid(mid)) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        return left
    }

    fun maxEvents(events: Array<IntArray>): Int {
        val pq = PriorityQueue(compareBy<IntArray> { it[0] })
        for (event in events) {
            pq.offer(event)
        }
        val pqEnd = PriorityQueue(compareBy<IntArray> { it[1] })
        var day = 0
        var count = 0
        while (pq.isNotEmpty() || pqEnd.isNotEmpty()) {
            while (pq.isNotEmpty() && pq.peek()!![0] <= day) {
                pqEnd.offer(pq.poll()!!)
            }
            if (pqEnd.isNotEmpty()) {
                if (pqEnd.poll()!![1] >= day) {
                    count++
                    day++
                }
            } else {
                day = pq.peek()!![0]
            }
        }
        return count
    }

    fun maxContinuousFreeTime(eventTime: Int, k: Int, startTime: IntArray, endTime: IntArray): Int {
        // arrange k times, keep relative order
        val n = startTime.size
        val spaces = IntArray(n + 1)
        spaces[0] = startTime[0]
        spaces[n] = eventTime - endTime[n - 1]
        for (i in 1 until n) {
            spaces[i] = startTime[i] - endTime[i - 1]
        }

        var left = 0
        var sum = 0
        var maxSum = 0
        for (right in spaces.indices) {
            sum += spaces[right]
            while (right - left > k) {
                sum -= spaces[left++]
            }
            maxSum = maxOf(maxSum, sum)
        }
        return maxSum
    }

    fun maxContinuousFreeTime(eventTime: Int, startTime: IntArray, endTime: IntArray): Int {
        // arrange 1 time, can change relative order
        val n = startTime.size
        val spaces = IntArray(n + 1)
        spaces[0] = startTime[0]
        spaces[n] = eventTime - endTime[n - 1]
        for (i in 1 until n) {
            spaces[i] = startTime[i] - endTime[i - 1]
        }
        val beforeMax = IntArray(n + 1)
        val afterMax = IntArray(n + 1)
        for (i in 1..n) {
            beforeMax[i] = maxOf(beforeMax[i - 1], spaces[i - 1])
            afterMax[n - i] = maxOf(afterMax[n - i + 1], spaces[n - i + 1])
        }
        var result = 0
        for (i in 0 until n) {
            val duration = endTime[i] - startTime[i]
            // find max before space[i] and max after space[i + 1]
            val sum = spaces[i] + spaces[i + 1] +
                    if (beforeMax[i] >= duration || afterMax[i + 1] >= duration) {
                        duration
                    } else {
                        0
                    }
            result = maxOf(result, sum)
        }
        return result
//        val n = startTime.size
//        val spaces = IntArray(n + 1)
//        spaces[0] = startTime[0]
//        spaces[n] = eventTime - endTime[n - 1]
//        for (i in 1 until n) {
//            spaces[i] = startTime[i] - endTime[i - 1]
//        }
//        val pq = PriorityQueue<Int>(compareBy { spaces[it]} )
//        for (i in 0..n) {
//            if (pq.size < 3) {
//                pq.offer(i)
//            } else if (spaces[pq.peek()!!] < spaces[i]) {
//                pq.poll()
//                pq.offer(i)
//            }
//        }
//        val top3 = LinkedList<Int>()
//        while (pq.isNotEmpty()) {
//            top3.addFirst(pq.poll()!!)
//        }
//        var result = 0
//        for (i in 0 until n) {
//            val duration = endTime[i] - startTime[i]
//            var moveTo = -1
//            for (j in top3.indices) {
//                val spaceIndex = top3[j]
//                if (spaceIndex != i && spaceIndex != i + 1 && spaces[spaceIndex] >= duration) {
//                    moveTo = spaceIndex
//                    break
//                }
//            }
//            val sum = spaces[i] + spaces[i + 1] +
//                    if (moveTo == -1) {
//                        0
//                    } else {
//                        duration
//                    }
//            result = maxOf(result, sum)
//        }
//        return result
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

    fun maximumSubsequence(nums: IntArray, k: Int): Int {
        // (sub[0] + sub[1]) % k == (sub[1] + sub[2]) % k == ... == (sub[x - 2] + sub[x - 1]) % k
        val n = nums.size
        var result = 0
        for (target in 0 until k) {
            val dp = IntArray(k)
            var max = 0
            for (i in 1..n) {
                val num = nums[i - 1] % k
                val other = (target + k - num) % k
                dp[num] = dp[other] + 1
                max = maxOf(max, dp[num])
            }
            result = maxOf(result, max)
        }
        return result
    }

    fun maximumSumAfterDeletingAtMostOne(arr: IntArray): Int {
        var hasDeletion = arr[0]
        var noDeletion = arr[0]
        var result = arr[0]
        for (i in 1 until arr.size) {
            hasDeletion = maxOf(noDeletion, hasDeletion + arr[i])
            noDeletion = maxOf(noDeletion + arr[i], arr[i])
            result = maxOf(result, hasDeletion, noDeletion)
        }
        return result
//        val n = arr.size
//        val leftMax = IntArray(n)
//        leftMax[0] = arr[0]
//        var result = arr[0]
//        for (i in 1 until n) {
//            leftMax[i] = maxOf(leftMax[i - 1] + arr[i], arr[i])
//            result = maxOf(result, leftMax[i])
//        }
//        val rightMax = IntArray(n)
//        rightMax[n - 1] = arr[n - 1]
//        for (i in n - 2 downTo 0) {
//            rightMax[i] = maxOf(rightMax[i + 1] + arr[i], arr[i])
//        }
//        for (i in 1 until n - 1) {
//            result = maxOf(result, leftMax[i - 1] + rightMax[i + 1])
//        }
//        return result
    }

    fun subarrayUniqueBitwiseORs(arr: IntArray): Int {
        val seen = mutableSetOf<Int>()
        val lastSeen = IntArray(30) { -1 }
        for (i in arr.indices) {
            var current = arr[i]
            if (current == 0) {
                seen.add(0)
                continue
            }
            var bit = 0
            var minIndex = i
            while (current != 0) {
                if (current and 1 == 1) {
                    minIndex = minOf(minIndex, lastSeen[bit])
                    lastSeen[bit] = i
                }
                current = current shr 1
                bit++
            }
            var orSum = 0
            for (j in i downTo minIndex + 1) {
                orSum = orSum or arr[j]
                seen.add(orSum)
            }
        }
        return seen.size
    }

    fun maxTotalFruit(fruits: IntArray): Int {
        val freq = IntArray(fruits.size)
        var types = 0
        var result = 0
        var left = 0
        for (right in fruits.indices) {
            if (++freq[fruits[right]] == 1) {
                types++
            }
            while (types > 2) {
                if (--freq[fruits[left]] == 0) {
                    types--
                }
                left++
            }
            result = maxOf(result, right - left + 1)
        }
        return result
    }

    fun numOfUnplacedFruits(fruits: IntArray, baskets: IntArray): Int {
        class TreeNode(var value: Int) {
            var left: TreeNode? = null
            var right: TreeNode? = null
        }

        fun buildTree(start: Int, end: Int): TreeNode {
            if (start == end) {
                return TreeNode(baskets[start])
            }
            val node = TreeNode(-1)
            val mid = start + ((end - start) shr 1)
            node.left = buildTree(start, mid)
            node.right = buildTree(mid + 1, end)
            node.value = maxOf(node.left!!.value, node.right!!.value)
            return node
        }

        fun findAndUse(node: TreeNode, target: Int): Int {
            if (target > node.value) return -1
            if (node.left == null && node.right == null) {
                node.value = 0
                return 0
            }
            if (node.left != null && node.left!!.value >= target) {
                val newMax = findAndUse(node.left!!, target)
                node.value = maxOf(newMax, node.right?.value ?: -1)
            } else {
                val newMax = findAndUse(node.right!!, target)
                node.value = maxOf(node.left?.value ?: -1, newMax)
            }
            return node.value
        }

        val root = buildTree(0, baskets.size - 1)
        var remains = 0
        for (fruit in fruits) {
            if (findAndUse(root, fruit) == -1) {
                remains++
            }
        }
        return remains
    }

    fun soupServings(n: Int): Double {
        if (n > 4500) return 1.0
        val units = (n + 24) / 25
        val memo = Array(units + 1) { DoubleArray(units + 1) { -1.0 } }

        fun calcProb(a: Int, b: Int): Double {
            if (a <= 0 && b <= 0) return 0.5
            if (a <= 0) return 1.0
            if (b <= 0) return 0.0

            if (memo[a][b] >= 0.0) return memo[a][b]

            val prob = 0.25 * (calcProb(a - 4, b) +
                    calcProb(a - 3, b - 1) +
                    calcProb(a - 2, b - 2) +
                    calcProb(a - 1, b - 3))

            memo[a][b] = prob
            return prob
        }

        return calcProb(units, units)
    }

    fun perfectPairs(nums: IntArray): Long {
        val sorted = nums.sortedBy { abs(it) }

        fun isValid(i: Int, j: Int): Boolean {
            val res1 = abs(sorted[i] - sorted[j])
            val res2 = abs(sorted[i] + sorted[j])
            val abs1 = abs(sorted[i])
            val abs2 = abs(sorted[j])
            val minRes: Int
            val maxRes: Int
            val minNum: Int
            val maxNum: Int
            if (res1 < res2) {
                minRes = res1
                maxRes = res2
            } else {
                minRes = res2
                maxRes = res1
            }
            if (abs1 < abs2) {
                minNum = abs1
                maxNum = abs2
            } else {
                minNum = abs2
                maxNum = abs1
            }
            return minRes <= minNum && maxRes >= maxNum
        }

        var result = 0L
        var left = 0
        for (right in 1 until sorted.size) {
            while (left < right && !isValid(left, right)) {
                left++
            }
            if (left < right) {
                result += right - left
            }
        }
        return result
    }

    fun minCostWithReverse(n: Int, edges: Array<IntArray>): Int {
        val edgesMap = Array(n) { mutableMapOf<Int, Int>() }
        for ((u, v, w) in edges) {
            edgesMap[u][v] = minOf(edgesMap[u].getOrDefault(v, Int.MAX_VALUE), w)
            edgesMap[v][u] = minOf(edgesMap[v].getOrDefault(u, Int.MAX_VALUE), w shl 1)
        }
        val dist = IntArray(n) { Int.MAX_VALUE }
        val pq = PriorityQueue<Pair<Int, Int>>(compareBy { it.second }) // node, cost
        pq.offer(0 to 0)
        dist[0] = 0
        while (pq.isNotEmpty()) {
            val (node, cost) = pq.poll()!!
            if (node == n - 1) return cost
            for ((next, weight) in edgesMap[node]) {
                val newCost = cost + weight
                if (newCost > dist[next]) continue
                dist[next] = newCost
                pq.offer(next to newCost)
            }
        }
        return -1
    }

    fun minCostToHome(
        startPos: IntArray,
        homePos: IntArray,
        rowCosts: IntArray,
        colCosts: IntArray
    ): Int {
        var cost = 0
        val rowStart = if (startPos[0] < homePos[0]) startPos[0] + 1 else homePos[0]
        val rowEnd = if (startPos[0] < homePos[0]) homePos[0] else startPos[0] - 1
        for (row in rowStart..rowEnd) {
            cost += rowCosts[row]
        }
        val colStart = if (startPos[1] < homePos[1]) startPos[1] + 1 else homePos[1]
        val colEnd = if (startPos[1] < homePos[1]) homePos[1] else startPos[1] - 1
        for (col in colStart..colEnd) {
            cost += colCosts[col]
        }
        return cost
    }

    fun makeTheIntegerZero(num1: Int, num2: Int): Int {
        var result = 0
        var remain = num1.toLong() // num1 - num2*result
        while (remain > num2 + result) {
            result++
            remain -= num2
            if (remain.countOneBits() <= result) {
                return result
            }
        }
        return -1
    }

    fun peopleAwareOfSecret(n: Int, delay: Int, forget: Int): Int {
        val MOD = 1_000_000_007
        val forgetOn = IntArray(n + 1)
        val shareOn = IntArray(n + 1)
        shareOn[delay] = 1
        forgetOn[forget] = 1
        var count = 1
        var active = 0
        for (day in 1 until n) {
            active = (active + MOD - forgetOn[day] + shareOn[day]) % MOD
            count = (count + MOD - forgetOn[day] + active) % MOD
            if (day + delay <= n) {
                shareOn[day + delay] = active
            }
            if (day + forget <= n) {
                forgetOn[day + forget] = active
            }
        }
        return count
    }

    fun minimumTeachings(n: Int, languages: Array<IntArray>, friendships: Array<IntArray>): Int {
        fun langInCommon(user1: Int, user2: Int): Boolean {
            for (lang1 in languages[user1 - 1]) {
                for (lang2 in languages[user2 - 1]) {
                    if (lang1 == lang2) {
                        return true
                    }
                }
            }
            return false
        }

        val toTeach = mutableSetOf<Int>()
        for ((u, v) in friendships) {
            if (!langInCommon(u, v)) {
                toTeach.add(u)
                toTeach.add(v)
            }
        }

        val langKnownBy = IntArray(n + 1)
        var mostCommon = 0
        for (user in toTeach) {
            for (lang in languages[user - 1]) {
                if (++langKnownBy[lang] > langKnownBy[mostCommon]) {
                    mostCommon = lang
                }
            }
        }

        return toTeach.size - langKnownBy[mostCommon]
    }

    class FoodRatings(foods: Array<String>, cuisines: Array<String>, ratings: IntArray) {

        val foodCuisine = mutableMapOf<String, String>()
        val cuisineFood = mutableMapOf<String, TreeSet<String>>()
        val foodRating = mutableMapOf<String, Int>()

        init {
            for (i in foods.indices) {
                foodCuisine[foods[i]] = cuisines[i]
                foodRating[foods[i]] = ratings[i]
                cuisineFood.getOrPut(cuisines[i]) {
                    TreeSet<String>(compareByDescending<String> { foodRating[it]!! }.thenBy { it })
                }.add(foods[i])
            }
        }

        fun changeRating(food: String, newRating: Int) {
            val cuisine = foodCuisine[food]!!
            val foodSet = cuisineFood[cuisine]!!
            foodSet.remove(food)
            foodRating[food] = newRating
            foodSet.add(food)
        }

        fun highestRated(cuisine: String): String {
            return cuisineFood[cuisine]!!.first()!!
        }
    }

    class TaskManager(tasks: List<List<Int>>) {
        val taskToUser = mutableMapOf<Int, Int>()
        val taskToPriority = mutableMapOf<Int, Int>()
        val priorityToTasks = TreeMap<Int, TreeSet<Int>>()

        init {
            for ((userId, taskId, priority) in tasks) {
                add(userId, taskId, priority)
            }
        }

        fun add(userId: Int, taskId: Int, priority: Int) {
            taskToUser[taskId] = userId
            taskToPriority[taskId] = priority
            priorityToTasks.getOrPut(priority) {
                TreeSet<Int>(compareByDescending { it })
            }.add(taskId)
        }

        fun edit(taskId: Int, newPriority: Int) {
            removeTaskFromPriority(taskId)
            taskToPriority[taskId] = newPriority
            priorityToTasks.getOrPut(newPriority) {
                TreeSet<Int>(compareByDescending { it })
            }.add(taskId)
        }

        fun removeTaskFromPriority(taskId: Int) {
            val oldPriority = taskToPriority[taskId]!!
            if (priorityToTasks[oldPriority]!!.size == 1) {
                priorityToTasks.remove(oldPriority)!!
            } else {
                priorityToTasks[oldPriority]!!.remove(taskId)
            }
        }

        fun rmv(taskId: Int) {
            removeTaskFromPriority(taskId)
            taskToPriority.remove(taskId)
            taskToUser.remove(taskId)
        }

        fun execTop(): Int {
            if (priorityToTasks.isEmpty()) return -1
            val taskId = if (priorityToTasks.lastEntry()!!.value.size == 1) {
                priorityToTasks.pollLastEntry()!!.value!!.firstOrNull()
            } else {
                priorityToTasks.lastEntry()!!.value!!.pollFirst()
            }
            if (taskId == null) return -1
            val result = taskToUser[taskId]!!
            taskToPriority.remove(taskId)
            taskToUser.remove(taskId)
            return result
        }
    }

    class Router(val memoryLimit: Int) {

        data class Packet(val source: Int, val destination: Int, val timestamp: Int)

        val queue = LinkedList<Packet>()
        val packetSet = mutableSetOf<Packet>()
        val destMap = mutableMapOf<Int, MutableList<Packet>>()

        fun addPacket(source: Int, destination: Int, timestamp: Int): Boolean {
            val newPacket = Packet(source, destination, timestamp)
            if (newPacket in packetSet) {
                return false
            } else {
                if (queue.size == memoryLimit) {
                    val first = queue.removeFirst()!!
                    destMap[first.destination]!!.removeFirst()
                    packetSet.remove(first)
                }
                queue.addLast(newPacket)
                packetSet.add(newPacket)
                destMap.getOrPut(destination) { mutableListOf<Packet>() }.add(newPacket)
                return true
            }
        }

        fun forwardPacket(): IntArray {
            return if (queue.isNotEmpty()) {
                val first = queue.removeFirst()!!
                destMap[first.destination]!!.removeFirst()
                packetSet.remove(first)
                intArrayOf(first.source, first.destination, first.timestamp)
            } else {
                intArrayOf()
            }
        }

        fun searchLeftMost(target: Int, packets: List<Packet>): Int {
            var left = 0
            var right = packets.size - 1
            while (left <= right) {
                val mid = left + ((right - left) shr 1)
                if (packets[mid].timestamp >= target) {
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            }
            return left
        }

        fun searchRightMost(target: Int, packets: List<Packet>): Int {
            var left = 0
            var right = packets.size - 1
            while (left <= right) {
                val mid = left + ((right - left) shr 1)
                if (packets[mid].timestamp <= target) {
                    left = mid + 1
                } else {
                    right = mid - 1
                }
            }
            return right
        }

        fun getCount(destination: Int, startTime: Int, endTime: Int): Int {
            val destPackets = destMap.getOrDefault(destination, emptyList())
            val left = searchLeftMost(startTime, destPackets)
            val right = searchRightMost(endTime, destPackets)
            return right - left + 1
        }
    }

    fun minimumTotal(triangle: List<List<Int>>): Int {
        val m = triangle.size
        val n = triangle[m - 1].size
        val dp = IntArray(n) { Int.MAX_VALUE }
        dp[0] = triangle[0][0]
        if (m == 1) return dp[0]
        var result = Int.MAX_VALUE
        for (row in 1 until m) {
            for (col in row downTo 0) {
                if (col == 0) {
                    dp[col] = dp[col] + triangle[row][col]
                } else if (col == row) {
                    dp[col] = dp[col - 1] + triangle[row][col]
                } else {
                    dp[col] = minOf(dp[col - 1], dp[col]) + triangle[row][col]
                }
                if (row == m - 1) {
                    result = minOf(result, dp[col])
                }
            }
        }
        return result
    }

    fun triangleNumber(nums: IntArray): Int {
        nums.sort()
        val n = nums.size
        var count = 0
        for (k in n - 1 downTo 2) {
            var left = 0
            var right = k - 1
            while (left < right) {
                if (nums[left] + nums[right] > nums[k]) {
                    count += right - left
                    right--
                } else {
                    left++
                }
            }
        }
        return count
    }
}