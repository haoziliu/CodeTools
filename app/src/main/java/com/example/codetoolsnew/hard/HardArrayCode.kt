package com.example.codetools.hard

import com.example.codetools.ArrayCode
import java.util.Arrays
import java.util.LinkedList
import java.util.PriorityQueue
import java.util.Stack
import java.util.TreeMap
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min


object HardArrayCode {

    fun firstMissingPositive(nums: IntArray): Int {
        // [1,2,-1] -> 3
        // [0,1,2] -> 3
        // [2,3,1] -> 4
        // [4,-4,-2,1] -> 2
        // [7,8,9,11,12] -> 1

        // num bigger than size won't be valid
        // round 1: mark negative and 0 as invalid big number
        // round 2: mark nums[abs(num)-1] as negative
        // round 3: check first non-negative
        val size = nums.size
        for (i in nums.indices) {
            if (nums[i] <= 0) {
                nums[i] = size + 1
            }
        }
        var nextIndex: Int
        for (num in nums) {
            nextIndex = abs(num) - 1
            if (nextIndex < size && nums[nextIndex] > 0) {
                nums[nextIndex] *= -1
            }
        }
        for (i in nums.indices) {
            if (nums[i] > 0) {
                return i + 1
            }
        }
        return size + 1
    }

    fun findMedianSortedArrays(nums1: IntArray, nums2: IntArray): Double {
//        var i = 0
//        var j = 0
//        val length = nums1.size + nums2.size
//        val mid = length / 2
//        var last = Int.MIN_VALUE
//        while (i + j < mid) {
//            last = if (i == nums1.size) {
//                nums2[j++];
//            } else if (j == nums2.size || nums1[i] < nums2[j]) {
//                nums1[i++];
//            } else {
//                nums2[j++];
//            }
//        }
//        val current = if (i == nums1.size) {
//            nums2[j];
//        } else if (j == nums2.size || nums1[i] < nums2[j]) {
//            nums1[i];
//        } else {
//            nums2[j];
//        }
//
//        return if (length % 2 == 0) {
//            (last + current) / 2.0
//        } else {
//            current.toDouble()
//        }

        // binary search to find the median
        val m = nums1.size
        val n = nums2.size
        if (m > n) {
            // make sure nums1 is shorter, because we need to do binary search on the shorter one
            return findMedianSortedArrays(nums2, nums1)
        }
        var iMin = 0
        var iMax = m
        val halfLen = (m + n + 1) / 2
        while (iMin <= iMax) {
            // i is the partition of nums1, while j is the partition of nums2, they both contribute to halfLen
            val i = iMin + (iMax - iMin) / 2
            val j = halfLen - i
            if (i < iMax && nums2[j - 1] > nums1[i]) {
                // i is too small, need to increase it
                iMin = i + 1
            } else if (i > iMin && nums1[i - 1] > nums2[j]) {
                // i is too big, need to decrease it
                iMax = i - 1
            } else {
                val maxLeft = when {
                    i == 0 -> nums2[j - 1]
                    j == 0 -> nums1[i - 1]
                    else -> max(nums1[i - 1], nums2[j - 1])
                }
                if ((m + n) % 2 == 1) {
                    return maxLeft.toDouble()
                }
                val minRight = when {
                    i == m -> nums2[j]
                    j == n -> nums1[i]
                    else -> min(nums1[i], nums2[j])
                }
                return (maxLeft + minRight) / 2.0
            }
        }
        return 0.0
    }

    fun subarraysWithKDistinct(nums: IntArray, k: Int): Int {
        // good: distinct integers number is exactly k.
        return ArrayCode.subarraysWithAtMostKDistinct(
            nums,
            k
        ) - ArrayCode.subarraysWithAtMostKDistinct(nums, k - 1)
    }

    fun subarraysWithFixedBound(nums: IntArray, minK: Int, maxK: Int): Long {
        var result = 0L
        var iMax = -1
        var iMin = -1
        var start = 0
        for (end in nums.indices) {
            if (nums[end] < minK || nums[end] > maxK) {
                start = end + 1
                continue
            }
            if (nums[end] == maxK) {
                iMax = end
            }
            if (nums[end] == minK) {
                iMin = end
            }
            // start is the first element that is bigger than minK and smaller than maxK
            // we find the smaller index between iMax and iMin, which is the end point
            // then we calculate the number of subarrays that can be formed by the start and end
            // using max() is to make sure start is smaller than end
            // as long as start/iMax/iMin is not updated, each loop will add the same number of subarrays
            result += max((min(iMax, iMin) - start + 1), 0)
        }
        return result
    }

    fun trapWater(height: IntArray): Int {
        // two pointers, keep left and right max,
        // since water is trapped by the lower side, we need to move the lower side
//        var i = 0
//        var leftMax = height[0]
//        var sum = 0
//        var j: Int = height.size - 1
//        var rightMax = height[j]
//        while (i < j) {
//            if (leftMax <= rightMax) {
//                sum += leftMax - height[i]
//                i++
//                leftMax = max(leftMax, height[i])
//            } else {
//                sum += rightMax - height[j]
//                j--
//                rightMax = max(rightMax, height[j])
//            }
//        }
//        return sum

        // two pointers: dynamically fill up water, check neighbour if it can trap water
//        var left = 0
//        var right = height.size - 1
//        var sum = 0
//        while (left + 1 < right) {
//            if (height[left] <= height[right]) {
//                if (height[left + 1] < height[left]) {
//                    sum += height[left] - height[left + 1]
//                }
//                height[left  + 1] = maxOf(height[left + 1], height[left])
//                left++
//            } else {
//                if (height[right - 1] < height[right]) {
//                    sum += height[right] - height[right - 1]
//                }
//                height[right - 1] = maxOf(height[right - 1], height[right])
//                right--
//            }
//        }
//        return sum

        // monotonic stack
        val stack = LinkedList<Int>()
        var sum = 0
        for (i in height.indices) {
            while (stack.isNotEmpty() && height[i] > height[stack.peek()!!]) {
                val top = stack.pop()
                if (stack.isEmpty()) break
                val distance = i - stack.peek()!! - 1
                val boundedHeight = min(height[i], height[stack.peek()!!]) - height[top]
                sum += distance * boundedHeight
            }
            stack.push(i)
        }
        return sum
    }

    fun trapRainWater(heightMap: Array<IntArray>): Int {
        // dynamically fill up water, check neighbour if it can trap water
        val DIRECTIONS = arrayOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)

        val m = heightMap.size
        val n = heightMap[0].size
        var sum = 0
        val visited = Array(m) { BooleanArray(n) }
        val pq = PriorityQueue<Pair<Int, Int>>(compareBy { heightMap[it.first][it.second] })
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    pq.offer(i to j)
                    visited[i][j] = true
                }
            }
        }
        while (pq.isNotEmpty()) {
            val (i, j) = pq.poll()!!
            for ((dx, dy) in DIRECTIONS) {
                val newX = i + dx
                val newY = j + dy
                if (newX !in 0 until m || newY !in 0 until n || visited[newX][newY]) continue
                if (heightMap[newX][newY] < heightMap[i][j]) {
                    sum += heightMap[i][j] - heightMap[newX][newY]
                }
                heightMap[newX][newY] = maxOf(heightMap[i][j], heightMap[newX][newY]) // 更新水位
                visited[newX][newY] = true
                pq.offer(newX to newY)
            }
        }
        return sum
    }

    private fun largestRectangleArea(heights: IntArray): Int {
        var maxArea = 0
        val stack = Stack<Int>()
        for (i in 0..heights.size) {
            while (stack.isNotEmpty() && (i == heights.size || heights[stack.peek()] > heights[i])) {
                val height = heights[stack.pop()]
                val width = if (stack.isEmpty()) i else i - stack.peek() - 1
                maxArea = maxOf(maxArea, height * width)
            }
            stack.add(i)
        }
        return maxArea
    }

    fun maximalRectangle(matrix: Array<CharArray>): Int {
        if (matrix.isEmpty()) return 0
        var maxArea = 0
        val hist = IntArray(matrix[0].size)
        for (row in matrix) {
            for (i in row.indices) {
                hist[i] = if (row[i] == '0') 0 else hist[i] + 1
            }
            maxArea = maxOf(maxArea, largestRectangleArea(hist))
        }
        return maxArea
    }

    fun minFallingPathSum(grid: Array<IntArray>): Int {
        val size = grid.size
        val minSumArray = IntArray(size)
        grid[0].forEachIndexed { index, item ->
            minSumArray[index] = item
        }

        fun findMinTwo(minTwoArray: IntArray, intArray: IntArray) {
            minTwoArray[0] = Int.MAX_VALUE
            minTwoArray[1] = Int.MAX_VALUE
            for (num in intArray) {
                if (num < minTwoArray[0]) {
                    minTwoArray[1] = minTwoArray[0]
                    minTwoArray[0] = num
                } else if (num < minTwoArray[1]) {
                    minTwoArray[1] = num
                }
            }
        }

        val lastMinTwo = IntArray(2)
        var proceed = false
        for (rowIndex in 1 until size) {
            for (i in 0 until size) {
                if (!proceed) {
                    findMinTwo(lastMinTwo, minSumArray)
                    proceed = true
                }
                if (minSumArray[i] != lastMinTwo[0]) {
                    minSumArray[i] = lastMinTwo[0] + grid[rowIndex][i]
                } else {
                    minSumArray[i] = lastMinTwo[1] + grid[rowIndex][i]
                }
            }
            proceed = false
        }
        return minSumArray.minOrNull() ?: Int.MAX_VALUE
    }

    fun sumOfDistancesInTree(n: Int, edges: Array<IntArray>): IntArray {
        val edgeMap = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { edge ->
            edgeMap.getOrPut(edge[0]) {
                mutableListOf()
            }.add(edge[1])
            edgeMap.getOrPut(edge[1]) {
                mutableListOf()
            }.add(edge[0])
        }

        val nodeCount = IntArray(n) { 1 }
        val result = IntArray(n)
        fun dfs0(current: Int, parent: Int) {
            edgeMap[current]?.forEach { child ->
                if (child != parent) {
                    dfs0(child, current)
                    nodeCount[current] += nodeCount[child]
                    result[current] += result[child] + nodeCount[child]
                }
            }
        }

        fun reRoot(current: Int, parent: Int) {
            edgeMap[current]?.forEach { child ->
                if (child != parent) {
                    // all sub nodes will -1, but others +1
                    result[child] = result[current] - nodeCount[child] + (n - nodeCount[child])
                    reRoot(child, current)
                }
            }
        }

        dfs0(0, -1)
        reRoot(0, -1)
        return result
    }

    fun minCostToHireWorkers(quality: IntArray, wage: IntArray, k: Int): Double {
        var qSum = 0
        val pq = PriorityQueue<Int>(reverseOrder())
        var currentQuality: Int
        // higher quality/cost ratio should be treated first
        return wage.indices.sortedByDescending { 1.0 * quality[it] / wage[it] }.minOf {
            currentQuality = quality[it]
            qSum += currentQuality
            pq += currentQuality
            if (pq.size > k) {
                qSum -= pq.poll() // remove the highest quality worker
            }
            if (pq.size >= k) {
                1.0 * qSum * wage[it] / currentQuality
            } else Double.MAX_VALUE
        }
    }

    fun maximumXORValueSum(nums: IntArray, k: Int, edges: Array<IntArray>): Long {
        var isFlippedOdd = false
        var flippedMinDiff = Int.MAX_VALUE // flipped > original, minimum delta
        var indexOfFlippedMinDiff = -1
        var unFlippedMinDiff = Int.MIN_VALUE // flipped < original, minimum abs delta
        var indexOfUnFlippedMinDiff = -1

        for (i in nums.indices) {
            val flipped = nums[i] xor k
            val newDiff = flipped - nums[i]
            if (newDiff > 0) {
                isFlippedOdd = !isFlippedOdd
                nums[i] = flipped
                if (newDiff < flippedMinDiff) {
                    // find the minimum delta
                    flippedMinDiff = newDiff
                    indexOfFlippedMinDiff = i
                }
            } else {
                if (newDiff > unFlippedMinDiff) {
                    // find the minimum delta
                    unFlippedMinDiff = newDiff
                    indexOfUnFlippedMinDiff = i
                }
            }
        }
        val result = nums.sumOf { it.toLong() }
        if (isFlippedOdd) {
            // flipped odd times, at least once
            return if (indexOfUnFlippedMinDiff == -1) {
                // all flipped, we just flip back the minimum one
                result - nums[indexOfFlippedMinDiff] + (nums[indexOfFlippedMinDiff] xor k)
            } else {
                // test to flip back the minimum flipped one, or flip the minimum un-flipped one
                maxOf(
                    result - nums[indexOfFlippedMinDiff] + (nums[indexOfFlippedMinDiff] xor k),
                    result - nums[indexOfUnFlippedMinDiff] + (nums[indexOfUnFlippedMinDiff] xor k)
                )
            }
        } else {
            // flipped even times or 0 times, we can return the result directly
            return result
        }
    }

    fun findMaximizedCapital(k: Int, w: Int, profits: IntArray, capital: IntArray): Int {
        val minCapitalQueue = PriorityQueue<Int> { i, j -> capital[i].compareTo(capital[j]) }
        val maxProfitQueue = PriorityQueue<Int> { i, j -> profits[j].compareTo(profits[i]) }

        for (i in profits.indices) {
            minCapitalQueue.offer(i)
        }

        var currentBalance = w
        for (i in 0 until k) {
            while (minCapitalQueue.isNotEmpty() && capital[minCapitalQueue.peek()] <= currentBalance) {
                maxProfitQueue.offer(minCapitalQueue.poll())
            }

            if (maxProfitQueue.isEmpty()) {
                break // 没有更多可以投资的项目
            }

            currentBalance += profits[maxProfitQueue.poll()]
        }

        return currentBalance
    }

    fun minPatches(nums: IntArray, n: Int): Int {
        var reuslt = 0
        var i = 0
        var maxReach = 0L

        while (maxReach < n) {
            if (i < nums.size && nums[i] <= maxReach + 1) {
                maxReach += nums[i].toLong()
                i++
            } else {
                maxReach += maxReach + 1
                reuslt++
            }
        }

        return reuslt
    }

    fun minKBitFlips(nums: IntArray, k: Int): Int {
//        var left = 0
//        var result = 0
//        while (left < nums.size) {
//            if (nums[left] == 0) {
//                if (left + k - 1 >= nums.size) return -1
//                for (i in left until left + k) {
//                    nums[i] = nums[i] xor 1
//                }
//                result++
//            }
//            left++
//        }
//        return result
        var result = 0
        var currentFlips = 0
        for (i in nums.indices) {
            if (i >= k && nums[i - k] == 2) {
                // remove the contribution of window start
                currentFlips--
            }
            if (nums[i] == currentFlips % 2) {
                if (i + k > nums.size) {
                    return -1
                }
                nums[i] = 2
                currentFlips++
                result++
            }
        }
        return result
    }

    fun maxNumEdgesToRemove(n: Int, edges: Array<IntArray>): Int {
        class UnionFind(n: Int) {
            private val parent = IntArray(n) { it }
            private val rank = IntArray(n) { 0 }

            fun find(x: Int): Int {
                if (parent[x] != x) {
                    parent[x] = find(parent[x])
                }
                return parent[x]
            }

            fun union(x: Int, y: Int): Boolean {
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
                    return true
                }
                return false
            }
        }

        val type1Edges = mutableListOf<IntArray>()
        val type2Edges = mutableListOf<IntArray>()
        val type3Edges = mutableListOf<IntArray>()

        for (edge in edges) {
            when (edge[0]) {
                1 -> type1Edges.add(edge)
                2 -> type2Edges.add(edge)
                3 -> type3Edges.add(edge)
            }
        }

        val ufA = UnionFind(n)
        val ufB = UnionFind(n)
        var usedEdges = 0

        for (edge in type3Edges) {
            val unionA = ufA.union(edge[1] - 1, edge[2] - 1)
            val unionB = ufB.union(edge[1] - 1, edge[2] - 1)
            if (unionA || unionB) {
                usedEdges++
            }
        }
        for (edge in type1Edges) {
            if (ufA.union(edge[1] - 1, edge[2] - 1)) {
                usedEdges++
            }
        }
        for (edge in type2Edges) {
            if (ufB.union(edge[1] - 1, edge[2] - 1)) {
                usedEdges++
            }
        }
        val connectedA = (0 until n).all { ufA.find(it) == ufA.find(0) }
        val connectedB = (0 until n).all { ufB.find(it) == ufB.find(0) }

        return if (connectedA && connectedB) {
            edges.size - usedEdges
        } else {
            -1
        }
    }


    fun candy(ratings: IntArray): Int {
        // at least 1; higher rating get more than neighbors
        val candies = IntArray(ratings.size) { 1 }
        for (i in 1..ratings.lastIndex) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1
            }
        }
        for (i in ratings.lastIndex - 1 downTo 0) {
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = maxOf(candies[i], candies[i + 1] + 1)
            }
        }
        return candies.sum()
    }

    fun survivedRobotsHealths(
        positions: IntArray,
        healths: IntArray,
        directions: String
    ): List<Int> {
        data class Robot(
            val position: Int,
            var health: Int,
            val direction: Char,
            val originalIndex: Int
        )

        val pq = PriorityQueue<Robot> { r1, r2 -> r1.position.compareTo(r2.position) }
        for (i in positions.indices) {
            pq.offer(Robot(positions[i], healths[i], directions[i], i))
        }
        val stack = LinkedList<Robot>()
        var current = pq.poll()
        while (current != null) {
            if (current.direction == 'L' && stack.peek()?.direction == 'R') {
                val top = stack.peek()!!
                when {
                    top.health > current.health -> {
                        top.health--
                        current = pq.poll()
                    }

                    current.health > top.health -> {
                        stack.pop()
                        current.health--
                    }

                    else -> {
                        stack.pop()
                        current = pq.poll()
                    }
                }
            } else {
                stack.push(current)
                current = pq.poll()
            }
        }
        return stack.sortedBy { it.originalIndex }.map { it.health }
    }

    fun buildMatrix(
        k: Int,
        rowConditions: Array<IntArray>,
        colConditions: Array<IntArray>
    ): Array<IntArray> {
        val belowRelations = Array(k + 1) { IntArray(k + 1) }
        for ((above, below) in rowConditions) {
            belowRelations[above][below] = 1
        }
        val rightRelations = Array(k + 1) { IntArray(k + 1) }
        for ((left, right) in colConditions) {
            rightRelations[left][right] = 1
        }

        fun topologicalSort(relations: Array<IntArray>): List<Int> {
            val inDegree = IntArray(k + 1)
            val adjList = Array(k + 1) { mutableListOf<Int>() }
            for (i in 1..k) {
                for (j in 1..k) {
                    if (relations[i][j] == 1) {
                        adjList[i].add(j)
                        inDegree[j]++
                    }
                }
            }
            val queue = LinkedList<Int>()
            for (i in 1..k) {
                if (inDegree[i] == 0) {
                    queue.add(i)
                }
            }
            val order = mutableListOf<Int>()
            while (queue.isNotEmpty()) {
                val node = queue.poll()
                order.add(node)
                for (neighbor in adjList[node]) {
                    inDegree[neighbor]--
                    if (inDegree[neighbor] == 0) {
                        queue.add(neighbor)
                    }
                }
            }
            return if (order.size == k) order else listOf()
        }

        val rowOrder = topologicalSort(belowRelations)
        val colOrder = topologicalSort(rightRelations)

        if (rowOrder.isEmpty() || colOrder.isEmpty()) {
            return arrayOf()
        }

        val rowPosition = IntArray(k + 1)
        val colPosition = IntArray(k + 1)
        for (i in 0 until k) {
            rowPosition[rowOrder[i]] = i
            colPosition[colOrder[i]] = i
        }

        val matrix = Array(k) { IntArray(k) { 0 } }
        for (num in 1..k) {
            val i = rowPosition[num]
            val j = colPosition[num]
            matrix[i][j] = num
        }

        return matrix
    }

    fun secondMinimum(n: Int, edges: Array<IntArray>, time: Int, change: Int): Int {
        val graph = Array(n) { mutableListOf<Int>() }
        for ((from, to) in edges) {
            graph[from - 1].add(to - 1)
            graph[to - 1].add(from - 1)
        }

        val firstRank = IntArray(n) { Int.MAX_VALUE }
        val secondRank = IntArray(n) { Int.MAX_VALUE }
        firstRank[0] = 0
        val queue = LinkedList<Pair<Int, Int>>()
        queue.offer(Pair(0, 0))
        while (queue.isNotEmpty()) {
            val (u, currentTime) = queue.poll()!!
            for (v in graph[u]) {
                val newElapsed = if (currentTime / change % 2 != 0) {
                    (currentTime / change + 1) * change + time
                } else {
                    currentTime + time
                }
                if (newElapsed < firstRank[v]) {
                    secondRank[v] = firstRank[v]
                    firstRank[v] = newElapsed
                    queue.offer(Pair(v, newElapsed))
                } else if (newElapsed > firstRank[v] && newElapsed < secondRank[v]) {
                    secondRank[v] = newElapsed
                    queue.offer(Pair(v, newElapsed))
                }
            }
        }
        return secondRank.last()

//        val elapseStack = Array(n) { PriorityQueue<Int>(reverseOrder()) }
//        elapseStack[0].offer(0)
//        val queue = LinkedList<Pair<Int, Int>>()
//        queue.offer(Pair(0, 0))
//
//        while (queue.isNotEmpty()) {
//            val (u, currentTime) = queue.poll()!!
//
//            for (v in graph[u]) {
//                val newElapsed = if (currentTime / change % 2 != 0) {
//                    (currentTime / change + 1) * change + time
//                } else {
//                    currentTime + time
//                }
//                if (!elapseStack[v].contains(newElapsed)) {
//                    if (elapseStack[v].size < 2) {
//                        elapseStack[v].offer(newElapsed)
//                        queue.offer(Pair(v, newElapsed))
//                    } else if (elapseStack[v].peek() > newElapsed) {
//                        elapseStack[v].poll()
//                        elapseStack[v].offer(newElapsed)
//                        queue.offer(Pair(v, newElapsed))
//                    }
//                }
//            }
//        }
//
//        return elapseStack.last().peek()!!
    }

    fun findSubstring(s: String, words: Array<String>): List<Int> {
        val wordLength = words[0].length
        if (s.length < words.size * wordLength) return listOf()

        val targetFreq = mutableMapOf<String, Int>()
        words.forEach { word ->
            targetFreq[word] = targetFreq.getOrDefault(word, 0) + 1
        }
        val targetSize = words.size

        val currentFreq = mutableMapOf<String, Int>()
        var count = 0
        val result = mutableListOf<Int>()
        for (i in 0 until wordLength) {
            currentFreq.clear()
            count = 0
            var start = i
            for (end in start + wordLength..s.length step wordLength) {
                val current = s.substring(end - wordLength, end)
                if (!targetFreq.containsKey(current)) {
                    start = end
                    currentFreq.clear()
                    count = 0
                    continue
                }
                currentFreq[current] = currentFreq.getOrDefault(current, 0) + 1
                count++
                while (start < end && currentFreq[current]!! > targetFreq[current]!!) {
                    val startWord = s.substring(start, start + wordLength)
                    currentFreq[startWord] = currentFreq[startWord]!! - 1
                    count--
                    start += wordLength
                }
                if (count == targetSize) {
                    result.add(start)
                }
            }
        }

        return result
    }

    class UnionFind(n: Int) {
        private val parent = IntArray(n) { it }
        private val rank = IntArray(n) { 1 }

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
                } else if (rank[rootY] > rank[rootX]) {
                    parent[rootX] = rootY
                } else {
                    parent[rootY] = rootX
                    rank[rootX] += 1
                }
            }
        }
    }

    fun longestConsecutive(nums: IntArray): Int {
//    nums.sort() // Step 1: Sort the array
//
//    var longestStreak = 1
//    var currentStreak = 1
//
//    for (i in 1 until nums.size) {
//        if (nums[i] != nums[i - 1]) {
//            if (nums[i] == nums[i - 1] + 1) {
//                currentStreak += 1
//            } else {
//                longestStreak = maxOf(longestStreak, currentStreak)
//                currentStreak = 1
//            }
//        }
//    }
//
//    return maxOf(longestStreak, currentStreak)

        if (nums.isEmpty()) return 0

        val uf = UnionFind(nums.size)
        val numToIndex = mutableMapOf<Int, Int>()

        for (i in nums.indices) {
            if (numToIndex.containsKey(nums[i])) continue
            numToIndex[nums[i]] = i

            if (numToIndex.containsKey(nums[i] - 1)) {
                uf.union(i, numToIndex[nums[i] - 1]!!)
            }
            if (numToIndex.containsKey(nums[i] + 1)) {
                uf.union(i, numToIndex[nums[i] + 1]!!)
            }
        }

        val count = IntArray(nums.size)
        var maxCount = 0

        for (i in nums.indices) {
            val rootX = uf.find(i)
            count[rootX]++
            maxCount = maxOf(maxCount, count[rootX])
        }

        return maxCount
    }

    fun kthSmallestDistancePair(nums: IntArray, k: Int): Int {
        fun countPairWithDistanceLessThanOrEqual(mid: Int): Int {
            var count = 0
            var start = 0
            for (end in nums.indices) {
                while (start < end && nums[end] - nums[start] > mid) {
                    start++
                }
                count += end - start
            }
            return count
        }

        nums.sort()
        var left = 0
        var right = nums.last() - nums[0]
        while (left < right) {
            val mid = left + (right - left) / 2
            val countPair = countPairWithDistanceLessThanOrEqual(mid)

            if (countPair >= k) {
                right = mid
            } else {
                left = mid + 1
            }
        }
        return left
    }

    fun stoneGameII(piles: IntArray): Int {
        val n = piles.size
        // dp[i][m] 表示从第 i 堆开始，并且当前 m 值下，玩家能够拿到的最多石子数
        val dp = Array(n + 1) { IntArray(n + 1) }
        // suffix_sum 用来存储从第 i 堆开始到最后一堆石子的总和
        val suffixSum = IntArray(n + 1)
        for (i in n - 1 downTo 0) {
            suffixSum[i] = suffixSum[i + 1] + piles[i]
        }

        // 从后往前初始化 dp 表
        for (i in n - 1 downTo 0) {
            for (m in 1..n) {
                // 如果剩余的堆数小于等于 2 * m，玩家可以直接拿走所有石子
                if (i + 2 * m >= n) {
                    dp[i][m] = suffixSum[i]
                } else {
                    // 否则进行状态转移，当前玩家尽量拿最多的石子
                    var maxStones = 0
                    for (x in 1..2 * m) {
                        maxStones = maxOf(maxStones, suffixSum[i] - dp[i + x][maxOf(m, x)])
                    }
                    dp[i][m] = maxStones
                }
            }
        }
        // 初始从堆 0 开始，m = 1
        return dp[0][1]
    }

    fun removeBoxes(boxes: IntArray): Int {
        val n = boxes.size
        val dp = Array(n) { Array(n) { IntArray(n) } }

        fun removeBoxesSub(i: Int, j: Int, k: Int): Int {
            if (i > j) return 0
            if (dp[i][j][k] > 0) return dp[i][j][k]

            var i0 = i
            var k0 = k

            // 将相同颜色连续的盒子分组处理
            while (i0 + 1 <= j && boxes[i0 + 1] == boxes[i0]) {
                i0++
                k0++
            }

            // 计算移除当前分组的得分
            var res = (k0 + 1) * (k0 + 1) + removeBoxesSub(i0 + 1, j, 0)

            // 尝试将 boxes[i] 连接到后面相同颜色的盒子
            for (m in i0 + 1..j) {
                if (boxes[i0] == boxes[m]) {
                    res =
                        maxOf(res, removeBoxesSub(i0 + 1, m - 1, 0) + removeBoxesSub(m, j, k0 + 1))
                }
            }

            dp[i][j][k] = res // 更新 dp 数组时，使用初始的 i 和 k 值
            return res
        }

        return removeBoxesSub(0, n - 1, 0)
    }

//    fun modifiedGraphEdges(
//        n: Int,
//        edges: Array<IntArray>,
//        source: Int,
//        destination: Int,
//        target: Int
//    ): Array<IntArray> {
//        val graph = Array(n) { mutableListOf<Pair<Int, Int>>() }
//        for ((i, edge) in edges.withIndex()) {
//            graph[edge[0]].add(Pair(edge[1], i))
//            graph[edge[1]].add(Pair(edge[0], i))
//        }
//
//        // Dijkstra函数
//        fun dijkstra(start: Int): IntArray {
//            val dist = IntArray(n) { Int.MAX_VALUE }
//            dist[start] = 0
//            val pq = PriorityQueue<Pair<Int, Int>>(compareBy { it.second })
//            pq.offer(Pair(start, 0))
//
//            while (pq.isNotEmpty()) {
//                val (node, d) = pq.poll()!!
//                if (d > dist[node]) continue
//
//                for ((neighbour, index) in graph[node]) {
//                    val weight = if (edges[index][2] == -1) 1 else edges[index][2]
//                    if (dist[neighbour] > dist[node] + weight) {
//                        dist[neighbour] = dist[node] + weight
//                        pq.offer(Pair(neighbour, dist[neighbour]))
//                    }
//                }
//            }
//
//            return dist
//        }
//
//        // 分别从source和destination进行Dijkstra
//        val distFromSource = dijkstra(source)
//        val distFromDestination = dijkstra(destination)
//
//        // 检查初始条件
//        if (distFromSource[destination] < target) return emptyArray()
//
//        // 尝试修改边的权重
//        for (edge in edges) {
//            val u = edge[0]
//            val v = edge[1]
//            val potentialWeightUtoV = target - (distFromSource[u] + distFromDestination[v])
//            val potentialWeightVtoU = target - (distFromSource[v] + distFromDestination[u])
//
//            if (edge[2] == -1 && (potentialWeightUtoV > 0 || potentialWeightVtoU > 0)) {
//
//                if (potentialWeightUtoV < 0) {
//                    edge[2] = potentialWeightVtoU
//                    distFromSource[u] =  distFromSource[v] + edge[2]
//                    distFromDestination[v] = distFromDestination[u] + edge[2]
//                } else if (potentialWeightVtoU < 0) {
//                    edge[2] = potentialWeightUtoV
//                    distFromSource[v] =  distFromSource[u] + edge[2]
//                    distFromDestination[u] =  distFromDestination[v] + edge[2]
//                } else {
//                    edge[2] = maxOf(1, minOf(potentialWeightUtoV, potentialWeightVtoU))
//                    if (potentialWeightUtoV < potentialWeightVtoU) {
//                        distFromSource[v] =  distFromSource[u] + edge[2]
//                        distFromDestination[u] =  distFromDestination[v] + edge[2]
//                    } else {
//                        distFromSource[u] =  distFromSource[v] + edge[2]
//                        distFromDestination[v] = distFromDestination[u] + edge[2]
//                    }
//                }
//
//                if (distFromSource[destination] == target) break
//                if (distFromSource[destination] > target) return emptyArray()
//            }
//        }
//
//        for (edge in edges) {
//            if (edge[2] < 0) edge[2] = 1
//        }
//        return edges
//    }

    fun modifiedGraphEdges(
        n: Int,
        edges: Array<IntArray>,
        source: Int,
        destination: Int,
        target: Int
    ): Array<IntArray> {
        val graph = mutableMapOf<Int, MutableList<Pair<Int, Int>>>() // only record id
        for ((i, edge) in edges.withIndex()) {
            graph.getOrPut(edge[0]) { mutableListOf() }.add(Pair(edge[1], i))
            graph.getOrPut(edge[1]) { mutableListOf() }.add(Pair(edge[0], i))
        }

        fun dijkstra(modify: Boolean): Pair<Int, Int> {
            val dp = IntArray(n) { Int.MAX_VALUE }
            dp[source] = 0
            val pq = PriorityQueue<Pair<Int, Int>>(compareBy { it.second })
            pq.offer(Pair(source, 0))
            val modId = dp.clone()

            while (pq.isNotEmpty()) {
                val (node, curr) = pq.poll()!!
                for ((neighbour, j) in graph[node] ?: listOf()) {
                    if ((modify || edges[j][2] != -1) && dp[neighbour] > curr + maxOf(
                            1,
                            edges[j][2]
                        )
                    ) {
                        dp[neighbour] = curr + maxOf(1, edges[j][2])
                        modId[neighbour] = if (edges[j][2] == -1) j else modId[node]
                        pq.offer(Pair(neighbour, dp[neighbour]))
                    }
                }
            }
            return Pair(dp[destination], modId[destination])
        }
        val (defaultDistance, _) = dijkstra(false);
        if (defaultDistance < target) return arrayOf()
        while (true) {
            val (dist, modId) = dijkstra(true)
            if (dist > target) return arrayOf()
            if (dist == target) break
            edges[modId][2] = 1 + target - dist
        }
        for (edge in edges) {
            if (edge[2] < 0) edge[2] = 1
        }
        return edges
    }

    fun findWords(board: Array<CharArray>, words: Array<String>): List<String> {
        class TrieNode() {
            val children = mutableMapOf<Char, TrieNode>()
            var word: String? = null
        }

        fun buildTrie(): TrieNode {
            val root = TrieNode()
            for (word in words) {
                var node = root
                for (char in word) {
                    node = node.children.getOrPut(char) { TrieNode() }
                }
                node.word = word
            }
            return root
        }

        val result = mutableListOf<String>()

        fun searchWord(i: Int, j: Int, node: TrieNode) {
            if (i < 0 || i >= board.size || j < 0 || j >= board[0].size) {
                return
            }
            val char = board[i][j]
            if (char !in node.children) {
                return
            }
            val child = node.children[char]!!
            if (child.word != null) {
                result.add(child.word!!)
                child.word = null
            }

            board[i][j] = '#'
            searchWord(i + 1, j, child)
            searchWord(i - 1, j, child)
            searchWord(i, j + 1, child)
            searchWord(i, j - 1, child)
            board[i][j] = char
        }

        val root = buildTrie()

        for (i in board.indices) {
            for (j in 0 until board[0].size) {
                searchWord(i, j, root)
            }
        }

        return result
    }

    fun sumPrefixScores(words: Array<String>): IntArray {
        class TrieNode {
            val children = Array<TrieNode?>(26) { null }
            var count = 0
        }

        fun buildTrie(): TrieNode {
            val root = TrieNode()
            for (word in words) {
                var node = root
                for (char in word) {
                    val code = char - 'a'
                    if (node.children[code] == null) {
                        node.children[code] = TrieNode()
                    }
                    node = node.children[code]!!
                    node.count++
                }
            }
            return root
        }

        val root = buildTrie()

        return IntArray(words.size) { i ->
            var node = root
            var sum = 0
            for (char in words[i]) {
                node = node.children[char - 'a']!!
                sum += node.count
            }
            sum
        }
    }


    class AllOneTreeMap() : TreeMap<Int, HashSet<String>>() {
        val keyToFreq = HashMap<String, Int>()

        fun update(key: String, inc: Int) {
            val currFreq = keyToFreq.remove(key) ?: 0
            get(currFreq)?.let {
                it.remove(key)
                if (it.isEmpty()) {
                    remove(currFreq)
                }
            }
            val newFreq = currFreq + inc
            if (newFreq > 0) {
                getOrPut(newFreq) { HashSet() }.add(key)
                keyToFreq[key] = newFreq
            }
        }

        fun inc(key: String) = update(key, 1)
        fun dec(key: String) = update(key, -1)
        fun getMaxKey() = if (isEmpty()) "" else lastEntry()!!.value.first()
        fun getMinKey() = if (isEmpty()) "" else firstEntry()!!.value.first()
    }

    class AllOne() {
        class Node(val name: Int) {
            var previous: Node? = null
            var next: Node? = null
            var keys = mutableSetOf<String>()

            override fun toString(): String {
                return name.toString()
            }
        }

        val nodes = mutableMapOf<Int, Node>()
        val keyFreq = mutableMapOf<String, Int>()

        val head = Node(0)
        val tail = Node(Int.MAX_VALUE)

        init {
            head.next = tail
            tail.previous = head
            nodes[0] = head
            nodes[Int.MAX_VALUE] = tail
        }

        fun removeNode(node: Node) {
            node.previous?.next = node.next
            node.next?.previous = node.previous
            nodes.remove(node.name)
        }

        fun inc(key: String) {
            val oldFreq = keyFreq[key] ?: 0
            val newFreq = oldFreq + 1
            keyFreq[key] = newFreq
            val oldNode = nodes[oldFreq]!!
            oldNode.keys.remove(key)
            val newNode = nodes.getOrPut(newFreq) { Node(newFreq) }
            newNode.keys.add(key)
            if (newNode.name != oldNode.next?.name) {
                newNode.next = oldNode.next
                oldNode.next?.previous = newNode
                oldNode.next = newNode
                newNode.previous = oldNode
            }

            if (oldNode.keys.isEmpty() && oldNode != head) {
                removeNode(oldNode)
            }
        }

        fun dec(key: String) {
            val oldFreq = keyFreq[key]!!
            val newFreq = oldFreq - 1
            keyFreq[key] = newFreq
            val oldNode = nodes[oldFreq]!!
            oldNode.keys.remove(key)
            val newNode = nodes.getOrPut(newFreq) { Node(newFreq) }
            newNode.keys.add(key)
            if (newNode.name != oldNode.previous?.name) {
                newNode.previous = oldNode.previous
                oldNode.previous?.next = newNode
                oldNode.previous = newNode
                newNode.next = oldNode
            }
            if (oldNode.keys.isEmpty()) {
                removeNode(oldNode)
            }
        }

        fun getMaxKey(): String {
            if (tail.previous == head) return ""
            return tail.previous!!.keys.first()
        }

        fun getMinKey(): String {
            if (head.next == tail) return ""
            return head.next!!.keys.first()
        }

    }

    fun mostBooked(n: Int, meetings: Array<IntArray>): Int {
        data class Room(val index: Int, var endAt: Long)

        val roomMeetings = IntArray(n)
        var mostRoom = 0
        val occupied = PriorityQueue<Room>(compareBy { it.endAt })
        val available = PriorityQueue<Room>(compareBy { it.index })
        meetings.sortBy { it[0] }
        var delayedTo = 0L
        for ((start, end) in meetings) {
            val realStart = if (start < delayedTo) {
                delayedTo
            } else start * 1L
            val duration = end - start

            // free up
            while (occupied.isNotEmpty() && occupied.peek()!!.endAt <= realStart) {
                available.offer(occupied.poll()!!)
            }
            val nextRoom = if (available.isNotEmpty()) {
                available.poll()!!.apply { endAt = realStart + duration }
            } else if (occupied.size < n) {
                Room(occupied.size, realStart + duration)
            } else {
                // delay
                delayedTo = occupied.peek()!!.endAt
                // free up all end at the same time
                while (occupied.isNotEmpty() && occupied.peek()!!.endAt == delayedTo) {
                    available.offer(occupied.poll()!!)
                }
                available.poll()!!.apply { endAt = delayedTo + duration }
            }
            roomMeetings[nextRoom.index]++
            if (roomMeetings[nextRoom.index] > roomMeetings[mostRoom]) {
                mostRoom = nextRoom.index
            } else if (roomMeetings[nextRoom.index] == roomMeetings[mostRoom]) {
                mostRoom = minOf(mostRoom, nextRoom.index)
            }

            occupied.offer(nextRoom)
        }

        return mostRoom
    }

    fun smallestRange(nums: List<List<Int>>): IntArray {
        // pq + sliding window
//        var result = intArrayOf(0, 0)
//        var minRange = Int.MAX_VALUE
//        val pq = PriorityQueue<Triple<Int, Int, Int>>(compareBy { it.third }) // groupIndex, indexInGroup, num
//        var totalSize = 0
//        for (i in nums.indices) {
//            totalSize += nums[i].size
//            pq.offer(Triple(i, 0, nums[i][0]))
//        }
//        val allNums = Array(totalSize) { 0 to 0 }
//        var index = 0
//        var start = 0
//        val groupFreq = IntArray(nums.size)
//        var coveredGroups = 0
//        while (pq.isNotEmpty()) {
//            val (groupIndex, columnIndex, num) = pq.poll()!!
//            allNums[index++] = groupIndex to num
//            if (columnIndex != nums[groupIndex].lastIndex) {
//                pq.offer(Triple(groupIndex, columnIndex + 1, nums[groupIndex][columnIndex + 1]))
//            }
//
//            if (groupFreq[groupIndex] == 0) {
//                coveredGroups++
//            }
//            groupFreq[groupIndex]++
//
//            while (start < index && groupFreq[allNums[start].first] > 1) {
//                groupFreq[allNums[start].first]--
//                start++
//            }
//            if (coveredGroups == nums.size) {
//                val rangeStart = allNums[start].second
//                val newRange = num - rangeStart
//                if (newRange < minRange || newRange == minRange && rangeStart < result[0]) {
//                    result = intArrayOf(rangeStart, num)
//                    minRange = newRange
//                }
//            }
//        }
//        return result

        // pq can be used as the window
        val pq = PriorityQueue<IntArray>(compareBy { it[2] }) // groupIndex, indexInGroup, num
        var currentMax = Int.MIN_VALUE
        for (i in nums.indices) {
            pq.offer(intArrayOf(i, 0, nums[i][0]))
            currentMax = maxOf(currentMax, nums[i][0])
        }
        var start = 0
        var end = Int.MAX_VALUE
        while (pq.size == nums.size) { // make sure pq covers all groups
            val (groupIndex, indexInGroup, num) = pq.poll()!!
            if (currentMax - num < end - start) {
                start = num
                end = currentMax
            }
            if (indexInGroup != nums[groupIndex].lastIndex) {
                val nextValue = nums[groupIndex][indexInGroup + 1]
                pq.offer(intArrayOf(groupIndex, indexInGroup + 1, nextValue))
                currentMax = maxOf(currentMax, nextValue)
            }
        }
        return intArrayOf(start, end)
    }

    class MedianFinder() {
        val pq1 = PriorityQueue<Int>(reverseOrder())
        var size1 = 0
        val pq2 = PriorityQueue<Int>()
        var size2 = 0

        init {
            pq1.offer(Int.MIN_VALUE)
            pq2.offer(Int.MAX_VALUE)
        }

        fun addNum(num: Int) {
            if (num <= pq1.peek()!!) {
                pq1.offer(num)
                size1++
                if (size1 - 1 > size2) {
                    pq2.offer(pq1.poll()!!)
                    size1--
                    size2++
                }
            } else {
                pq2.offer(num)
                size2++
                if (size2 - 1 > size1) {
                    pq1.offer(pq2.poll()!!)
                    size2--
                    size1++
                }
            }
        }

        fun findMedian(): Double {
            if ((size1 + size2) % 2 == 0) {
                return (pq1.peek()!! + pq2.peek()!!) / 2.0
            } else if (size1 > size2) {
                return pq1.peek()!! * 1.0
            } else {
                return pq2.peek()!! * 1.0
            }
        }
    }

    fun minimumMountainRemovals(nums: IntArray): Int {
        // pre-compute two longest increasing subsequence (LIS)
        val n = nums.size
        val leftDp = IntArray(n) { 1 }
        val rightDp = IntArray(n) { 1 }

        for (i in 1 until n) {
            for (j in 0 until i) {
                if (nums[i] > nums[j]) {
                    leftDp[i] = maxOf(leftDp[i], leftDp[j] + 1)
                }
            }
        }
        for (i in n - 2 downTo 0) {
            for (j in n - 1 downTo i + 1) {
                if (nums[i] > nums[j]) {
                    rightDp[i] = maxOf(rightDp[i], rightDp[j] + 1)
                }
            }
        }
        var maxLength = 0
        for (i in 1 until n - 1) {
            // must bigger than 1
            if (leftDp[i] > 1 && rightDp[i] > 1) {
                maxLength = maxOf(maxLength, leftDp[i] + rightDp[i] - 1)
            }
        }
        return n - maxLength
    }

//    fun minimumTotalDistance(robot: List<Int>, factory: Array<IntArray>): Long {
//        val robotArray = robot.sorted()
//        factory.sortBy { it[0] }
//        val factoryList = mutableListOf<Int>()
//        for ((facPos, count) in factory) {
//            repeat(count) {
//                factoryList.add(facPos)
//            }
//        }
//        val n = robot.size
//        val m = factoryList.size
//        val dp = Array(n + 1) { LongArray(m + 1) { Long.MAX_VALUE / 2 } }
//        dp[0][0] = 0  // 没有人和没有工厂的情况下，移动距离为0
//        for (i in 0..n) {
//            for (j in 1..m) {
//                dp[i][j] = dp[i][j - 1]  // 不使用第 j 个
//                if (i > 0) {
//                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + abs(robotArray[i - 1] - factoryList[j - 1]))
//                }
//            }
//        }
//        return dp[n][m]
//    }

    fun minimumTotalDistance(robot: List<Int>, factory: Array<IntArray>): Long {
        factory.sortBy { it[0] }
        val robotArray = robot.sorted()
        val dp = Array(robotArray.size) { LongArray(factory.size + 1) { Long.MAX_VALUE / 2 } }
        for (robotIndex in robotArray.indices) {
            for ((facIndex, fac) in factory.withIndex()) {
                dp[robotIndex][facIndex + 1] = dp[robotIndex][facIndex]
                var dist = 0L
                // 如果要选当前工厂，将倒数最多x个机器人都考虑入选
                for (i in robotIndex downTo maxOf(0, robotIndex - fac[1] + 1)) {
                    dist += abs(robotArray[i] - fac[0])
                    // i-1 是前一个机器人下标，dp[i - 1][facIndex]表示前面所有机器人和前面所有工厂的最优解
                    dp[robotIndex][facIndex + 1] =
                        minOf(
                            dp[robotIndex][facIndex + 1],
                            dist + if (i == 0) 0L else dp[i - 1][facIndex]
                        )
                }
            }
        }
        return dp.last().last()
    }

    fun shortestSubarray(nums: IntArray, k: Int): Int {
        var min = Int.MAX_VALUE
        val prefixSum = LongArray(nums.size + 1)
        val queue = LinkedList<Int>()
        for (i in prefixSum.indices) {
            if (i > 0) {
                prefixSum[i] = prefixSum[i - 1] + nums[i - 1]
            }

            while (queue.isNotEmpty() && prefixSum[i] - prefixSum[queue.first] >= k) {
                min = minOf(min, i - queue.pollFirst()!!)
            }

            while (queue.isNotEmpty() && prefixSum[i] <= prefixSum[queue.last]) {
                queue.pollLast()
            }

            queue.add(i)
        }

        return if (min == Int.MAX_VALUE) -1 else min
    }


    fun slidingPuzzle(board: Array<IntArray>): Int {

        fun getNeighbours(pos: Int): IntArray {
            return when (pos) {
                0 -> intArrayOf(1, 3)
                1 -> intArrayOf(0, 2, 4)
                2 -> intArrayOf(1, 5)
                3 -> intArrayOf(0, 4)
                4 -> intArrayOf(1, 3, 5)
                else -> intArrayOf(2, 4)
            }
        }

        val target = intArrayOf(1, 2, 3, 4, 5, 0)
        val seen = mutableSetOf<Int>()
        val initArray = IntArray(6) {
            board[it / 3][it % 3]
        }
        var initPos = -1
        for (i in initArray.indices) {
            if (initArray[i] == 0) {
                initPos = i
                break
            }
        }
        val queue = LinkedList<Triple<IntArray, Int, Int>>()
        queue.offer(Triple(initArray.copyOf(), initPos, 0))
        while (queue.isNotEmpty()) {
            val (array, zeroPos, count) = queue.poll()!!
            if (array.contentEquals(target)) return count
            getNeighbours(zeroPos).forEach { neighbour ->
                val arrayCopy = array.copyOf()
                arrayCopy[zeroPos] = arrayCopy[neighbour]
                arrayCopy[neighbour] = 0
                val pattern = arrayCopy.contentHashCode()
                if (pattern !in seen) {
                    seen.add(pattern)
                    queue.offer(Triple(arrayCopy, neighbour, count + 1))
                }
            }
        }
        return -1
    }

    fun minimumObstacles(grid: Array<IntArray>): Int {
        val m = grid.size
        val n = grid[0].size
        // dp with relaxation
//        val dp = Array(m) { IntArray(n) { Int.MAX_VALUE } }
//        dp[0][0] = 0
//        var changed = true
//        while (changed) {
//            changed = false
//            for (row in 0 until m) {
//                for (col in 0 until n) {
//                    if (row == 0 && col == 0) continue
//                    val newValue = minOf(
//                        if (row > 0) dp[row - 1][col] else Int.MAX_VALUE,
//                        if (col > 0) dp[row][col - 1] else Int.MAX_VALUE,
//                        if (row < m - 1) dp[row + 1][col] else Int.MAX_VALUE,
//                        if (col < n - 1) dp[row][col + 1] else Int.MAX_VALUE,
//                    ) + if (grid[row][col] == 1) 1 else 0
//                    if (newValue < dp[row][col]) {
//                        dp[row][col] = newValue
//                        changed = true
//                    }
//                }
//            }
//        }
//        return dp[m - 1][n - 1]

        // Dijkstra
        val DIRECTIONS = arrayOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)
        val visited = Array(m) { BooleanArray(n) }
        visited[0][0] = true
        val pq = PriorityQueue<Triple<Int, Int, Int>>(compareBy { it.third })
        pq.offer(Triple(0, 0, 0))
        while (pq.isNotEmpty()) {
            val (row, col, current) = pq.poll()!!
            if (row == m - 1 && col == n - 1) return current
            for ((deltaX, deltaY) in DIRECTIONS) {
                val newX = row + deltaX
                val newY = col + deltaY
                if (newX in 0 until m && newY in 0 until n && !visited[newX][newY]) {
                    val newCost = if (grid[newX][newY] == 1) current + 1 else current
                    visited[newX][newY] = true
                    pq.offer(Triple(newX, newY, newCost))

                }
            }
        }
        return -1
    }

    fun minimumTime(grid: Array<IntArray>): Int {
        if (grid[0][1] > 1 && grid[1][0] > 1) return -1
        // Dijkstra
        val DIRECTIONS = arrayOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)
        val m = grid.size
        val n = grid[0].size
        val visited = Array(m) { BooleanArray(n) }
        visited[0][0] = true
        val pq = PriorityQueue<Triple<Int, Int, Int>>(compareBy { it.third })
        pq.offer(Triple(0, 0, 0))
        while (pq.isNotEmpty()) {
            val (row, col, currentTime) = pq.poll()!!
            if (row == m - 1 && col == n - 1) return currentTime
            for ((deltaX, deltaY) in DIRECTIONS) {
                val newRow = row + deltaX
                val newCol = col + deltaY
                if (newRow in 0 until m && newCol in 0 until n && !visited[newRow][newCol]) {
                    var newTime = currentTime + 1
                    if (newTime < grid[newRow][newCol]) {
                        newTime = grid[newRow][newCol] + (grid[newRow][newCol] - newTime) % 2
                    }
                    visited[newRow][newCol] = true
                    pq.offer(Triple(newRow, newCol, newTime))
                }
            }
        }
        return -1
    }

    fun validArrangement(pairs: Array<IntArray>): Array<IntArray> {
        // Hierholzer 算法找 Eulerian 路径
        val inDegree = mutableMapOf<Int, Int>()
        val outDegree = mutableMapOf<Int, Int>()
        val graph = mutableMapOf<Int, MutableList<Int>>()
        for ((start, end) in pairs) {
            graph.getOrPut(start) { mutableListOf() }.add(end)
            outDegree[start] = outDegree.getOrDefault(start, 0) + 1
            inDegree[end] = inDegree.getOrDefault(end, 0) + 1
        }
        val path = mutableListOf<Int>() // 倒序存储的路径
        val result = mutableListOf<IntArray>()

        fun dfs(start: Int) {
            graph[start]?.let {
                while (it.isNotEmpty()) {
                    val v = it.removeAt(0)
                    dfs(v)
                }
            }
            path.add(start)
        }

        // 起点的入度比出度少1
        var startNode = -1
        for (node in outDegree.keys + inDegree.keys) {
            val out = outDegree.getOrDefault(node, 0)
            val inD = inDegree.getOrDefault(node, 0)
            if (out - inD == 1) {
                startNode = node
                break
            }
        }
        // 没有明确起点，欧拉回路，随便选一个点
        if (startNode == -1) {
            startNode = pairs[0][0]
        }

        dfs(startNode)

        for (i in path.size - 2 downTo 0) {
            result.add(intArrayOf(path[i + 1], path[i]))
        }
        return result.toTypedArray()
    }

    fun maxKDivisibleComponents(n: Int, edges: Array<IntArray>, values: IntArray, k: Int): Int {
        if (n == 1) return 1
        var split = 0
        val graph = Array(n) { mutableListOf<Int>() }
        for ((a, b) in edges) {
            graph[a].add(b)
            graph[b].add(a)
        }

        //        val leaves = LinkedList<Int>()
        //        for (i in 0 until n) {
        //            if (graph[i].size == 1) {
        //                leaves.offer(i)
        //            }
        //        }
        //        while (leaves.isNotEmpty()) {
        //            val size = leaves.size
        //            repeat(size) {
        //                val leaf = leaves.poll()!!
        //                graph[leaf].getOrNull(0)?.let { neighbour ->
        //                    if (values[leaf] % k == 0) {
        //                        split++
        //                    } else {
        //                        values[neighbour] = (values[neighbour] % k + values[leaf] % k) % k
        //                    }
        //                    graph[neighbour].remove(leaf)
        //                    if (graph[neighbour].size == 1) {
        //                        leaves.offer(neighbour)
        //                    }
        //                }
        //            }
        //        }

        // cut leaves
//        fun treatLeaf(leaf: Int) {
//            graph[leaf].getOrNull(0)?.let { neighbour ->
//                if (values[leaf] % k == 0) {
//                    split++
//                } else {
//                    values[neighbour] = (values[neighbour] % k + values[leaf] % k) % k
//                }
//                graph[neighbour].remove(leaf)
//                if (graph[neighbour].size == 1) {
//                    treatLeaf(neighbour)
//                }
//            }
//            graph[leaf].removeAt(0)
//        }
//        for (i in 0 until n) {
//            if (graph[i].size == 1) {
//                treatLeaf(i)
//            }
//        }
//        return split + 1

        // dfs calculate subtree sum and cut
        fun treeSum(root: Int, parent: Int): Int {
            var sum = values[root]
            for (neighbour in graph[root]) {
                if (neighbour == parent) continue
                sum += treeSum(neighbour, root)
            }
            if (sum % k == 0) {
                // cut
                split++
                return 0
            }
            return sum % k
        }

        treeSum(0, -1)
        return split
    }

    fun leftmostBuildingQueries(heights: IntArray, queries: Array<IntArray>): IntArray {
        for (q in queries) {
            if (q[0] > q[1]) {
                val tmp = q[0]
                q[0] = q[1]
                q[1] = tmp
            }
        }
        val sortedQueries = queries.withIndex()
            .sortedWith(compareByDescending<IndexedValue<IntArray>> { it.value[1] }.thenBy { it.value[0] })
        val result = IntArray(queries.size) { -1 }
        val stack = LinkedList<Int>()
        var i = heights.size - 1
        var lastA = -1
        var lastB = -1
        var lastResult = -1
        for ((index, query) in sortedQueries) {
            val (a, b) = query
            if (a == lastA && b == lastB) {
                result[index] = lastResult
            } else if (a == b || heights[a] < heights[b]) {
                result[index] = b
            } else {
                while (i > b) {
                    while (stack.isNotEmpty() && heights[stack.first()] < heights[i]) {
                        stack.pollFirst()
                    }
                    stack.addFirst(i)
                    i--
                }
                var left = 0
                var right = stack.size - 1
                while (left <= right) {
                    val mid = left + ((right - left) shr 1)
                    if (heights[stack[mid]] > heights[a]) {
                        right = mid - 1
                    } else {
                        left = mid + 1
                    }
                }
                if (left in stack.indices) {
                    result[index] = stack[left]
                }
            }
            lastA = a
            lastB = b
            lastResult = result[index]
        }
        return result
    }

    fun minimumDiameterAfterMerge(edges1: Array<IntArray>, edges2: Array<IntArray>): Int {

        fun findDiameter(edges: Array<IntArray>): Int {
            val n = edges.size + 1
            val graph = Array(n) { mutableListOf<Int>() }
            for ((u, v) in edges) {
                graph[u].add(v)
                graph[v].add(u)
            }

            var maxDepth = 0
            var farthestNode = -1

            fun dfs(current: Int, parent: Int, depth: Int) {
                if (graph[current].size == 1 && graph[current][0] == parent) {
                    if (depth > maxDepth) {
                        maxDepth = depth
                        farthestNode = current
                    }
                    return
                }
                graph[current].forEach { child ->
                    if (child != parent) {
                        dfs(child, current, depth + 1)
                    }
                }
            }

            dfs(0, -1, 0)
            if (farthestNode != -1) {
                dfs(farthestNode, -1, 0)
            }
            return maxDepth
        }

        val d1 = findDiameter(edges1)
        val d2 = findDiameter(edges2)
        return maxOf(d1, d2, (d1 + 1) / 2 + (d2 + 1) / 2 + 1)
    }

    fun maxSumOfThreeSubarrays(nums: IntArray, k: Int): IntArray {
        val n = nums.size
        val sums = IntArray(n - k + 1)
        var start = 0
        var sum = 0
        for (end in 0 until n) {
            sum += nums[end]
            if (end - start + 1 != k) continue
            sums[start] = sum
            sum -= nums[start]
            start++
        }
        val leftBest = IntArray(n - k + 1)
        var best = 0
        for (i in sums.indices) {
            if (sums[i] > sums[best]) {
                best = i
            }
            leftBest[i] = best
        }
        val rightBest = IntArray(n - k + 1)
        best = 0
        for (i in sums.indices.reversed()) {
            if (sums[i] >= sums[best]) {
                best = i
            }
            rightBest[i] = best
        }
        sum = 0
        val result = intArrayOf(0, 0, 0)
        for (i in sums.indices) {
            if (i - k < 0) continue
            if (i + k >= sums.size) break
            val currentSum = sums[leftBest[i - k]] + sums[i] + sums[rightBest[i + k]]
            if (currentSum > sum) {
                sum = currentSum
                result[0] = leftBest[i - k]
                result[1] = i
                result[2] = rightBest[i + k]
            }
        }
        return result
    }

    fun minCostToReachLast(grid: Array<IntArray>): Int {
        val DIRECTIONS = arrayOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)
        val m = grid.size
        val n = grid[0].size
//        val pq = PriorityQueue<Triple<Int, Int, Int>>(compareBy { it.third })
//        val minMemo = Array(m) { IntArray(n) { Int.MAX_VALUE} }
//        minMemo[0][0] = 0
//        pq.offer(Triple(0, 0, 0))
//        while (pq.isNotEmpty()) {
//            val (i, j, cost) = pq.poll()!!
//            if (i == m - 1 && j == n - 1) return cost
//            for (k in DIRECTIONS.indices) {
//                val nextI = i + DIRECTIONS[k][0]
//                val nextJ = j + DIRECTIONS[k][1]
//                if (nextI !in 0 until m || nextJ !in 0 until n) continue
//                val nextCost = if (k + 1 == grid[i][j]) cost else cost + 1
//                if (nextCost < minMemo[nextI][nextJ]) {
//                    minMemo[nextI][nextJ] = nextCost
//                    pq.offer(Triple(nextI, nextJ, nextCost))
//                }
//            }
//        }
//        return -1

        val deque = LinkedList<Triple<Int, Int, Int>>() // 使用双端队列
        val minMemo = Array(m) { IntArray(n) { Int.MAX_VALUE } }
        minMemo[0][0] = 0
        deque.addFirst(Triple(0, 0, 0))
        while (deque.isNotEmpty()) {
            val (i, j, cost) = deque.removeFirst()
            if (i == m - 1 && j == n - 1) return cost
            for (k in DIRECTIONS.indices) {
                val nextI = i + DIRECTIONS[k].first
                val nextJ = j + DIRECTIONS[k].second
                if (nextI !in 0 until m || nextJ !in 0 until n) continue
                val nextCost = if (k + 1 == grid[i][j]) cost else cost + 1
                if (nextCost < minMemo[nextI][nextJ]) {
                    minMemo[nextI][nextJ] = nextCost
                    if (k + 1 == grid[i][j]) { // 以同方向为优
                        deque.addFirst(Triple(nextI, nextJ, nextCost)) // 权重为 0 的路径
                    } else {
                        deque.addLast(Triple(nextI, nextJ, nextCost)) // 权重为 1 的路径
                    }
                }
            }
        }
        return -1
    }

    fun maximumInvitations(favorite: IntArray): Int {
        val n = favorite.size
        val inDegree = IntArray(n)
        for (person in 0 until n) {
            inDegree[favorite[person]]++
        }

        // Topological sorting to remove non-cycle nodes
        val queue = LinkedList<Int>()
        for (person in 0 until n) {
            if (inDegree[person] == 0) {
                queue.offer(person)
            }
        }
        val depth = IntArray(n) { 1 } // Depth: node to leaf
        while (queue.isNotEmpty()) {
            val currentNode = queue.poll()!!
            val nextNode = favorite[currentNode]
            depth[nextNode] = maxOf(depth[nextNode], depth[currentNode] + 1)
            if (--inDegree[nextNode] == 0) {
                queue.offer(nextNode)
            }
        }

        var longestCycle = 0
        var twoCycleInvitations = 0

        // Detect cycles
        for (person in 0 until n) {
            if (inDegree[person] == 0) continue  // Already processed
            var cycleLength = 0
            var current = person
            while (inDegree[current] != 0) {
                inDegree[current] = 0 // Mark as visited
                cycleLength++
                current = favorite[current]
            }

            if (cycleLength == 2) {
                // For 2-cycles, add the depth of both nodes
                twoCycleInvitations += depth[person] + depth[favorite[person]]
            } else {
                longestCycle = maxOf(longestCycle, cycleLength)
            }
        }

        return maxOf(longestCycle, twoCycleInvitations)
    }

    // 二分染色法判断有无奇环
//        val color = IntArray(n + 1) { -1 }
//        for (start in 1..n) {
//            if (color[start] != -1) continue
//            color[start] = 0
//            val queue = LinkedList<Int>()
//            queue.offer(start)
//            while (queue.isNotEmpty()) {
//                val node = queue.poll()!!
//                for (neighbour in graph[node]) {
//                    if (color[neighbour] == -1) {
//                        color[neighbour] = 1 - color[node]
//                        queue.offer(neighbour)
//                    } else if (color[neighbour] == color[node]) {
//                        return -1
//                    }
//                }
//            }
//        }

    fun magnificentSets(n: Int, edges: Array<IntArray>): Int {
        val uf = UnionFind(n + 1)
        val graph = Array(n + 1) { mutableListOf<Int>() }
        for ((u, v) in edges) {
            graph[u].add(v)
            graph[v].add(u)
            uf.union(u, v)
        }

        // 以每个节点为树root计算树高，如果同层节点相连，则是有奇环。
        val height = mutableMapOf<Int, Int>()
        for (start in 1..n) {
            val visitedAt = IntArray(n + 1) { -1 }
            val queue = LinkedList<Int>()
            queue.offer(start)
            visitedAt[start] = 0
            var currentHeight = 0
            while (queue.isNotEmpty()) {
                val levelSize = queue.size
                for (i in 0 until levelSize) {
                    val node = queue.poll()!!
                    for (neighbour in graph[node]) {
                        if (visitedAt[neighbour] == -1) {
                            visitedAt[neighbour] = currentHeight + 1
                            queue.offer(neighbour)
                        } else if (visitedAt[neighbour] == currentHeight) {
                            return -1
                        }
                    }
                }
                currentHeight++
            }
            // 找到连通分量顶点，更新最高高度
            val root = uf.find(start)
            height[root] = maxOf(height.getOrDefault(root, 0), currentHeight)
        }

        var result = 0
        for ((key, value) in height) {
            result += value
        }
        return result
    }

    val DIRECTIONS = arrayOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)
    //    fun largestIsland(grid: Array<IntArray>): Int {
    //        val n = grid.size
    //        val maxSize = n * n
    //        var islandId = 2
    //        var maxArea = 0
    //        val areas = IntArray(maxSize + 2)
    //
    //        fun dfs(r: Int, c: Int, id: Int) {
    //            grid[r][c] = id
    //            areas[id]++
    //            for ((dx, dy) in DIRECTIONS) {
    //                val newR = r + dx
    //                val newC = c + dy
    //                if (newR !in 0 until n || newC !in 0 until n || grid[newR][newC] != 1) continue
    //                dfs(newR, newC, id)
    //            }
    //        }
    //
    //        for (i in 0 until n) {
    //            for (j in 0 until n) {
    //                if (grid[i][j] == 1) {
    //                    dfs(i, j, islandId)
    //                    maxArea = maxOf(maxArea, areas[islandId])
    //                    islandId++
    //                }
    //            }
    //        }
    //        if (maxArea == 0) return 1
    //        else if (maxArea == maxSize) return maxSize
    //
    //        for (i in 0 until n) {
    //            for (j in 0 until n) {
    //                if (grid[i][j] == 0) {
    //                    var areaSum = 1
    //                    val neighbours = mutableSetOf<Int>()
    //                    for ((dx, dy) in DIRECTIONS) {
    //                        val newX = i + dx
    //                        val newY = j + dy
    //                        if (newX in 0 until n && newY in 0 until n && grid[newX][newY] > 1) {
    //                            neighbours.add(grid[newX][newY])
    //                        }
    //                    }
    //                    for (neighbour in neighbours) {
    //                        areaSum += areas[neighbour]
    //                    }
    //                    maxArea = maxOf(maxArea, areaSum)
    //                }
    //            }
    //        }
    //        return maxArea
    //    }

    fun largestIsland(grid: Array<IntArray>): Int {
        val n = grid.size

        fun index(x: Int, y: Int) : Int = x * n + y

        val uf = UnionFind(n * n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (grid[i][j] == 1) {
                    if (i < n -1 && grid[i + 1][j] == 1) {
                        uf.union(index(i, j), index(i + 1, j))
                    }
                    if (j < n - 1 && grid[i][j + 1] == 1) {
                        uf.union(index(i, j), index(i, j + 1))
                    }
                }
            }
        }
        var maxArea = 0
        val areas = mutableMapOf<Int, Int>()
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (grid[i][j] == 1) {
                    val root = uf.find(index(i, j))
                    areas[root] = areas.getOrDefault(root, 0) + 1
                    maxArea = maxOf(maxArea, areas[root]!!)
                }
            }
        }
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (grid[i][j] == 0) {
                    var areaSum = 1
                    val neighbours = mutableSetOf<Int>()
                    for ((dx, dy) in DIRECTIONS) {
                        val newX = i + dx
                        val newY = j + dy
                        if (newX in 0 until n && newY in 0 until n && grid[newX][newY] == 1) {
                            neighbours.add(uf.find(index(newX, newY)))
                        }
                    }
                    for (neighbour in neighbours) {
                        areaSum += areas[neighbour]!!
                    }
                    maxArea = maxOf(maxArea, areaSum)
                }
            }
        }
        return maxArea
    }

    fun maxPointsSameLine(points: Array<IntArray>): Int {
        //fun gcd(a: Int, b: Int): Int {
        //            return if (b == 0) a else gcd(b, a % b)
        //        }
        //
        //        var result = 0
        //        for (i in points.indices) {
        //            val freq = mutableMapOf<Pair<Int, Int>, Int>()
        //            var max = 0
        //            for (j in i + 1 until points.size) {
        //                var dx = points[j][0] - points[i][0]
        //                var dy = points[j][1] - points[i][1]
        //                if (dx == 0) {
        //                    dy = 1
        //                } else if (dy == 0) {
        //                    dx = 1
        //                } else {
        //                    val factor = gcd(dx, dy)
        //                    dx /= factor
        //                    dy /= factor
        //                }
        //                val key = dx to dy
        //                freq[key] = freq.getOrDefault(key, 0) + 1
        //                max = maxOf(max, freq[key]!!)
        //            }
        //            result = maxOf(result, max + 1)
        //        }
        //        return result

        if (points.size <= 2) return points.size
        var max = 2
        for (i in 0 until points.size) {
            val x1 = points[i][0]
            val y1 = points[i][1]
            for (j in i + 1 until points.size) {
                val x2 = points[j][0]
                val y2 = points[j][1]
                var localMax = 2
                for (z in j + 1 until points.size) {
                    val x = points[z][0]
                    val y = points[z][1]
                    if ((x - x1) * (y2 - y1) == (y - y1) * (x2 - x1)) {
                        localMax++
                    }
                }
                max = maxOf(max, localMax)
            }
        }
        return max
    }
}