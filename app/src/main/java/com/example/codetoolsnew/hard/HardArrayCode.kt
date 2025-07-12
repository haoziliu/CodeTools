package com.example.codetools.hard

import com.example.codetools.ArrayCode
import java.util.LinkedList
import java.util.PriorityQueue
import java.util.Stack
import java.util.TreeMap
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min


object HardArrayCode {
    private val MODULO = 1_000_000_007

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
        //        // better solution
        //        var isFlippedOdd = false
        //        var flippedMinDiff = Int.MAX_VALUE // flipped > original, minimum delta
        //        var unFlippedMinDiff = Int.MAX_VALUE // flipped < original, minimum abs delta
        //        var sum = 0L
        //
        //        for (num in nums) {
        //            val flipped = num xor k
        //            val newDiff = flipped - num
        //            if (flipped > num) {
        //                isFlippedOdd = !isFlippedOdd
        //                sum += flipped
        //                flippedMinDiff = minOf(flippedMinDiff, flipped - num)
        //            } else {
        //                sum += num
        //                unFlippedMinDiff = minOf(unFlippedMinDiff, num - flipped)
        //            }
        //        }
        //        if (isFlippedOdd) {
        //            return sum - minOf(flippedMinDiff, unFlippedMinDiff)
        //        } else {
        //            return sum
        //        }

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
        val n = ratings.size
        if (n == 1) return 1
        val candies = IntArray(n) { 1 }
        for (i in 1 until n) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1
            }
        }
        var result = ratings[n - 1]
        for (i in n - 2 downTo 0) {
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = maxOf(candies[i], candies[i + 1] + 1)
            }
            result += candies[i]
        }
        return result
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
        data class Room(val index: Int, var end: Long)

        meetings.sortBy { it[0] }
        val hold = IntArray(n)
        var maxIndex = 0
        val pqFree = PriorityQueue<Int>()
        val pqTaken = PriorityQueue(compareBy<Room> { it.end }.thenBy { it.index })
        for (i in 0 until n) {
            pqFree.offer(i)
        }
        for (meeting in meetings) {
            val meetingStart = meeting[0].toLong()
            val meetingEnd = meeting[1].toLong()
            val duration = meetingEnd - meetingStart
            while (pqTaken.isNotEmpty() && pqTaken.peek()!!.end <= meetingStart) {
                pqFree.offer(pqTaken.poll()!!.index)
            }
            val room = if (pqFree.isNotEmpty()) {
                Room(pqFree.poll()!!, meetingEnd)
            } else {
                pqTaken.poll()!!.apply { end += duration }
            }
            pqTaken.offer(room)
            hold[room.index]++
            if (hold[room.index] > hold[maxIndex] ||
                hold[room.index] == hold[maxIndex] && maxIndex > room.index
            ) {
                maxIndex = room.index
            }
        }
        return maxIndex
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

        fun index(x: Int, y: Int): Int = x * n + y

        val uf = UnionFind(n * n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (grid[i][j] == 1) {
                    if (i < n - 1 && grid[i + 1][j] == 1) {
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

    fun longestCommonPrefixAfterExclude(words: Array<String>, k: Int): IntArray {
        class TrieNode(var depth: Int = 0) {
            val children = Array<TrieNode?>(26) { null }
            var count = 0
        }

        fun buildTrie(): TrieNode {
            val root = TrieNode()
            for (i in words.indices) {
                var node = root
                node.count++
                for (letter in words[i]) {
                    val idx = letter - 'a'
                    if (node.children[idx] == null) {
                        node.children[idx] = TrieNode(node.depth + 1)
                    }
                    node = node.children[idx]!!
                    node.count++
                }
            }
            return root
        }

        val trieRoot = buildTrie()
        val validCount = TreeMap<Int, Int>()

        fun addValid(depth: Int) {
            validCount[depth] = validCount.getOrDefault(depth, 0) + 1
        }

        fun removeValid(depth: Int) {
            val count = validCount.getOrDefault(depth, 0)
            if (count <= 1) {
                validCount.remove(depth)
            } else {
                validCount[depth] = count - 1
            }
        }

        fun dfs(node: TrieNode) {
            if (node.count >= k) {
                addValid(node.depth)
            }
            for (child in node.children) {
                if (child != null) {
                    dfs(child)
                }
            }
        }
        dfs(trieRoot)

        fun updateNode(node: TrieNode, delta: Int) {
            val oldCount = node.count
            val wasValid = oldCount >= k
            node.count = oldCount + delta
            val isValid = node.count >= k
            if (wasValid && !isValid) {
                removeValid(node.depth)
            } else if (!wasValid && isValid) {
                addValid(node.depth)
            }
        }

        fun removeWord(word: String) {
            var node = trieRoot
            updateNode(node, -1)
            for (letter in word) {
                node = node.children[letter - 'a']!!
                updateNode(node, -1)
            }
        }

        fun restoreWord(word: String) {
            var node = trieRoot
            updateNode(node, 1)
            for (letter in word) {
                node = node.children[letter - 'a']!!
                updateNode(node, 1)
            }
        }

        val n = words.size
        val result = IntArray(n)
        if (n <= k) return result
        for (i in 0 until n) {
            removeWord(words[i])
            val maxDepth = if (validCount.isEmpty()) 0 else validCount.lastKey()
            result[i] = maxDepth
            restoreWord(words[i])
        }
        return result
    }

    fun maxPoints(grid: Array<IntArray>, queries: IntArray): IntArray {
        val DIRECTIONS = arrayOf(1 to 0, -1 to 0, 0 to 1, 0 to -1)
        val m = grid.size
        val n = grid[0].size
        val sorted = queries.withIndex().sortedBy { it.value }
        val pq = PriorityQueue<Triple<Int, Int, Int>>(compareBy { it.first }) // value, x, y
        pq.offer(Triple(grid[0][0], 0, 0))
        grid[0][0] *= -1
        var total = 0
        val result = IntArray(queries.size)
        for ((originalIndex, query) in sorted) {
            while (pq.isNotEmpty() && pq.peek()!!.first < query) {
                val (value, x, y) = pq.poll()!!
                total++
                for ((dx, dy) in DIRECTIONS) {
                    val newX = x + dx
                    val newY = y + dy
                    if (newX !in 0 until m || newY !in 0 until n || grid[newX][newY] < 0) continue
                    pq.offer(Triple(grid[newX][newY], newX, newY))
                    grid[newX][newY] *= -1
                }
            }
            result[originalIndex] = total
        }
        return result
    }

    fun maxPointsUF(grid: Array<IntArray>, queries: IntArray): IntArray {
        val m = grid.size
        val n = grid[0].size
        val sorted = queries.withIndex().sortedBy { it.value }

        val cells = Array(m * n) { index ->
            index / n to index % n
        }
        cells.sortBy { grid[it.first][it.second] }

        class UnionFind(size: Int) {
            val parent = IntArray(size) { it }
            val count = IntArray(size) { 1 }

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
                    if (count[rootX] >= count[rootY]) {
                        parent[rootY] = rootX
                        count[rootX] += count[rootY]
                    } else {
                        parent[rootX] = rootY
                        count[rootY] += count[rootX]
                    }
                }
            }
        }

        val uf = UnionFind(m * n)

        val activated = Array(m) { BooleanArray(n) }
        var index = 0
        val result = IntArray(queries.size)
        for ((originalIndex, query) in sorted) {
            while (index < cells.size) {
                val (x, y) = cells[index]
                if (grid[x][y] >= query) break
                activated[x][y] = true
                for ((dx, dy) in DIRECTIONS) {
                    val newX = x + dx
                    val newY = y + dy
                    if (newX !in 0 until m || newY !in 0 until n || !activated[newX][newY]) continue
                    uf.union(x * n + y, newX * n + newY)
                }
                index++
            }
            result[originalIndex] = if (activated[0][0]) {
                uf.count[uf.find(0)]
            } else {
                0
            }
        }
        return result
    }

    fun maximumScore(nums: List<Int>, k: Int): Int {
        val n = nums.size
        var maxNum = 0
        for (num in nums) {
            maxNum = maxOf(maxNum, num)
        }
        val limit = Math.sqrt(maxNum.toDouble()).toInt() + 1
        val isPrime = BooleanArray(limit + 1) { true }
        if (limit >= 0) {
            isPrime[0] = false
            if (limit >= 1) isPrime[1] = false
        }
        for (i in 2..limit) {
            if (isPrime[i]) {
                for (j in i * i..limit step i) {
                    isPrime[j] = false
                }
            }
        }
        val primes = mutableListOf<Int>()
        for (i in 2..limit) {
            if (isPrime[i]) primes.add(i)
        }

        val primeFactorCache = mutableMapOf<Int, Int>()

        fun countPrimeFactors(num: Int): Int {
            if (num in primeFactorCache) return primeFactorCache[num]!!
            var nVal = num
            var count = 0
            for (p in primes) {
                if (p * p > nVal) break
                var found = false
                while (nVal % p == 0) {
                    found = true
                    nVal /= p
                }
                if (found) count++
            }
            if (nVal > 1) count++
            primeFactorCache[num] = count
            return count
        }

        fun modPow(a: Int, b: Long): Int {
            var result = 1L
            var base = a.toLong() % MODULO
            var exp = b
            while (exp > 0) {
                if (exp and 1L == 1L) {
                    result = (result * base) % MODULO
                }
                base = (base * base) % MODULO
                exp = exp shr 1
            }
            return result.toInt()
        }

        val scores = IntArray(n)
        val prevGreaterOrEqual = IntArray(n) { -1 }
        val stack = ArrayDeque<Int>()
        for (i in 0 until n) {
            scores[i] = countPrimeFactors(nums[i])
            while (stack.isNotEmpty() && scores[stack.last()] < scores[i]) {
                stack.removeLast()
            }
            if (stack.isNotEmpty()) {
                prevGreaterOrEqual[i] = stack.last()
            }
            stack.addLast(i)
        }
        val contributions = LongArray(n)
        val nextGreater = IntArray(n) { n }
        stack.clear()
        for (i in n - 1 downTo 0) {
            while (stack.isNotEmpty() && scores[stack.last()] <= scores[i]) {
                stack.removeLast()
            }
            if (stack.isNotEmpty()) {
                nextGreater[i] = stack.last()
            }
            stack.addLast(i)

            val left = i - prevGreaterOrEqual[i]
            val right = nextGreater[i] - i
            contributions[i] = 1L * left * right
        }
        val pq = PriorityQueue<Int>(compareBy({ -nums[it] }, { -contributions[it] }))
        for (i in 0 until n) {
            pq.offer(i)
        }
        var result = 1L
        var remaining = 1L * k
        while (pq.isNotEmpty()) {
            if (remaining <= 0) break
            val index = pq.poll()!!
            result =
                result * modPow(nums[index], minOf(1L * remaining, contributions[index])) % MODULO
            remaining -= contributions[index]
        }

        return result.toInt()
    }

    fun maxActiveSectionsAfterTrade(s: String, queries: Array<IntArray>): List<Int> { // todo TLE
        val sorted = queries.withIndex().sortedWith(compareBy({ it.value[0] }, { it.value[1] }))
        val result = MutableList(queries.size) { 0 }
        val groups = ArrayDeque<Int>()
        var totalOnes = 0
        var start = -1
        var end = -1
        for (i in s.indices) {
            if (s[i] == '1') totalOnes++
        }
        var endFlag = true
        for ((originalIndex, query) in sorted) {
            if (query[0] > end) {
                groups.clear()
                for (i in query[0]..query[1]) {
                    if (s[i] == '1') {
                        endFlag = true
                    } else {
                        if (endFlag) {
                            groups.add(1)
                        } else {
                            groups[groups.size - 1]++
                        }
                        endFlag = false
                    }
                }
            } else {
                for (i in start until query[0]) {
                    if (s[i] == '0') {
                        if (--groups[0] == 0) {
                            groups.removeFirst()
                        }
                    }
                }
                if (query[1] < end) {
                    for (i in end downTo query[1] + 1) {
                        if (s[i] == '0') {
                            if (--groups[groups.size - 1] == 0) {
                                groups.removeLast()
                                endFlag = true
                            }
                        } else if (i - 1 >= 0 && s[i - 1] == '0') {
                            endFlag = false
                        }
                    }
                } else if (query[1] > end) {
                    for (i in end + 1..query[1]) {
                        if (s[i] == '1') {
                            endFlag = true
                        } else {
                            if (endFlag) {
                                groups.add(1)
                            } else {
                                groups[groups.size - 1]++
                            }
                            endFlag = false
                        }
                    }
                }
            }

            start = query[0]
            end = query[1]

            var removed = 0
            for (i in 0 until groups.size - 1) {
                removed = maxOf(removed, groups[i] + groups[i + 1])
            }
            result[originalIndex] = totalOnes + removed
        }
        return result
    }

    fun putMarbles(weights: IntArray, k: Int): Long {
        val n = weights.size
        val target = minOf(k - 1, n - k)
        if (target == 0) return 0L
        val pqMax = PriorityQueue<Long>()
        var maxSum = 0L
        val pqMin = PriorityQueue<Long>(reverseOrder())
        var minSum = 0L
        for (i in 0 until n - 1) {
            val sum = weights[i].toLong() + weights[i + 1]
            if (pqMax.size < target) {
                pqMax.offer(sum)
                maxSum += sum
            } else if (pqMax.peek()!! < sum) {
                val polled = pqMax.poll()!!
                maxSum -= polled
                pqMax.offer(sum)
                maxSum += sum

                if (pqMin.size < target) {
                    pqMin.offer(polled)
                    minSum += polled
                } else if (pqMin.peek()!! > polled) {
                    minSum -= pqMin.poll()!!
                    pqMin.offer(polled)
                    minSum += polled
                }
            } else if (pqMin.size < target) {
                pqMin.offer(sum)
                minSum += sum
            } else if (pqMin.peek()!! > sum) {
                minSum -= pqMin.poll()!!
                pqMin.offer(sum)
                minSum += sum
            }
        }
        return maxSum - minSum
    }

    fun countGoodTriplets(nums1: IntArray, nums2: IntArray): Long {
        val n = nums1.size
        // 构造 pos 数组：记录每个数字在 nums2 中的位置（转换为 1-indexed 方便 BIT 使用）
        val pos2 = IntArray(n)
        for (i in 0 until n) {
            pos2[nums2[i]] = i + 1  // 1-indexed
        }

        // 构建转换数组 a：a[i] = 在 nums2 中 nums1[i] 的位置
        val a = IntArray(n) { i -> pos2[nums1[i]] }

        // 计算左侧比 a[j] 小的个数 L[j]
        val leftCount = IntArray(n)
        val leftTree = ArrayCode.FenwickTree(n)
        for (j in 0 until n) {
            // query 返回 [1, a[j]-1] 中元素个数
            leftCount[j] = leftTree.query(a[j] - 1)
            leftTree.update(a[j], 1)
        }

        // 计算右侧比 a[j] 大的个数 R[j]
        val rightCount = IntArray(n)
        val rightTree = ArrayCode.FenwickTree(n)
        for (j in n - 1 downTo 0) {
            // query 返回 [a[j]+1, n] 中的个数
            rightCount[j] = rightTree.query(n) - rightTree.query(a[j])
            rightTree.update(a[j], 1)
        }

        // 最终答案为所有 j 的 leftCount[j] * rightCount[j] 累加
        var ans = 0L
        for (j in 0 until n) {
            ans += leftCount[j].toLong() * rightCount[j]
        }
        return ans
    }

    fun countIdealArrays(n: Int, maxValue: Int): Int {
        val MAX_CHAIN_LENGTH = 15
        val maxDim = maxOf(n, maxValue)

        // comb[k][v] = C(v-1, k-1) mod MOD
        val comb = Array(MAX_CHAIN_LENGTH) { LongArray(maxDim + 1) }
        // combPrefix[k][v] = sum_{i=1..v} comb[k][i]
        val combPrefix = Array(MAX_CHAIN_LENGTH) { LongArray(maxDim + 1) }
        // chainCountByLength[k] = 枚举所有 "值链" 后，长度恰为 k 的链总数
        val chainCountByLength = LongArray(MAX_CHAIN_LENGTH)

        // 初始化：comb[1][v] = C(v-1,0) = 1, 前缀和为 v
        for (v in 1..maxDim) {
            comb[1][v] = 1L
            combPrefix[1][v] = v.toLong()
        }
        // 预处理其他组合数及前缀和
        for (length in 2 until MAX_CHAIN_LENGTH) {
            for (value in length..maxDim) {
                comb[length][value] = combPrefix[length - 1][value - 1]
                combPrefix[length][value] =
                    (comb[length][value] + combPrefix[length][value - 1]) % MODULO
            }
        }

        // 嵌套函数：DFS 枚举以 lastValue 结尾的所有 "值链"
        fun dfs(lastValue: Int, currentLength: Int) {
            chainCountByLength[currentLength]++
            var nextVal = lastValue * 2
            while (nextVal <= maxValue) {
                dfs(nextVal, currentLength + 1)
                nextVal += lastValue
            }
        }

        // 从每个起点开始 DFS
        for (start in 1..maxValue) {
            dfs(start, 1)
        }

        // 累加结果：每条长度为 k 的链，有 C(n-1, k-1) 种插空方式
        var result = 0L
        for (length in 1 until MAX_CHAIN_LENGTH) {
            result = (result + chainCountByLength[length] * comb[length][n] % MODULO) % MODULO
        }

        return result.toInt()
    }

    fun countSubarraysWithFixedBounds(nums: IntArray, minK: Int, maxK: Int): Long {
        var bad = -1
        var left = -1
        var right = -1
        var count = 0L
        for (i in nums.indices) {
            if (nums[i] !in minK..maxK) {
                bad = i
            }
            if (nums[i] == minK) {
                left = i
            }
            if (nums[i] == maxK) {
                right = i
            }
            count += maxOf(0, minOf(left, right) - bad)
        }
        return count
    }

    fun countCells(grid: Array<CharArray>, pattern: String): Int {
        val m = grid.size
        val n = grid[0].size
        val p = pattern.length
        val S = m * n

        val hori = String(CharArray(S) { i -> grid[i / n][i % n] })
        val vert = String(CharArray(S) { i -> grid[i % m][i / m] })

        fun buildLPS(): IntArray {
            val lps = IntArray(p)
            var len = 0
            var i = 1
            while (i < p) {
                if (pattern[i] == pattern[len]) {
                    lps[i++] = ++len
                } else if (len > 0) {
                    len = lps[len - 1]
                } else {
                    lps[i++] = 0
                }
            }
            return lps
        }

        fun kmpSearch(text: String, lps: IntArray): List<Int> {
            val res = mutableListOf<Int>()
            var i = 0
            var j = 0
            while (i < text.length) {
                if (text[i] == pattern[j]) {
                    i++; j++
                    if (j == p) {
                        res += (i - j)
                        j = lps[j - 1]
                    }
                } else if (j > 0) {
                    j = lps[j - 1]
                } else {
                    i++
                }
            }
            return res
        }

        val lps = buildLPS()
        val hMatches = kmpSearch(hori, lps)
        val vMatches = kmpSearch(vert, lps)

        val hDiff = IntArray(S + 1)
        val vDiff = IntArray(S + 1)
        for (s in hMatches) {
            hDiff[s]++
            hDiff[s + p]--
        }
        for (s in vMatches) {
            vDiff[s]++
            vDiff[s + p]--
        }
        val hCov = IntArray(S)
        val vCov = IntArray(S)
        run {
            var acc = 0
            for (i in 0 until S) {
                acc += hDiff[i]
                hCov[i] = acc
            }
        }
        run {
            var acc = 0
            for (i in 0 until S) {
                acc += vDiff[i]
                vCov[i] = acc
            }
        }

        var count = 0
        for (r in 0 until m) {
            for (c in 0 until n) {
                val hi = r * n + c
                val vi = c * m + r
                if (hCov[hi] > 0 && vCov[vi] > 0) count++
            }
        }
        return count
    }

    fun maxTaskAssign(tasks: IntArray, workers: IntArray, pills: Int, strength: Int): Int {
        tasks.sort()
        workers.sort()

//        fun canFinish(k: Int): Boolean {
//            var costPills = 0
//            val map = TreeMap<Int, Int>()
//            for (worker in workers) {
//                map[worker] = map.getOrDefault(worker, 0) + 1
//            }
//            for (taskIndex in k - 1 downTo 0) {
//                var key = map.ceilingKey(tasks[taskIndex])
//                if (key != null) {
//                    map[key] = map[key]!! - 1
//                    if (map[key] == 0) map.remove(key)
//                    continue
//                }
//                if (costPills == pills) return false
//                key = map.ceilingKey(tasks[taskIndex] - strength)
//                if (key != null) {
//                    map[key] = map[key]!! - 1
//                    if (map[key] == 0) map.remove(key)
//                    costPills++
//                } else {
//                    return false
//                }
//            }
//            return true
//        }

//        fun canFinish(k: Int): Boolean {
//            val workerList = workers.toMutableList()
//            var costPills = 0
//            for (taskIndex in k - 1 downTo 0) {
//                if (workerList.size == 0) return false
//                if (workerList.last() >= tasks[taskIndex]) {
//                    workerList.removeLast()
//                } else if (costPills == pills || workerList.last() + strength < tasks[taskIndex]) {
//                    return false
//                } else {
//                    var left = 0
//                    var right = workerList.size - 1
//                    while (left <= right) {
//                        val mid = left + ((right - left) shr 1)
//                        if (workerList[mid] + strength >= tasks[taskIndex]) {
//                            left = mid + 1
//                        } else {
//                            right = mid - 1
//                        }
//                    }
//                    workerList.removeAt(right)
//                    if (++costPills > pills) return false
//                }
//            }
//            return true
//        }

        fun canFinish(k: Int): Boolean {
            val taskList = ArrayDeque<Int>()
            var taskIndex = 0
            var costPills = 0
            for (workerIndex in workers.size - k until workers.size) {
                val worker = workers[workerIndex]
                while (taskIndex < k && worker + strength >= tasks[taskIndex]) {
                    taskList.add(tasks[taskIndex++])
                }
                if (taskList.isEmpty()) return false
                if (taskList[0] <= worker) {
                    taskList.removeFirst()
                } else {
                    if (++costPills > pills) return false
                    taskList.removeLast()
                }
            }
            return true
        }

        var left = 0
        var right = tasks.size
        while (left <= right) {
            val mid = left + ((right - left) shr 1)
            if (canFinish(mid)) {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return right
    }

    fun colorTheGrid(m: Int, n: Int): Int {
        // 1. 计算 3^m
        var maxMask = 1
        repeat(m) { maxMask *= 3 }

        // 2. 生成所有列内合法的状态（上下不相同）
        val states = mutableListOf<Int>()
        for (mask in 0 until maxMask) {
            var tmp = mask
            var prevColor = -1
            var valid = true
            for (i in 0 until m) {
                val color = tmp % 3
                tmp /= 3
                if (color == prevColor) {
                    valid = false
                    break
                }
                prevColor = color
            }
            if (valid) states.add(mask)
        }
        val S = states.size

        // 3. 预处理列间兼容性：同一行颜色不能相同
        val compatible = Array(S) { BooleanArray(S) }
        for (i in 0 until S) {
            for (j in 0 until S) {
                var a = states[i]
                var b = states[j]
                var okPair = true
                for (k in 0 until m) {
                    if (a % 3 == b % 3) {
                        okPair = false
                        break
                    }
                    a /= 3
                    b /= 3
                }
                compatible[i][j] = okPair
            }
        }

        // 4. 列 DP
        var previous = LongArray(S) { 1L } // 第一列每个状态都只有 1 种方式
        repeat(n - 1) {
            val current = LongArray(S)
            for (j in 0 until S) {
                for (i in 0 until S) {
                    if (compatible[i][j]) {
                        current[j] = (current[j] + previous[i]) % MODULO
                    }
                }
            }
            previous = current
        }

        // 5. 汇总结果
        return (previous.sum() % MODULO).toInt()
    }

    fun largestPathValue(colors: String, edges: Array<IntArray>): Int {
        val n = colors.length
        val inDegree = IntArray(n)
        val graph = Array(n) { mutableListOf<Int>() }
        for ((u, v) in edges) {
            graph[u].add(v)
            inDegree[v]++
        }
        var processed = 0
        val dp = Array(n) { IntArray(26) } // arrived at i, color j's max count
        val queue = LinkedList<Int>()
        for (i in 0 until n) {
            if (inDegree[i] == 0) {
                queue.offer(i)
                dp[i][colors[i] - 'a'] = 1
            }
        }
        var result = 1
        while (queue.isNotEmpty()) {
            val node = queue.poll()!!
            processed++
            for (next in graph[node]) {
                val nextColor = colors[next] - 'a'
                for (color in 0 until 26) {
                    dp[next][color] =
                        maxOf(dp[next][color], dp[node][color] + if (color == nextColor) 1 else 0)
                    result = maxOf(result, dp[next][color])
                }
                if (--inDegree[next] == 0) {
                    queue.offer(next)
                }
            }
        }
        return if (processed == n) result else -1
    }


    fun possibleStringCount(word: String, k: Int): Int {
        val n = word.length
        if (n == 0) return 0
        val groups = mutableListOf<Int>()
        var count = 1
        for (i in 0 until n) {
            if (i > 0) {
                if (word[i] == word[i - 1]) {
                    count++
                } else {
                    groups.add(count)
                    count = 1
                }
            }
        }
        groups.add(count)

        var total = 1L
        for (i in groups) {
            total = (total * i) % MODULO
        }
        if (k <= groups.size) {
            return (total % MODULO).toInt()
        }

        var prev = IntArray(k)
        prev[0] = 1
        for (i in groups) {
            val current = IntArray(k)
            var sum = 0L
            for (j in 0 until k) {
                if (j > 0) {
                    sum = (sum + prev[j - 1]) % MODULO
                }
                if (j > i) {
                    sum = (sum - prev[j - i - 1] + MODULO) % MODULO
                }
                current[j] = sum.toInt()
            }
            prev = current
        }

        var invalid = 0L
        for (i in groups.size until k) {
            invalid = (invalid + prev[i]) % MODULO
        }
        return ((total - invalid + MODULO) % MODULO).toInt()
    }

    fun maxEventValue(events: Array<IntArray>, k: Int): Int {
        events.sortBy { it[0] }
        val n = events.size
        val memo = Array(n) { IntArray(k + 1) { -1 } }

        fun findStartAfter(time: Int): Int {
            var left = 0
            var right = n - 1
            while (left <= right) {
                val mid = left + ((right - left) shr 1)
                if (events[mid][0] > time) {
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            }
            return left
        }

//        val dp = Array(n + 1) { IntArray(k + 1) { 0 } }
//        val next = IntArray(n) {
//            findStartAfter(events[it][1])
//        }
//        for (i in n - 1 downTo 0) {
//            for (j in 1..k) {
//                dp[i][j] = maxOf(dp[i + 1][j], events[i][2] + dp[next[i]][j - 1])
//            }
//        }
//
//        return dp[0][k]

        fun solve(index: Int, remain: Int): Int {
            if (index >= n || remain == 0) {
                return 0
            }
            if (memo[index][remain] != -1) {
                return memo[index][remain]
            }
            val notTake = solve(index + 1, remain)
            val nextIndex = findStartAfter(events[index][1])
            val take = events[index][2] + solve(nextIndex, remain - 1)
            memo[index][remain] = maxOf(notTake, take)
            return memo[index][remain]
        }

        return solve(0, k)
    }

    fun earliestAndLatest(n: Int, firstPlayer: Int, secondPlayer: Int): IntArray {
        // brute-force not optimised
        val firstBit = 1 shl (n - firstPlayer)
        val secondBit = 1 shl (n - secondPlayer)
        var minMeet = Int.MAX_VALUE
        var maxMeet = 0

        fun processState(state: Int, l: Int, r: Int, next: MutableSet<Int>): Boolean {
            if (state.countOneBits() < 2) {
                return false
            }
            var left = l
            var right = r
            while (left > right && (left and state == 0 || right and state == 0)) {
                if (left and state == 0) {
                    left = left shr 1
                }
                if (right and state == 0) {
                    right = right shl 1
                }
            }
            if (left <= right) {
                next.add(state)
                return false
            }
            if (left == firstBit && right == secondBit) {
                return true
            }
            return processState(state xor left, left shr 1, right shl 1, next) ||
                    processState(state xor right, left shr 1, right shl 1, next)
        }

        var round = 0
        var queue = mutableSetOf<Int>()
        var next = mutableSetOf<Int>()
        queue.add((1 shl n) - 1)
        while (queue.isNotEmpty()) {
            round++
            next.clear()
            for (state in queue) {
                val found = processState(state, 1 shl (n - 1), 1, next)
                if (found) {
                    minMeet = minOf(minMeet, round)
                    maxMeet = maxOf(maxMeet, round)
                }
            }
            val tmp = queue
            queue = next
            next = tmp
        }
        return intArrayOf(minMeet, maxMeet)
    }
}