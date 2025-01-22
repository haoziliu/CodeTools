package com.example.codetools

import java.util.LinkedList
import java.util.PriorityQueue

object MatrixCode {
    fun imageSmoother(img: Array<IntArray>): Array<IntArray> {
        val m = img.size
        val n = img[0].size
        val result = Array(m) { IntArray(n) }
        var sum = 0
        var count = 0
        for (i in 0 until m) {
            for (j in 0 until n) {
                sum = 0
                count = 0
                for (x in i - 1..i + 1) {
                    for (y in j - 1..j + 1) {
                        if (x in 0 until m && y in 0 until n) {
                            sum += img[x][y]
                            count++
                        }
                    }
                }
                result[i][j] = sum / count
            }
        }
        return result
    }

    fun wordExist(board: Array<CharArray>, word: String): Boolean {
        if (board.isEmpty()) {
            return false
        }

        val rows = board.size
        val cols = board[0].size

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                if (searchWord(board, i, j, word)) {
                    return true
                }
            }
        }

        return false
    }

    fun searchWord(board: Array<CharArray>, i: Int, j: Int, word: String): Boolean {
        val DIRECTIONS =
            arrayOf(intArrayOf(1, 0), intArrayOf(0, 1), intArrayOf(-1, 0), intArrayOf(0, -1))

        if (word.isEmpty()) {
            return true
        }
        // cell index invalid or cell value not match
        if (i < 0 || i >= board.size || j < 0 || j >= board[0].size) {
            return false
        }
        val char = board[i][j]
        if (char != word[0]) {
            return false
        }
        board[i][j] = '#' // Mark cell as visited

        val rest = word.substring(1)
        // Explore neighboring cells recursively
        val found = DIRECTIONS.any { (dx, dy) ->
            searchWord(board, i + dx, j + dy, rest)
        }

        board[i][j] = char // Restore cell

        return found
    }

    fun minPathSum(grid: Array<IntArray>): Int {
        val m = grid.size
        val n = grid[0].size
        val dp = Array(m) { IntArray(n) }
        // Initialize the first cell of dp array
        dp[0][0] = grid[0][0]
        // Initialize the first column of dp array
        for (i in 1 until m) {
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        }
        // Initialize the first row of dp array
        for (j in 1 until n) {
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        }
        // Calculate minimum path sum for each cell
        for (i in 1 until m) {
            for (j in 1 until n) {
                // check from which direction the minimum path sum is coming
                dp[i][j] = grid[i][j] + minOf(dp[i - 1][j], dp[i][j - 1])
            }
        }
        return dp[m - 1][n - 1]
    }

    // dfs, time limit exceeded
//    fun minPathSum(grid: Array<IntArray>): Int {
//        val m = grid.size
//        val n = grid[0].size
//        val minSum = intArrayOf(Int.MAX_VALUE)
//        pathSum(grid, 0, 0, m, n, 0, minSum)
//        return minSum[0]
//    }
//
//    fun pathSum(grid: Array<IntArray>, i: Int, j: Int, m: Int, n: Int, currentSum: Int, minSum: IntArray) {
//        val newSum = currentSum + grid[i][j]
//        if (i == m - 1 && j == n - 1) {
//            minSum[0] = min(newSum, minSum[0])
//            return
//        }
//        if (newSum < minSum[0]) {
//            if (i != m - 1) {
//                pathSum(grid, i + 1, j, m, n, newSum, minSum)
//            }
//            if (j != n - 1) {
//                pathSum(grid, i, j + 1, m, n, newSum, minSum)
//            }
//        }
//    }

    fun islandPerimeter(grid: Array<IntArray>): Int {
        val row = grid.size
        val col = grid[0].size
        var result = 0
        for (i in 0 until row) {
            for (j in 0 until col) {
                if (grid[i][j] == 1) {
//                    cellAboveIsCounted = i > 0 && grid[i - 1][j] == 1
//                    cellLeftIsCounted = j > 0 && grid[i][j - 1] == 1
//                    if (!(cellAboveIsCounted && cellLeftIsCounted)) {
//                        result += if (cellAboveIsCounted || cellLeftIsCounted) 2 else 4
//                    }
                    if (i > 0 && grid[i - 1][j] == 1 && j > 0 && grid[i][j - 1] == 1) {

                    } else if (i > 0 && grid[i - 1][j] == 1 || j > 0 && grid[i][j - 1] == 1) {
                        result += 2

                    } else {
                        result += 4
                    }
                }
            }
        }
        return result
    }

    fun numIslands(grid: Array<CharArray>): Int {

        fun dfs(i: Int, j: Int) {
            if (i < 0 || i >= grid.size || j < 0 || j >= grid[0].size || grid[i][j] == '0') {
                return
            }
            grid[i][j] = '0' // Mark the visited land
            dfs(i + 1, j) // down
            dfs(i - 1, j) // up
            dfs(i, j + 1) // right
            dfs(i, j - 1) // left
        }

        var result = 0
        for (i in grid.indices) {
            for (j in grid[0].indices) {
                if (grid[i][j] == '1') {
                    dfs(i, j)
                    result++
                }
            }
        }
        return result
    }

    fun findFarmland(land: Array<IntArray>): Array<IntArray> {
//        fun dfs(i: Int, j: Int, max: IntArray, isHorizontal: Boolean) {
//            if (i < 0 || i >= land.size || j < 0 || j >= land[0].size || land[i][j] == 0) {
//                return
//            }
//            land[i][j] = 0 // Mark the visited land
//            if (isHorizontal) {
//                max[1] = j
//            } else {
//                max[0] = i
//            }
//            dfs(i, j + 1, max, true)
//            dfs(i + 1, j, max, false)
//        }
//
//        val result = mutableListOf<IntArray>()
//        val max = IntArray(2) { 0 }
//        for (i in land.indices) {
//            for (j in land[0].indices) {
//                if (land[i][j] == 1) {
//                    max[0] = i
//                    max[1] = j
//                    dfs(i, j, max, true)
//                    result.add(intArrayOf(i, j, max[0], max[1]))
//                }
//            }
//        }
//        return result.toTypedArray()

        val result = mutableListOf<IntArray>()
        for (i in land.indices) {
            for (j in land[0].indices) {
                if (land[i][j] == 1) {
                    var maxJ = j + 1
                    while (land[i].getOrNull(maxJ) == 1) {
                        land[i][maxJ++] = 0
                    }
                    var maxI = i + 1
                    while (land.getOrNull(maxI)?.get(j) == 1) {
                        land[maxI++][j] = 0
                    }
                    for (i1 in i + 1 until maxI) {
                        for (j1 in j + 1 until maxJ) {
                            land[i1][j1] = 0
                        }
                    }
                    result.add(intArrayOf(i, j, maxI - 1, maxJ - 1))
                }
            }
        }
        return result.toTypedArray()
    }

    fun largestLocal(grid: Array<IntArray>): Array<IntArray> {
        val size = grid.size - 2
        val result = Array(size) { IntArray(size) }
        for (i in 0 until size) {
            for (j in 0 until size) {
                result[i][j] = maxOf(
                    grid[i][j], grid[i][j + 1], grid[i][j + 2],
                    grid[i + 1][j], grid[i + 1][j + 1], grid[i + 1][j + 2],
                    grid[i + 2][j], grid[i + 2][j + 1], grid[i + 2][j + 2]
                )
            }
        }
        return result
    }

    fun matrixScoreAfterFlipping(grid: Array<IntArray>): Int {
        val m = grid.size
        val n = grid[0].size
        grid.forEach { row ->
            if (row[0] == 0) {
                for (i in row.indices) {
                    row[i] = row[i] xor 1
                }
            }
        }
        var sum = 0
        var sumColumn = 0
        for (columnIndex in 0 until n) {
            for (rowIndex in 0 until m) {
                sumColumn += grid[rowIndex][columnIndex]
            }
            sum += if (sumColumn < m / 2.0) {
                (m - sumColumn) shl (n - 1 - columnIndex)
            } else {
                sumColumn shl (n - 1 - columnIndex)
            }
            sumColumn = 0
        }
        return sum
    }

    fun getMaximumGold(grid: Array<IntArray>): Int {
        if (grid.isEmpty()) {
            return 0
        }

        fun findMax(i: Int, j: Int, sum: Int): Int {
            // cell index invalid or cell value not match
            if (i < 0 || i >= grid.size || j < 0 || j >= grid[0].size || grid[i][j] == 0) {
                return sum
            }
            val newSum = sum + grid[i][j]
            val temp = grid[i][j]
            grid[i][j] = 0 // Mark cell as visited
            // Explore neighboring cells recursively
            val value = maxOf(
                findMax(i + 1, j, newSum),
                findMax(i - 1, j, newSum),
                findMax(i, j + 1, newSum),
                findMax(i, j - 1, newSum)
            )
            grid[i][j] = temp // Restore cell
            return value
        }

        var result = 0
        for (i in grid.indices) {
            for (j in grid[0].indices) {
                result = maxOf(result, findMax(i, j, 0))
            }
        }
        return result
    }

    fun maximumSafenessFactor(grid: List<List<Int>>): Int {
        val n = grid.size
        if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1) return 0

        val directions = listOf(Pair(1, 0), Pair(0, 1), Pair(-1, 0), Pair(0, -1))
        val distanceMap = Array(n) { IntArray(n) { Int.MAX_VALUE } }
        val queue = LinkedList<Pair<Int, Int>>()
        // Multi-source BFS to calculate distances to nearest thief
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (grid[i][j] == 1) {
                    distanceMap[i][j] = 0
                    queue.add(Pair(i, j))
                }
            }
        }
        while (queue.isNotEmpty()) {
            val (x, y) = queue.poll()
            for (dir in directions) {
                val nx = x + dir.first
                val ny = y + dir.second
                if (nx in 0 until n && ny in 0 until n && distanceMap[nx][ny] == Int.MAX_VALUE) {
                    distanceMap[nx][ny] = distanceMap[x][y] + 1
                    queue.add(Pair(nx, ny))
                }
            }
        }

        val pq: PriorityQueue<IntArray> = PriorityQueue { a, b -> b[0] - a[0] }
        val dp = Array(n) { IntArray(n) { -1 } }
        pq.offer(intArrayOf(distanceMap[0][0], 0, 0))
        dp[0][0] = distanceMap[0][0]

        while (pq.isNotEmpty()) {
            val (safeFactor, x, y) = pq.poll()
            if (x == n - 1 && y == n - 1) {
                return safeFactor
            }
            for (dir in directions) {
                val nx = x + dir.first
                val ny = y + dir.second
                if (nx in 0 until n && ny in 0 until n) {
                    val newFactor = minOf(safeFactor, distanceMap[nx][ny])
                    if (newFactor > dp[nx][ny]) {
                        dp[nx][ny] = newFactor
                        pq.offer(intArrayOf(newFactor, nx, ny))
                    }
                }
            }
        }

        return -1
    }

    fun uniquePathsWithObstacles(obstacleGrid: Array<IntArray>): Int {
        val m = obstacleGrid.size
        val n = obstacleGrid[0].size
        if (obstacleGrid[0][0] == 1 || obstacleGrid[m - 1][n - 1] == 1) return 0

//        val dp = Array(m) { IntArray(n) }
//        dp[0][0] = 1
//        for (i in 1 until m) {
//            dp[i][0] = if (obstacleGrid[i][0] == 1) 0 else dp[i - 1][0]
//        }
//        for (i in 1 until n) {
//            dp[0][i] = if (obstacleGrid[0][i] == 1) 0 else dp[0][i - 1]
//        }
//        for (i in 1 until m) {
//            for (j in 1 until n) {
//                dp[i][j] = if (obstacleGrid[i][j] == 1) 0 else dp[i - 1][j] + dp[i][j - 1]
//            }
//        }
//        return dp[m - 1][n - 1]

        // 1D memorization can resolve
        val dp = IntArray(n)
        dp[0] = 1
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0
                } else if (j > 0) {
                    dp[j] += dp[j - 1]
                }
            }
        }
        return dp[n - 1]
    }


    fun replaceSurrounded(board: Array<CharArray>): Unit {
        val m = board.size
        val n = board[0].size

        fun conserveEdgeDfs(i: Int, j: Int) {
            if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != 'O') {
                return
            }
            board[i][j] = 'S'
            conserveEdgeDfs(i + 1, j)
            conserveEdgeDfs(i, j + 1)
            conserveEdgeDfs(i - 1, j)
            conserveEdgeDfs(i, j - 1)
        }

        for (i in 0 until m) {
            for (j in 0 until n) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    if (board[i][j] == 'O') {
                        conserveEdgeDfs(i, j)
                    }
                }
            }
        }

        // Replace inner 'O's with 'X'
        for (i in 1 until m - 1) {
            for (j in 1 until n - 1) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X'
                }
            }
        }

        // replace back "S"
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (board[i][j] == 'S') {
                    board[i][j] = 'O'
                }
            }
        }
    }

    fun luckyNumbers(matrix: Array<IntArray>): List<Int> {
//        val m = matrix.size
//        val n = matrix[0].size
//        val minPos = IntArray(m)
//        val maxPos = IntArray(n)
//        for (i in 0 until m) {
//            var rowMin = Int.MAX_VALUE
//            for (j in 0 until n) {
//                if (matrix[i][j] < rowMin) {
//                    rowMin = matrix[i][j]
//                    minPos[i] = j
//                }
//            }
//        }
//        for (j in 0 until n) {
//            var colMax = Int.MIN_VALUE
//            for (i in 0 until m) {
//                if (matrix[i][j] > colMax) {
//                    colMax = matrix[i][j]
//                    maxPos[j] = i
//                }
//            }
//        }
//        val result = mutableListOf<Int>()
//        for (i in 0 until m) {
//            for (j in 0 until n) {
//                if (minPos[i] == j && maxPos[j] == i) {
//                    result.add(matrix[i][j])
//                }
//            }
//        }
//        return result

        val minRow = IntArray(matrix.size) { Int.MAX_VALUE }
        val maxCol = IntArray(matrix[0].size) { 0 }
        for (i in matrix.indices) {
            for (j in matrix[0].indices) {
                minRow[i] = minOf(matrix[i][j], minRow[i])
                maxCol[j] = maxOf(matrix[i][j], maxCol[j])
            }
        }
        val result = mutableListOf<Int>()
        for (i in matrix.indices) {
            for (j in matrix[0].indices) {
                if (matrix[i][j] == minRow[i] && matrix[i][j] == maxCol[j]) {
                    result.add(matrix[i][j])
                }
            }
        }
        return result
    }

    fun isValidSudoku(board: Array<CharArray>): Boolean {
        // Only the filled cells need to be validated
//        val columnSet = Array(9) { mutableSetOf<Int>() }
//        val boxSet = Array(9) { mutableSetOf<Int>() }
//
//        for (i in 0..8) {
//            val rowSet = mutableSetOf<Int>()
//            for (j in 0..8) {
//                if (board[i][j] == '.') continue
//                val num = board[i][j].digitToInt()
//                if (rowSet.contains(num)) {
//                    return false
//                } else {
//                    rowSet.add(num)
//                }
//
//                if (columnSet[j].contains(num)) {
//                    return false
//                } else {
//                    columnSet[j].add(num)
//                }
//
//                val boxIndex = i / 3 * 3 + j / 3
//                if (boxSet[boxIndex].contains(num)) {
//                    return false
//                } else {
//                    boxSet[boxIndex].add(num)
//                }
//            }
//        }
//        return true

        val rows = Array(9) { BooleanArray(9) }
        val cols = Array(9) { BooleanArray(9) }
        val boxes = Array(9) { BooleanArray(9) }
        for (rowIndex in 0 until 9) {
            for (colIndex in 0 until 9) {
                val char = board[rowIndex][colIndex]
                if (char == '.') continue
                val num = char - '1'
                val boxIndex = rowIndex / 3 * 3 + colIndex / 3
                if (rows[rowIndex][num] || cols[colIndex][num] || boxes[boxIndex][num]) return false
                rows[rowIndex][num] = true
                cols[colIndex][num] = true
                boxes[boxIndex][num] = true
            }
        }
        return true
    }

    fun rotateClockwise(matrix: Array<IntArray>): Unit {
        val n = matrix.size
        for (layer in 0 until n / 2) {
            val first = layer
            val last = n - 1 - layer
            for (i in first until last) {
                val offset = i - first
                // 存储上边
                val tmp = matrix[first][i]
                // 左边移到上边
                matrix[first][i] = matrix[last - offset][first]
                // 下边移到左边
                matrix[last - offset][first] = matrix[last][last - offset]
                // 右边移到下边
                matrix[last][last - offset] = matrix[i][last]
                // 上边移到右边
                matrix[i][last] = tmp
            }
        }
    }

    fun restoreMatrix(rowSum: IntArray, colSum: IntArray): Array<IntArray> {
        val m = rowSum.size
        val n = colSum.size
        val matrix = Array(m) { IntArray(n) }
        for (i in 0 until m) {
            for (j in 0 until n) {
                val value = minOf(rowSum[i], colSum[j])
                matrix[i][j] = value
                rowSum[i] -= value
                colSum[j] -= value
            }
        }
        return matrix
    }

    fun spiralMatrixIII(rows: Int, cols: Int, rStart: Int, cStart: Int): Array<IntArray> {
        val total = rows * cols
        val result = Array(total) { IntArray(2) }
        var currentR = rStart
        var currentC = cStart
        result[0][0] = rStart
        result[0][1] = cStart
        var index = 1
        var step = 0
        var direction = 0
        val directions =
            arrayOf(intArrayOf(0, 1), intArrayOf(1, 0), intArrayOf(0, -1), intArrayOf(-1, 0))
        var shouldAddStep = true
        while (index < total) {
            if (shouldAddStep) step++
            shouldAddStep = !shouldAddStep
            for (i in 1..step) {
                currentR += directions[direction][0]
                currentC += directions[direction][1]
                if (currentR in 0 until rows && currentC in 0 until cols) {
                    result[index][0] = currentR
                    result[index][1] = currentC
                    index++
                }
            }
            direction = (direction + 1) % 4
        }
        return result
    }

    fun spiralOrder(matrix: Array<IntArray>): List<Int> {
        var rows = matrix.size
        var cols = matrix[0].size
        val total = rows * cols
        val directions =
            arrayOf(intArrayOf(0, 1), intArrayOf(1, 0), intArrayOf(0, -1), intArrayOf(-1, 0))
        var direction = 0
        val result = mutableListOf<Int>()
        var count = 0
        var currentR = 0
        var currentC = -1
        while (count < total) {
            repeat(cols) {
                currentR += directions[direction][0]
                currentC += directions[direction][1]
                result.add(matrix[currentR][currentC])
                count++
            }
            direction = (direction + 1) % 4
            cols--
            rows--
            repeat(rows) {
                currentR += directions[direction][0]
                currentC += directions[direction][1]
                result.add(matrix[currentR][currentC])
                count++
            }
            direction = (direction + 1) % 4
        }
        return result
    }

    fun spiralMatrix(m: Int, n: Int, head: ListCode.ListNode?): Array<IntArray> {
        val directions =
            arrayOf(intArrayOf(0, 1), intArrayOf(1, 0), intArrayOf(0, -1), intArrayOf(-1, 0))
        val matrix = Array(m) { IntArray(n) { -1 } }
        var current = head
        var i = 0
        var j = 0
        var direction = 0
        while (current != null) {
            matrix[i][j] = current.`val`
            current = current.next
            val nextI = i + directions[direction][0]
            val nextJ = j + directions[direction][1]
            if (nextI < 0 || nextI > m - 1 || nextJ < 0 || nextJ > n - 1 || matrix[nextI][nextJ] != -1) {
                direction = (direction + 1) % 4
                i += directions[direction][0]
                j += directions[direction][1]
            } else {
                i = nextI
                j = nextJ
            }
        }
        return matrix
    }

    fun generateSpiralMatrix(n: Int): Array<IntArray> {
        val directions =
            arrayOf(intArrayOf(0, 1), intArrayOf(1, 0), intArrayOf(0, -1), intArrayOf(-1, 0))
        val matrix = Array(n) { IntArray(n) }
        var direction = 0
        var i = 0
        var j = 0
        for (num in 1..n * n) {
            matrix[i][j] = num
            val nextI = i + directions[direction][0]
            val nextJ = j + directions[direction][1]
            if (nextI < 0 || nextI >= n || nextJ < 0 || nextJ >= n || matrix[nextI][nextJ] != 0) {
                direction = (direction + 1) % 4
                i += directions[direction][0]
                j += directions[direction][1]
            } else {
                i = nextI
                j = nextJ
            }
        }
        return matrix
    }

    fun numMagicSquaresInside(grid: Array<IntArray>): Int {
        fun isMagic(nums: IntArray): Boolean {
            return arrayOf(
                intArrayOf(4, 9, 2, 3, 5, 7, 8, 1, 6),
                intArrayOf(8, 3, 4, 1, 5, 9, 6, 7, 2),
                intArrayOf(6, 1, 8, 7, 5, 3, 2, 9, 4),
                intArrayOf(2, 7, 6, 9, 5, 1, 4, 3, 8),
                intArrayOf(2, 9, 4, 7, 5, 3, 6, 1, 8),
                intArrayOf(4, 3, 8, 9, 5, 1, 2, 7, 6),
                intArrayOf(8, 1, 6, 3, 5, 7, 4, 9, 2),
                intArrayOf(6, 7, 2, 1, 5, 9, 8, 3, 4),
            ).any { it.contentEquals(nums) }
        }

        var result = 0
        for (i in 1 until grid.lastIndex) {
            for (j in 1 until grid[0].lastIndex) {
                if (grid[i][j] == 5 && isMagic(
                        intArrayOf(
                            grid[i - 1][j - 1], grid[i - 1][j], grid[i - 1][j + 1],
                            grid[i][j - 1], grid[i][j], grid[i][j + 1],
                            grid[i + 1][j - 1], grid[i + 1][j], grid[i + 1][j + 1]
                        )
                    )
                ) {
                    result++
                }
            }
        }
        return result
    }

    fun regionsBySlashes(grid: Array<String>): Int {
        // '/', '\\', or ' '
        val n = grid.size * 3
        val expanded = Array(n) { IntArray(n) }
        for (i in grid.indices) {
            for (j in grid[0].indices) {
                val x = 3 * i
                val y = 3 * j
                if (grid[i][j] == '/') {
                    expanded[x][y + 2] = 1
                    expanded[x + 1][y + 1] = 1
                    expanded[x + 2][y] = 1
                } else if (grid[i][j] == '\\') {
                    expanded[x][y] = 1
                    expanded[x + 1][y + 1] = 1
                    expanded[x + 2][y + 2] = 1
                }
            }
        }
        fun dfs(i: Int, j: Int) {
            if (i < 0 || i >= n || j < 0 || j >= n || expanded[i][j] == 1) {
                return
            }
            expanded[i][j] = 1
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)
        }

        var result = 0
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (expanded[i][j] == 0) {
                    dfs(i, j)
                    result++
                }
            }
        }
        return result
    }

    fun minDays(grid: Array<IntArray>): Int {
        // to make 0 island or 2 or more islands -> any island can be cut by max. 2
        val rows = grid.size
        val cols = grid[0].size

        fun dfs(matrix: Array<IntArray>, i: Int, j: Int) {
            if (i < 0 || i >= rows || j < 0 || j >= cols || matrix[i][j] == 0) {
                return
            }
            matrix[i][j] = 0
            dfs(matrix, i + 1, j)
            dfs(matrix, i - 1, j)
            dfs(matrix, i, j + 1)
            dfs(matrix, i, j - 1)
        }

        fun countIsland(): Int {
            val tempGrid = Array(rows) { IntArray(cols) }
            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    tempGrid[i][j] = grid[i][j]
                }
            }
            var count = 0
            for (i in 0 until rows) {
                for (j in 0 until cols) {
                    if (tempGrid[i][j] != 0) {
                        dfs(tempGrid, i, j)
                        count++
                    }
                }
            }
            return count
        }

        val initialCount = countIsland()
        if (initialCount != 1) {
            return 0
        }

        for (i in grid.indices) {
            for (j in grid[0].indices) {
                if (grid[i][j] != 0) {
                    grid[i][j] = 0
                    val newCount = countIsland()
                    grid[i][j] = 1
                    if (newCount != 1) {
                        return 1
                    }
                }
            }
        }
        return 2
    }

    fun setZeroes(matrix: Array<IntArray>): Unit {
        val m = matrix.size
        val n = matrix[0].size
//        val rows = hashSetOf<Int>()
//        val cols = hashSetOf<Int>()
//        for (i in 0 until m) {
//            for (j in 0 until n) {
//                if (matrix[i][j] == 0) {
//                    rows.add(i)
//                    cols.add(j)
//                }
//            }
//        }
//        for (i in 0 until m) {
//            for (j in 0 until n) {
//                if (i in rows || j in cols) {
//                    matrix[i][j] = 0
//                }
//            }
//        }
        var firstRowZero = false
        var firstColZero = false
        for (i in 0 until m) {
            for (j in 0 until n) {
                if (matrix[i][j] == 0) {
                    if (i == 0) firstRowZero = true
                    if (j == 0) firstColZero = true
                    matrix[0][j] = 0
                    matrix[i][0] = 0
                }
            }
        }
        for (i in 1 until m) {
            for (j in 1 until n) {
                if (matrix[0][j] == 0 || matrix[i][0] == 0) {
                    matrix[i][j] = 0
                }
            }
        }
        if (firstRowZero) {
            for (j in 0 until n) {
                matrix[0][j] = 0
            }
        }
        if (firstColZero) {
            for (i in 0 until m) {
                matrix[i][0] = 0
            }
        }
    }

    fun findRotation(mat: Array<IntArray>, target: Array<IntArray>): Boolean {
        val n = mat.size
        val rotations = BooleanArray(4) { true }
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (rotations[0] && mat[i][j] != target[j][n - 1 - i]) {
                    rotations[0] = false
                }
                if (rotations[1] && mat[i][j] != target[n - 1 - i][n - 1 - j]) {
                    rotations[1] = false
                }
                if (rotations[2] && mat[i][j] != target[n - 1 - j][i]) {
                    rotations[2] = false
                }
                if (rotations[3] && mat[i][j] != target[i][j]) {
                    rotations[3] = false
                }
            }
        }
        return rotations.any { it }
    }

    fun maximalSquare(matrix: Array<CharArray>): Int {
        val m = matrix.size
        val n = matrix[0].size
        val dp = IntArray(n)
        var prev = 0 // dp[n-1][n-1]
        var maxStreak = 0
        for (i in 0 until m) {
            for (j in 0 until n) {
                val tmp = dp[j] // 暂存当前 dp[j] 值
                if (matrix[i][j] == '1') {
                    if (j == 0) {
                        dp[j] = 1
                    } else {
                        dp[j] = minOf(dp[j - 1], dp[j], prev) + 1
                    }
                    maxStreak = maxOf(maxStreak, dp[j])
                } else {
                    dp[j] = 0
                }
                prev = tmp
            }
        }
        return maxStreak * maxStreak
    }

    fun uniquePaths(m: Int, n: Int): Int {
        var previous = IntArray(n) { 1 }
        var current = IntArray(n)
        var tmp = IntArray(n)
        for (row in 1 until m) {
            current[0] = 1
            for (col in 1 until n) {
                current[col] = current[col - 1] + previous[col]
            }
            tmp = previous
            previous = current
            current = tmp
        }
        return previous[n - 1]
    }

    fun countSquares(matrix: Array<IntArray>): Int {
        val m = matrix.size
        val n = matrix[0].size
        val dp = IntArray(n)
        var count = 0
        var prev = 0 // 用于存储左上角元素的临时变量

        for (row in 0 until m) {
            for (col in 0 until n) {
                val temp = dp[col] // 暂存当前 dp[col] 值
                if (matrix[row][col] == 1) {
                    dp[col] = if (col == 0) 1 else minOf(dp[col - 1], dp[col], prev) + 1
                    count += dp[col]
                } else {
                    dp[col] = 0
                }
                prev = temp // 更新左上角值
            }
        }
        return count
    }

    fun totalNQueens(n: Int): Int {
        var total = 0

//        val taken = Array(n) { BooleanArray(n) }
//        fun dfs(row: Int) {
//            if (row == n) {
//                total++
//                return
//            }
//            for (col in 0 until n) {
//                if (!taken[row][col]) {
//                    taken[row][col] = true
//                    for (delta in 1 until n - row) {
//                        taken[row + delta][col] = true
//                        if (col - delta >= 0) taken[row + delta][col - delta] = true
//                        if (col + delta < n) taken[row + delta][col + delta] = true
//                    }
//                    dfs(row + 1)
//                    taken[row][col] = false
//                    for (delta in 1 until n - row) {
//                        taken[row + delta][col] = false
//                        if (col - delta >= 0) taken[row + delta][col - delta] = false
//                        if (col + delta < n) taken[row + delta][col + delta] = false
//                    }
//                }
//            }
//        }

        val diagN = (n shl 1) - 1
        val cols = BooleanArray(n)
        val ascDiag = BooleanArray(diagN)
        val descDiag = BooleanArray(diagN)

        fun dfs(row: Int) {
            if (row == n) {
                total++
                return
            }
            for (col in 0 until n) {
                val ascIndex = row + col
                val descIndex = row + n - 1 - col
                if (cols[col] || ascDiag[ascIndex] || descDiag[descIndex]) continue

                cols[col] = true
                ascDiag[ascIndex] = true
                descDiag[descIndex] = true
                dfs(row + 1)
                cols[col] = false
                ascDiag[ascIndex] = false
                descDiag[descIndex] = false
            }
        }

        dfs(0)
        return total
    }

    fun solveNQueens(n: Int): List<List<String>> {
        val cols = BooleanArray(n)
        val ascDiagnols = BooleanArray((n shl 1) - 1)
        val descDiagnols = BooleanArray((n shl 1) - 1)
        val result = mutableListOf<List<String>>()
        val matrix = Array(n) { CharArray(n) { '.' } }

        fun fillRow(row: Int) {
            if (row == n) {
                result.add(matrix.map { String(it) })
                return
            }
            for (col in 0 until n) {
                val ascIndex = row + col
                val descIndex = row + n - 1 - col
                if (cols[col] || ascDiagnols[ascIndex] || descDiagnols[descIndex]) continue

                cols[col] = true
                ascDiagnols[ascIndex] = true
                descDiagnols[descIndex] = true
                matrix[row][col] = 'Q'

                fillRow(row + 1)

                cols[col] = false
                ascDiagnols[ascIndex] = false
                descDiagnols[descIndex] = false
                matrix[row][col] = '.'
            }
        }

        fillRow(0)
        return result
    }

    fun gridGame(grid: Array<IntArray>): Long {
        val n = grid[0].size
        var suffix0 = 0L
        for (i in 1 until n) {
            suffix0 += grid[0][i]
        }
        var result = suffix0
        var prefix1 = 0L
        for (i in 1 until n) {
            suffix0 -= grid[0][i]
            prefix1 += grid[1][i - 1]
            result = minOf(result, maxOf(suffix0, prefix1))
        }
        return result
    }

    fun highestPeak(isWater: Array<IntArray>): Array<IntArray> {
        // multi-source BFS
//        val m = isWater.size
//        val n = isWater[0].size
//        val pq = LinkedList<Pair<Int, Int>>()
//        for (r in 0 until m) {
//            for (c in 0 until n) {
//                if (isWater[r][c] == 1) {
//                    isWater[r][c] = 0
//                    pq.offer(r to c)
//                } else {
//                    isWater[r][c] = -1
//                }
//            }
//        }
//        while (pq.isNotEmpty()) {
//            val (r, c) = pq.poll()!!
//            for ((dx, dy) in DIRECTIONS) {
//                val newR = r + dx
//                val newC = c + dy
//                if (newR !in 0 until m || newC !in 0 until n || isWater[newR][newC] != -1) continue
//                isWater[newR][newC] = isWater[r][c] + 1
//                pq.offer(newR to newC)
//            }
//        }
//        return isWater

        // 状态转移无方向性, 无条件最优解, 可用两次dp解决
        val m = isWater.size
        val n = isWater[0].size
        val result = Array(m) { IntArray(n) { 2000 } }
        for (r in 0 until m) {
            for (c in 0 until n) {
                if (isWater[r][c] == 0) {
                    result[r][c] = minOf(
                        result[r][c],
                        if (r > 0) result[r - 1][c] + 1 else Int.MAX_VALUE,
                        if (c > 0) result[r][c - 1] + 1 else Int.MAX_VALUE
                    )
                } else {
                    result[r][c] = 0
                }
            }
        }
        for (r in m - 1 downTo 0) {
            for (c in n - 1 downTo 0) {
                result[r][c] = minOf(
                    result[r][c],
                    if (r < m - 1) result[r + 1][c] + 1 else Int.MAX_VALUE,
                    if (c < n - 1) result[r][c + 1] + 1 else Int.MAX_VALUE
                )
            }
        }
        return result
    }
}