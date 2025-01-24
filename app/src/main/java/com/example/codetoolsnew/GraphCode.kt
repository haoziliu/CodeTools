package com.example.codetools

import java.util.LinkedList
import java.util.PriorityQueue

object GraphCode {

    fun validPath(n: Int, edges: Array<IntArray>, source: Int, destination: Int): Boolean {
//        if (source == destination) return true
//        val edgeMap = mutableMapOf<Int, MutableList<Int>>()
//        edges.forEach { edge ->
//            edgeMap.getOrPut(edge[0]) {
//                MutableList()
//            }.add(edge[1])
//            edgeMap.getOrPut(edge[1]) {
//                MutableList()
//            }.add(edge[0])
//        }
//
//        fun hasDestination(source: Int, destination: Int): Boolean {
//            if (edgeMap[source] == null) return false
//            return if (edgeMap[source]!!.contains(destination)) {
//                true
//            } else {
//                val nextSet = edgeMap[source]!!
//                edgeMap.remove(source)
//                nextSet.any {
//                    edgeMap[it]?.remove(source)
//                    hasDestination(it, destination)
//                }
//            }
//        }
//
//        return hasDestination(source, destination)

        val visited = BooleanArray(n) { false }
        val queue = LinkedList<Int>()
        queue.add(source)
        visited[source] = true
        while (queue.isNotEmpty()) {
            val current = queue.removeFirst()
            if (current == destination) return true
            for (i in edges.indices) {
                val (u, v) = edges[i]
                if (u == current && !visited[v]) {
                    queue.addLast(v)
                    visited[v] = true
                }
                if (v == current && !visited[u]) {
                    queue.addLast(u)
                    visited[u] = true
                }
            }
        }
        return false
    }


    fun openLock(deadends: Array<String>, target: String): Int {
        val deadSet = deadends.toSet()
        val visited = mutableSetOf<String>()
        val queue = ArrayDeque<Pair<String, Int>>()
        queue.add(Pair("0000", 0))
        while (queue.isNotEmpty()) {
            val (state, turns) = queue.removeFirst()
            if (state in deadSet || state in visited) continue
            if (state == target) {
                return turns
            }
            visited.add(state)

            for (i in 0..3) {
                val digit = state[i] - '0'
                for (dir in listOf(-1, 1)) {
                    val newDigit = (digit + dir + 10) % 10
                    val next = state.substring(0, i) + "$newDigit" + state.substring(i + 1)
                    if (next !in visited && next !in deadSet) {
                        queue.add(Pair(next, turns + 1))
                    }
                }
            }
        }

        return -1
    }

    fun findMinHeightTrees(n: Int, edges: Array<IntArray>): List<Int> {
        if (n == 1) return listOf(0)
        val countMap = IntArray(n)
        val edgeMap = mutableMapOf<Int, MutableList<Int>>()
        edges.forEach { edge ->
            countMap[edge[0]]++
            countMap[edge[1]]++
            edgeMap.getOrPut(edge[0]) {
                mutableListOf()
            }.add(edge[1])
            edgeMap.getOrPut(edge[1]) {
                mutableListOf()
            }.add(edge[0])
        }

        val queue = LinkedList<Int>() // leaf queue
        countMap.forEachIndexed { index, count ->
            if (count == 1) {
                queue.add(index)
            }
        }

        var remainingNodes = n
        var levelSize = 0
        while (remainingNodes > 2) {
            levelSize = queue.size
            for (i in 0 until levelSize) {
                val current = queue.removeFirst()
                edgeMap[current]?.forEach { other ->
                    if (--countMap[other] == 1) {
                        queue.addLast(other)
                    }
                }
            }
            remainingNodes -= levelSize
        }

        return queue.toList()
    }

    class Node(var `val`: Int) {
        var neighbors: ArrayList<Node?> = ArrayList<Node?>()
    }

    fun cloneGraph(node: Node?): Node? {
        val nodeMap = hashMapOf<Node, Node>()

        fun build(current: Node?): Node? {
            if (current == null) return null
            return if (nodeMap[current] != null) {
                nodeMap[current]
            } else {
                val newNode = Node(current.`val`)
                nodeMap[current] = newNode
                val neighbors = ArrayList<Node?>()
                current.neighbors.forEach { neighbor ->
                    neighbors.add(build(neighbor))
                }
                newNode.neighbors = neighbors
                newNode
            }
        }

        return build(node)
    }

    fun findCenter(edges: Array<IntArray>): Int {
//        val freq = IntArray(edges.size + 2)
//        var maxNode = 0
//        edges.forEach { edge ->
//            edge.forEach { node ->
//                if (freq[node]++ > freq[maxNode]) {
//                    maxNode = node
//                }
//            }
//        }
//        return maxNode
        if (edges[0][0] == edges[1][0] || edges[0][0] == edges[1][1]) {
            return edges[0][0]
        } else {
            return edges[0][1]
        }
    }

    fun maximumImportance(n: Int, roads: Array<IntArray>): Long {
        val freq = IntArray(n)
        roads.forEach { road ->
            road.forEach { city ->
                freq[city]++
            }
        }
        freq.sortDescending()
        var result = 0L
        for (i in freq.indices) {
            result += freq[i].toLong() * (n - i)
        }
        return result
    }

    fun calcEquation(
        equations: List<List<String>>,
        values: DoubleArray,
        queries: List<List<String>>
    ): DoubleArray {
        val graph = mutableMapOf<String, MutableMap<String, Double>>()

        for ((index, edge) in equations.withIndex()) {
            val a = edge[0]
            val b = edge[1]
            val value = values[index]

            if (a !in graph) graph[a] = mutableMapOf()
            if (b !in graph) graph[b] = mutableMapOf()

            graph[a]!![b] = value
            graph[b]!![a] = 1.0 / value
        }

        // Floyd-Warshall
        for (k in graph.keys) {
            for (i in graph.keys) {
                for (j in graph.keys) {
                    if (i in graph[k]!! && j in graph[k]!!) {
                        val newValue = graph[i]!![k]!! * graph[k]!![j]!!
                        if (j !in graph[i]!! || newValue > graph[i]!![j]!!) {
                            // 如果当前不存在从顶点i到顶点j的路径，或者新的路径的权重大于当前的路径的权重
                            graph[i]!![j] = newValue
                            graph[j]!![i] = 1.0 / newValue
                        }
                    }
                }
            }
        }

        val result = DoubleArray(queries.size)
        queries.withIndex().forEach { query ->
            when {
                graph[query.value[0]]?.containsKey(query.value[1]) == true -> {
                    result[query.index] = graph[query.value[0]]!![query.value[1]]!!
                }

                else -> {
                    result[query.index] = -1.0
                }
            }
        }

        return result
    }

    fun getAncestorsFloyd(n: Int, edges: Array<IntArray>): List<List<Int>> {
        val graph = Array(n) { BooleanArray(n) { false } }

        for (edge in edges) {
            graph[edge[0]][edge[1]] = true
        }

        // 使用 Floyd-Warshall 算法计算传递闭包
        for (k in 0 until n) {
            for (i in 0 until n) {
                for (j in 0 until n) {
                    graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j])
                }
            }
        }

        val result = List(n) { mutableListOf<Int>() }
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (graph[j][i]) {
                    result[i].add(j)
                }
            }
        }
        return result
    }

    fun getAncestorsDFS(n: Int, edges: Array<IntArray>): List<List<Int>> {
        val graph = Array(n) { mutableListOf<Int>() }
        for (edge in edges) {
            graph[edge[1]].add(edge[0])
        }

        val ancestors = Array(n) { sortedSetOf<Int>() }
        val visited = Array(n) { BooleanArray(n) }

        fun dfs(node: Int, ancestor: Int) {
            if (visited[node][ancestor]) return
            visited[node][ancestor] = true
            ancestors[node].add(ancestor)
            for (parent in graph[ancestor]) {
                dfs(node, parent)
            }
        }

        for (i in 0 until n) {
            for (ancestor in graph[i]) {
                dfs(i, ancestor)
            }
        }

        val result = List(n) { mutableListOf<Int>() }
        for (i in 0 until n) {
            result[i].addAll(ancestors[i])
        }

        return result
    }

    fun maxProbability(n: Int, edges: Array<IntArray>, succProb: DoubleArray, start_node: Int, end_node: Int): Double {
//        val graph = Array(n) { DoubleArray(n) }
//        for (i in edges.indices) {
//            graph[edges[i][0]][edges[i][1]] = succProb[i]
//            graph[edges[i][1]][edges[i][0]] = succProb[i]
//        }
//        for (k in 0 until n) {
//            for (i in 0 until n) {
//                for (j in 0 until n) {
//                    if (graph[i][k] != 0.0 && graph[k][j] != 0.0) {
//                        graph[i][j] = maxOf(graph[i][j], graph[i][k] * graph[k][j])
//                    }
//                }
//            }
//        }
//        return graph[start_node][end_node]

        // Dijkstra
        val graph = Array(n) { mutableListOf<Pair<Int, Double>>() }
        for (i in edges.indices) {
            graph[edges[i][0]].add(Pair(edges[i][1], succProb[i]))
            graph[edges[i][1]].add(Pair(edges[i][0], succProb[i]))
        }
        val maxProb = DoubleArray(n) { 0.0 } // prob starting from start_node to i
        maxProb[start_node] = 1.0
        val pq = PriorityQueue<Pair<Int, Double>>(compareByDescending { it.second })
        pq.add(Pair(start_node, 1.0))
        while (pq.isNotEmpty()) {
            val (node, prob) = pq.poll()!!
            if (node == end_node) return prob
            for ((neighbor, edgeProb) in graph[node]) {
                val newProb = prob * edgeProb
                if (newProb > maxProb[neighbor]) {
                    maxProb[neighbor] = newProb
                    pq.add(Pair(neighbor, newProb))
                }
            }
        }
        return 0.0
    }

    fun countPaths(n: Int, roads: Array<IntArray>): Int {
        val MOD = 1000000007
        val graph = Array(n) { mutableListOf<Pair<Int, Int>>() }
        for ((u, v, time) in roads) {
            graph[u].add(Pair(v, time))
            graph[v].add(Pair(u, time))
        }
        val pathCount = IntArray(n) { 0 }
        pathCount[0] = 1
        val nodeTime = LongArray(n) { Long.MAX_VALUE }
        nodeTime[0] = 0
        val pq = PriorityQueue<Pair<Int, Long>>(compareBy { it.second })
        pq.offer(Pair(0, 0))
        while (pq.isNotEmpty()) {
            val (node, time) = pq.poll()!!
            if (time > nodeTime[node]) continue
            for ((neighbour, roadTime) in graph[node]) {
                val newTime = time + roadTime
                if (newTime < nodeTime[neighbour]) {
                    nodeTime[neighbour] = newTime
                    pathCount[neighbour] = pathCount[node]
                    pq.offer(Pair(neighbour, newTime))
                } else if (newTime == nodeTime[neighbour]) {
                    pathCount[neighbour] = (pathCount[neighbour] + pathCount[node]) % MOD
                }
            }
        }
        return pathCount[n - 1]
    }

    fun countSubIslands(grid1: Array<IntArray>, grid2: Array<IntArray>): Int {
        val rows = grid1.size
        val cols = grid1[0].size

        fun dfs(i: Int, j: Int): Boolean {
            if (i < 0 || i >= rows || j < 0 || j >= cols || grid2[i][j] == 0) {
                return true
            }
            grid2[i][j] = 0
            val result1 = dfs(i + 1, j)
            val result2 = dfs(i - 1, j)
            val result3 = dfs(i, j + 1)
            val result4 = dfs(i, j - 1)
            return grid1[i][j] == 1 && result1 && result2 && result3 && result4
        }

        var result = 0
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                if (grid2[i][j] == 1) {
                    if (dfs(i, j)) {
                        result++
                    }
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

    fun countSubIslandsUF(grid1: Array<IntArray>, grid2: Array<IntArray>): Int {
        val rows = grid1.size
        val cols = grid1[0].size
        val uf2 = UnionFind(rows * cols)

        fun index(i: Int, j: Int) = i * cols + j

        for (i in 0 until rows) {
            for (j in 0 until cols) {
                if (grid2[i][j] == 1) {
                    if (i + 1 < rows && grid2[i + 1][j] == 1) {
                        uf2.union(index(i, j), index(i + 1, j))
                    }
                    if (j + 1 < cols && grid2[i][j + 1] == 1) {
                        uf2.union(index(i, j), index(i, j + 1))
                    }
                }
            }
        }

        val totalRoots = mutableSetOf<Int>()
        val invalidRoots = mutableSetOf<Int>()
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                if (grid2[i][j] == 1) {
                    val root = uf2.find(index(i, j))
                    totalRoots.add(root)
                    if (grid1[i][j] == 0) {
                        invalidRoots.add(root)
                    }
                }
            }
        }

        return totalRoots.size - invalidRoots.size
    }

    fun canFinish(numCourses: Int, prerequisites: Array<IntArray>): Boolean {
        val graph = Array(numCourses) { mutableListOf<Int>() }
        for ((u, v) in prerequisites) {
            graph[u].add(v)
        }

        val visited = BooleanArray(numCourses) // visited course is safe
        val inRecurStack = BooleanArray(numCourses)

        fun validCourse(index: Int): Boolean {
            if (visited[index]) return true
            if (inRecurStack[index]) return false
            inRecurStack[index] = true
            for (req in graph[index]) {
                if (!validCourse(req)) {
                    visited[index] = false
                    return false
                }
            }
            visited[index] = true
            inRecurStack[index] = false
            return true
        }

        for (i in 0 until numCourses) {
            if (!validCourse(i)) {
                return false
            }
        }
        return true
    }

    fun eventualSafeNodes(graph: Array<IntArray>): List<Int> {
        val n = graph.size
        val nodeValid = BooleanArray(n)
        val visited = BooleanArray(n)
        for (from in graph.indices) {
            if (graph[from].isEmpty()) {
                visited[from] = true
                nodeValid[from] = true
            }
        }

        val inRecurStack = BooleanArray(n)

        fun checkNode(index: Int): Boolean {
            if (visited[index]) {
                return nodeValid[index]
            }
            if (inRecurStack[index]) {
                return false
            }
            inRecurStack[index] = true
            var isValid = true
            for (next in graph[index]) {
                if (!checkNode(next)) {
                    isValid = false
                    break
                }
            }
            inRecurStack[index] = false
            visited[index] = true
            nodeValid[index] = isValid
            return isValid
        }

        val result = mutableListOf<Int>()
        for (i in 0 until n) {
            if (checkNode(i)) {
                result.add(i)
            }
        }
        return result
    }

    fun findOrder(numCourses: Int, prerequisites: Array<IntArray>): IntArray {
        val inDegree = IntArray(numCourses)
        val graph = Array(numCourses) { mutableListOf<Int>() }
        for ((a, b) in prerequisites) {
            graph[a].add(b)
            inDegree[b]++
        }
        val queue = LinkedList<Int>()
        // 0 require courses can add to queue
        for (i in 0 until numCourses) {
            if (inDegree[i] == 0) {
                queue.offer(i)
            }
        }
        var index = numCourses - 1
        val order = IntArray(numCourses)
        while (queue.isNotEmpty()) {
            val course = queue.poll()!!
            order[index--] = course
            for (next in graph[course]) {
                inDegree[next]--
                if (inDegree[next] == 0) {
                    queue.offer(next)
                }
            }
        }
        return if (index == -1) order else intArrayOf()
    }

    fun removeStones(stones: Array<IntArray>): Int {
        val adj = mutableMapOf<Int, MutableList<Int>>()
        val offset = 10001

        // 构建邻接表，将横坐标和偏移后的纵坐标连接
        for ((u, v) in stones) {
            adj.getOrPut(u) { mutableListOf() }.add(v + offset)
            adj.getOrPut(v + offset) { mutableListOf() }.add(u)
        }

        // DFS 查找连通分量
        var components = 0
        val visited = mutableSetOf<Int>()

        fun dfs(node: Int) {
            visited.add(node)
            for (neighbor in adj[node] ?: emptyList()) {
                if (neighbor !in visited) {
                    dfs(neighbor)
                }
            }
        }

        // 遍历所有节点，找到所有连通分量
        for ((u, v) in stones) {
            if (u !in visited) {
                dfs(u)
                components++
            }
        }

        return stones.size - components
    }

    fun removeStonesUF(stones: Array<IntArray>): Int {

        class UnionFind {
            private val parent = mutableMapOf<Int, Int>()
            private val rank = mutableMapOf<Int, Int>()

            fun find(x: Int): Int {
                if (parent.getOrPut(x) { x } != x) {
                    parent[x] = find(parent[x]!!)
                }
                return parent[x]!!
            }

            fun union(x: Int, y: Int) {
                val rootX = find(x)
                val rootY = find(y)
                if (rootX != rootY) {
                    val rankX = rank.getOrPut(rootX) { 0 }
                    val rankY = rank.getOrPut(rootY) { 0 }
                    if (rankX > rankY) {
                        parent[rootY] = rootX
                    } else if (rankX < rankY) {
                        parent[rootX] = rootY
                    } else {
                        parent[rootY] = rootX
                        rank[rootX] = rankX + 1
                    }
                }
            }
        }

        val uf = UnionFind()
        for ((x, y) in stones) {
            uf.union(x, y + 10001) // 将行和列映射到不同的范围，连接x, y，画出一条线
        }

        val rootSet = mutableSetOf<Int>()
        for ((x, y) in stones) {
            rootSet.add(uf.find(x)) // 统计行的根节点
            rootSet.add(uf.find(y + 10001)) // 统计列的根节点
        }
        return stones.size - rootSet.size
    }

    fun findChampion(n: Int, edges: Array<IntArray>): Int {
        val inDegree = IntArray(n)
        for (edge in edges) {
            inDegree[edge[1]]++
        }
        var result = -1
        for (i in 0 until n) {
            if (inDegree[i] == 0) {
                if (result == -1) {
                    result = i
                } else {
                    return -1
                }
            }
        }
        return result
    }

    fun shortestDistanceAfterQueries(n: Int, queries: Array<IntArray>): IntArray {
//        val minSteps = IntArray(n) { Int.MAX_VALUE }
//        val connected = Array(n) { from ->
//            BooleanArray(n) { to ->
//                to == from + 1
//            }
//        }
//        val result = IntArray(queries.size)
//        for (i in queries.indices) {
//            connected[queries[i][0]][queries[i][1]] = true
//            for (j in 1 until n) {
//                for (k in j - 1 downTo 0) {
//                    if (connected[k][j]) {
//                        minSteps[j] = minOf(minSteps[j], minSteps[k] + 1)
//                    }
//                }
//            }
//            result[i] = minSteps[n - 1]
//        }
//        return result

        val graph = Array(n - 1) { mutableListOf(it + 1) }

        fun bfs() : Int {
            val steps = IntArray(n) { Int.MAX_VALUE }
            steps[0] = 0
            val queue = LinkedList<Int>()
            queue.offer(0)
            while (queue.isNotEmpty()) {
                val current = queue.poll()!!
                for (neighbour in graph[current]) {
                    if (steps[neighbour] > steps[current] + 1) {
                        steps[neighbour] = steps[current] + 1
                        if (neighbour == n - 1) break
                        queue.offer(neighbour)
                    }
                }
            }
            return steps[n - 1]
        }

        val result = IntArray(queries.size)
        for (i in queries.indices) {
            graph[queries[i][0]].add(queries[i][1])
            result[i] = bfs()
        }
        return result
    }
}