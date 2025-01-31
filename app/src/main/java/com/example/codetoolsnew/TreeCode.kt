package com.example.codetools

import java.util.LinkedList
import java.util.Queue
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.pow


object TreeCode {
    class TreeNode(var `val`: Int) {
        var left: TreeNode? = null
        var right: TreeNode? = null

        override fun toString(): String {
            return this.toIntArray().joinToString()
        }
    }

    fun TreeNode?.toIntArray(): Array<Int?> {
        val result = mutableListOf<Int?>()
        if (this == null) return result.toTypedArray()

        val queue: Queue<TreeNode?> = LinkedList()
        queue.offer(this)

        while (queue.isNotEmpty()) {
            val node = queue.poll()
            if (node == null) {
                result.add(null)
            } else {
                result.add(node.`val`)
                queue.offer(node.left)
                queue.offer(node.right)
            }
        }

        return result.toTypedArray()
    }

    fun IntArray.intArrayToTree(): TreeNode? {
        if (this.isEmpty()) return null

        val root = TreeNode(this[0])
        val queue: Queue<TreeNode> = LinkedList()
        queue.offer(root)

        var i = 1
        while (i < this.size) {
            val parent = queue.poll()

            val leftValue = this[i]
            if (leftValue != Int.MIN_VALUE) {
                parent.left = TreeNode(leftValue)
                queue.offer(parent.left)
            }
            i++

            if (i < this.size) {
                val rightValue = this[i]
                if (rightValue != Int.MIN_VALUE) {
                    parent.right = TreeNode(rightValue)
                    queue.offer(parent.right)
                }
                i++
            }
        }

        return root
    }

    fun inorderTraversalIterative(root: TreeNode?): List<Int> {
        val result = mutableListOf<Int>()
        val stack = LinkedList<TreeNode>()
        var current = root

        while (current != null || stack.isNotEmpty()) {
            while (current != null) {
                stack.push(current)
                current = current.left
            }

            current = stack.pop()
            result.add(current.`val`)

            current = current.right
        }
        return result
    }

    fun inorderTraversalMorris(root: TreeNode?): List<Int> {
        val result = mutableListOf<Int>()
        var current = root

        while (current != null) {
            if (current.left == null) {
                result.add(current.`val`)
                current = current.right
            } else {
                var predecessor = current.left
                while (predecessor!!.right != null && predecessor.right != current) {
                    predecessor = predecessor.right
                }
                if (predecessor.right == null) {
                    predecessor.right = current
                    current = current.left
                } else {
                    predecessor.right = null
                    result.add(current.`val`)
                    current = current.right
                }
            }
        }
        return result
    }

    fun findModeMorris(root: TreeNode?): IntArray {
        val result = mutableListOf<Int>()
        var maxFreq = 0
        var previous = -100001
        var count = 0

        fun processNode(node: TreeNode) {
            if (node.`val` == previous) {
                count++
            } else {
                count = 1
                previous = node.`val`
            }
            if (count > maxFreq) {
                maxFreq = count
                result.clear()
            }
            if (count == maxFreq) {
                result.add(previous)
            }
        }

        var current = root
        while (current != null) {
            if (current.left == null) {
                processNode(current)
                current = current.right
            } else {
                var processor = current.left
                while (processor!!.right != null && processor.right != current) {
                    processor = processor.right
                }
                if (processor.right == null) {
                    processor.right = current
                    current = current.left
                } else {
                    processor.right = null
                    processNode(current)
                    current = current.right
                }
            }
        }
        return result.toIntArray()
    }


    fun postorderTraversal(root: TreeNode?): List<Int> {
        val result = mutableListOf<Int>()

        fun postOrder(node: TreeNode?) {
            if (node == null) return

            if (node.left != null) {
                postOrder(node.left)
            }
            if (node.right != null) {
                postOrder(node.right)
            }
            result.add(node.`val`)
        }
        postOrder(root)
        return result
    }

    fun postorderTraversalIterative(root: TreeNode?): List<Int> {
        val result = mutableListOf<Int>()
        val stack = LinkedList<TreeNode>()
        var current = root
        var prev: TreeNode? = null // to make sure right child has been visited

        while (current != null || stack.isNotEmpty()) {
            // 1. 一直向左走，直到左子树为空
            while (current != null) {
                stack.push(current)
                current = current.left
            }
            // 2. 查看栈顶节点，但不弹出
            current = stack.peek()
            // 3. 如果右子树为空或者已经访问过，弹出当前节点并加入结果
            if (current?.right == null || current.right == prev) {
                stack.pop()
                result.add(current.`val`)
                prev = current
                current = null  // 继续处理栈中的上一个节点
            } else {
                // 4. 如果右子树还没被访问，先处理右子树
                current = current.right
            }
        }
        return result
    }

    class Node(var `val`: Int) {
        var children: List<Node?> = listOf()
    }

    fun postOrder(root: Node?): List<Int> {
        val result = mutableListOf<Int>()
        val stack = LinkedList<Node>()
        val visited = mutableSetOf<Node>()

        if (root != null) stack.push(root)
        while (stack.isNotEmpty()) {
            val node = stack.peek()!!
            if (node.children.isEmpty() || node in visited) {
                result.add(stack.pop().`val`)
            } else {
                visited.add(node)
                for (i in node.children.size - 1 downTo 0) {
                    stack.push(node.children[i])
                }
            }
        }
        return result
    }

    fun maxDepth(root: Node?): Int {
        if (root == null) return 0
        var max = 0
        root.children.forEach { child ->
            max = maxOf(max, maxDepth(child))
        }
        return max + 1
    }

    fun maxDepthIterative(root: Node?): Int {
        if (root == null) return 0
        var depth = 0
        val queue = LinkedList<Node>()
        queue.offer(root)
        while (queue.isNotEmpty()) {
            val size = queue.size
            for (i in 0 until size) {
                val node = queue.poll()!!
                node.children.forEach { child ->
                    queue.offer(child)
                }
            }
            depth++
        }
        return depth
    }

    fun isSameTree(p: TreeNode?, q: TreeNode?): Boolean {
//        val result1 = if (p?.left != null && q?.left != null) {
//            isSameTree(p.left, q.left)
//        } else !(p?.left != null || q?.left != null)
//        if (!result1) return false
//
//        val result2 = if (p != null && q != null) {
//            p.`val` == q.`val`
//        } else !(p != null || q != null)
//        if (!result2) return false
//
//        val result3 = if (p?.right != null && q?.right != null) {
//            isSameTree(p.right, q.right)
//        } else !(p?.right != null || q?.right != null)
//        if (!result3) return false
//
//        return true

        // 如果两个节点都为空，则它们相同
        if (p == null && q == null) return true
        // 如果其中一个节点为空，或者节点的值不相等，则它们不相同
        if (p == null || q == null || p.`val` != q.`val`) return false
        // 递归比较左子树和右子树
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right)
    }

    fun isSymmetricDFS(root: TreeNode?): Boolean {
        return isMirrorTree(root?.left, root?.right)
    }

    fun isMirrorTree(p: TreeNode?, q: TreeNode?): Boolean {
        if (p == null && q == null) return true
        if (p == null || q == null || p.`val` != q.`val`) return false
        return isMirrorTree(p.left, q.right) && isMirrorTree(p.right, q.left)
    }

    fun isSymmetricBFS(root: TreeNode?): Boolean {
        val queue = LinkedList<TreeNode?>()
        queue.add(root?.left)
        queue.add(root?.right)
        while (queue.isNotEmpty()) {
            val p = queue.poll()
            val q = queue.poll()
            if (p == null && q == null) continue
            if (p == null || q == null || q.`val` != p.`val`) {
                return false
            }
            queue.add(p.left)
            queue.add(q.right)
            queue.add(p.right)
            queue.add(q.left)
        }
        return true
    }

    fun maxDepth(root: TreeNode?): Int {
        if (root == null) return 0
        return 1 + maxOf(maxDepth(root.left), maxDepth(root.right))
    }

    fun maxDepthIterative(root: TreeNode?): Int {
        val queue = LinkedList<TreeNode>()
        queue.offer(root)
        var depth = 0
        while (queue.isNotEmpty()) {
            val size = queue.size
            for (i in 0 until size) {
                val node = queue.poll()!!
                node.left?.let { queue.offer(it) }
                node.right?.let { queue.offer(it) }
            }
            depth++
        }
        return depth
    }

    fun sortedArrayToBST(nums: IntArray): TreeNode? {
//        if (nums.isEmpty()) return null
//        val mid = (nums.size - 1) / 2
//        val root = TreeNode(nums[mid]).apply {
//            left = if (mid > 0) sortedArrayToBST(nums.copyOfRange(0, mid)) else null
//            right = if (nums.size > mid + 1) sortedArrayToBST(
//                nums.copyOfRange(
//                    mid + 1,
//                    nums.size
//                )
//            ) else null
//        }
//        return root

        fun buildTree(from: Int, to: Int) : TreeNode? {
            if (from > to) return null
            val mid = from + ((to - from) shr 1)
            return TreeNode(nums[mid]).apply {
                left = buildTree(from, mid - 1)
                right = buildTree(mid + 1, to)
            }
        }
        return buildTree(0, nums.size - 1)
    }

    fun isBalanced(root: TreeNode?): Boolean {
        if (root == null) return true
        if (abs(maxDepth(root.left) - maxDepth(root.right)) > 1) return false
        return isBalanced(root.left) && isBalanced(root.right)
    }

    fun minDepth(root: TreeNode?): Int {
        if (root == null) return 0
        if (root.left == null && root.right == null) return 1

        return when {
            root.left == null -> {
                1 + minDepth(root.right)
            }

            root.right == null -> {
                1 + minDepth(root.left)
            }

            else -> {
                1 + min(minDepth(root.left), minDepth(root.right))
            }
        }
    }

    fun hasPathSum(root: TreeNode?, targetSum: Int): Boolean {
        if (root == null) return false
        val restSum = targetSum - root.`val`
        if (root.left == null && root.right == null) return root.`val` == restSum
        return hasPathSum(root.right, restSum) || hasPathSum(root.left, restSum)
    }

    fun sumPathNumbers(root: TreeNode?): Int {
        //[4,9,0,5,1] -> 495 + 491 + 40
        if (root == null) return 0
        val currentPath = LinkedList<Int>()
        var sum = 0

        fun dfs(node: TreeNode?) {
            if (node == null) return
            currentPath.addLast(node.`val`)
            // If the current node is a leaf node, calculate sum of this path
            if (node.left == null && node.right == null) {
                var position = currentPath.size - 1
                for (n in currentPath) {
                    sum += n * 10.0.pow(position--).toInt()
                }
            } else {
                dfs(node.left)
                dfs(node.right)
            }
            // Backtrack: Remove the current node from the current path
            currentPath.removeLast()
        }

        dfs(root)
        return sum
    }

    fun sumOfLeaves(root: TreeNode?): Int {
        if (root == null) return 0
        if (root.left == null && root.right == null) {
            // If it's a leaf node, add its value
            return root.`val`
        }
        val leftSum = sumOfLeaves(root.left)
        val rightSum = sumOfLeaves(root.right)
        return leftSum + rightSum
    }

    fun sumOfLeftLeaves(root: TreeNode?): Int {
        return sumOfLeftLeavesHelper(root, false)
    }

    fun sumOfLeftLeavesHelper(node: TreeNode?, isLeftChild: Boolean): Int {
        if (node == null) return 0
        if (node.left == null && node.right == null) {
            // If it's a leaf node and a left child, add its value
            return if (isLeftChild) node.`val` else 0
        }
        val leftSum = sumOfLeftLeavesHelper(node.left, true)
        val rightSum = sumOfLeftLeavesHelper(node.right, false)
        return leftSum + rightSum
    }

    fun addOneRow(root: TreeNode?, `val`: Int, depth: Int): TreeNode? {
        if (root == null) return null
        if (depth == 1) return TreeNode(`val`).apply { left = root }
        val queue = LinkedList<TreeNode>()
        queue.add(root)
        var currentDepth = 1
        var levelCount = 1

        while (queue.isNotEmpty()) {
            for (i in 0 until levelCount) {
                val node = queue.removeFirst()
                if (currentDepth == depth - 1) {
                    node.left = TreeNode(`val`).apply { left = node.left }
                    node.right = TreeNode(`val`).apply { right = node.right }
                } else {
                    if (node.left != null) queue.addLast(node.left!!)
                    if (node.right != null) queue.addLast(node.right!!)
                }
            }
            levelCount = queue.size
            currentDepth++
        }

        return root
    }

    fun smallestFromLeaf(root: TreeNode?): String {
        // be careful that "hud" is smaller than "jd"
        if (root == null) return ""

        val currentPath = StringBuilder()
        var result: String? = null

        fun dfs(node: TreeNode?) {
            if (node == null) return
            currentPath.append(Char('a'.code + node.`val`))
            if (node.left == null && node.right == null) {
                currentPath.reverse().toString().let {
                    result = if (result == null) {
                        it
                    } else {
                        minOf(it, result!!)
                    }
                }
                currentPath.reverse()
            } else {
                dfs(node.left)
                dfs(node.right)
            }
            currentPath.deleteCharAt(currentPath.length - 1)
        }

        dfs(root)
        return result ?: ""
    }

    fun averageOfLevels(root: TreeNode?): DoubleArray {
        if (root == null) return DoubleArray(0)
        val queue = ArrayDeque<TreeNode>()
        queue.add(root)
        val result = mutableListOf<Double>()
        var levelSum = 0.0
        var levelCount = 1

        // bfs
        while (queue.isNotEmpty()) {
            for (i in 0 until levelCount) {
                val node = queue.removeFirst()
                levelSum += node.`val`
                if (node.left != null) queue.addLast(node.left!!)
                if (node.right != null) queue.addLast(node.right!!)
            }
            result.add(levelSum / levelCount)
            levelSum = 0.0
            levelCount = queue.size
        }
        return result.toDoubleArray()
    }

    fun flatten(root: TreeNode?): Unit {
        if (root == null) return

        fun preorderTraversal(node: TreeNode?): TreeNode? {
            var left = if (node?.left != null) {
                preorderTraversal(node.left)
            } else {
                null
            }
            val right = if (node?.right != null) {
                preorderTraversal(node.right)
            } else {
                null
            }
            if (left == null) {
                node?.right = right
            } else {
                node?.right = left
                while (left?.right != null) {
                    left = left.right
                }
                left?.right = right
            }
            node?.left = null
            return node
        }

        preorderTraversal(root)

        println(root.toIntArray().joinToString())
    }

    fun rightSideView(root: TreeNode?): List<Int> {
        if (root == null) return listOf()
        val result = mutableListOf<Int>()
        val queue = LinkedList<TreeNode>()
        queue.add(root)
        var levelSize = 1
        while (queue.isNotEmpty()) {
            result.add(queue.last().`val`)
            for (i in 0 until levelSize) {
                val current = queue.removeFirst()
                current.left?.let { queue.addLast(it) }
                current.right?.let { queue.addLast(it) }
            }
            levelSize = queue.size
        }
        return result
    }

    fun evaluateTree(root: TreeNode?): Boolean {
        return if (root?.left == null || root.right == null) {
            root!!.`val` == 1
        } else {
            val left = evaluateTree(root.left)
            val right = evaluateTree(root.right)
            if (root.`val` == 2) {
                left or right
            } else {
                left and right
            }
        }
    }

    fun buildTreePreIn(preorder: IntArray, inorder: IntArray): TreeNode? {
        if (preorder.isEmpty() || inorder.isEmpty()) return null
        val inorderIndexMap = inorder.withIndex().associate { it.value to it.index }

        fun buildTreeRecursive(preStart: Int, preEnd: Int, inStart: Int, inEnd: Int): TreeNode? {
            if (preStart > preEnd || inStart > inEnd) return null

            val rootValue = preorder[preStart]
            val root = TreeNode(rootValue)
            val rootIndex = inorderIndexMap[rootValue]!!

            val leftTreeSize = rootIndex - inStart

            root.left =
                buildTreeRecursive(preStart + 1, preStart + leftTreeSize, inStart, rootIndex - 1)
            root.right =
                buildTreeRecursive(preStart + leftTreeSize + 1, preEnd, rootIndex + 1, inEnd)

            return root
        }

        return buildTreeRecursive(0, preorder.lastIndex, 0, inorder.lastIndex)
    }

    fun buildTreeInPost(inorder: IntArray, postorder: IntArray): TreeNode? {
        if (inorder.isEmpty()) return null
        val inorderIndexMap = inorder.withIndex().associate { it.value to it.index }

        fun constructTree(inStart: Int, inEnd: Int, postStart: Int, postEnd: Int): TreeNode? {
            if (inStart > inEnd || postStart > postEnd) return null

            val root = TreeNode(postorder[postEnd])
            val rootIndex = inorderIndexMap[postorder[postEnd]]!!
            val leftSize = rootIndex - inStart

            root.left =
                constructTree(inStart, rootIndex - 1, postStart, postStart + leftSize - 1)
            root.right =
                constructTree(rootIndex + 1, inEnd, postStart + leftSize, postEnd - 1)
            return root
        }

        return constructTree(0, inorder.lastIndex, 0, postorder.lastIndex)
    }

    fun removeTargetLeafNodes(root: TreeNode?, target: Int): TreeNode? {
        val queue = LinkedList<TreeNode>()
        val parentMap = mutableMapOf<TreeNode, Pair<TreeNode, Boolean>>() // isLeft
        val dummy = TreeNode(0).apply { left = root }

        fun dfs(node: TreeNode?) {
            if (node == null) return
            if (node.left == null && node.right == null) {
                if (node.`val` == target) {
                    queue.offer(node)
                }
                return
            }
            if (node.left != null) {
                if (node.left!!.`val` == target) {
                    parentMap[node.left!!] = Pair(node, true)
                }
                dfs(node.left)
            }
            if (node.right != null) {
                if (node.right!!.`val` == target) {
                    parentMap[node.right!!] = Pair(node, false)
                }
                dfs(node.right)
            }
        }
        dfs(dummy)

        var current: TreeNode
        var parent: TreeNode
        while (queue.isNotEmpty()) {
            current = queue.removeFirst()
            parent = parentMap[current]!!.first
            if (parentMap[current]!!.second) {
                parent.left = null
            } else {
                parent.right = null
            }
            if (parent.left == null && parent.right == null && parent.`val` == target) {
                queue.offer(parent)
            }
        }
        return dummy.left
    }

    fun distributeCoins(root: TreeNode?): Int {
        var moves = 0

        // calculate excess coins on this node
        fun dfs(node: TreeNode?): Int {
            if (node == null) return 0

            val leftExcess = dfs(node.left)
            val rightExcess = dfs(node.right)

            // The total moves is the sum of absolute excess coins moved in and out of this node
            moves += abs(leftExcess) + abs(rightExcess)

            // Current node's excess coins (including coins received from children)
            return node.`val` + leftExcess + rightExcess - 1
        }

        dfs(root)
        return moves
    }

    fun invertTree(root: TreeNode?): TreeNode? {
        if (root?.left != null || root?.right != null) {
            val invertedRight = invertTree(root.right)
            val invertedLeft = invertTree(root.left)
            root.left = invertedRight
            root.right = invertedLeft
        }
        return root
    }

    fun getMinimumDifferenceInBST(root: TreeNode?): Int {
        var min = Int.MAX_VALUE
        var previous = -1
        fun dfs(node: TreeNode?) {
            if (node == null) return
            dfs(node.left)
            if (previous != -1) {
                min = minOf(min, node.`val` - previous)
            }
            previous = node.`val`
            dfs(node.right)
        }
        dfs(root)
        return min
    }

    fun isValidBST(root: TreeNode?): Boolean {
        var previous: Int? = null
        fun inOrder(node: TreeNode?): Boolean {
            if (node == null) return true
            if (!inOrder(node.left)) return false
            if (previous != null && previous!! >= node.`val`) {
                return false
            }
            previous = node.`val`
            if (!inOrder(node.right)) return false
            return true
        }
        return inOrder(root)
    }

    fun bstToGst(root: TreeNode?): TreeNode? {
        var sum = 0
        fun dfs(node: TreeNode?) {
            if (node == null) return
            dfs(node.right)

            sum += node.`val`
            node.`val` = sum

            dfs(node.left)
        }

        dfs(root)
        return root
    }

    fun kthSmallestDFS(root: TreeNode?, k: Int): Int {
        var count = 0
        var result = -1
        fun dfs(node: TreeNode?) {
            if (node == null) return
            dfs(node.left)
            count++
            if (count == k) result = node.`val`
            else dfs(node.right)
        }

        dfs(root)
        return result
    }

    fun kthSmallestBFS(root: TreeNode?, k: Int): Int {
        val queue = ArrayDeque<TreeNode>()
        var current = root
        var count = 0
        while (queue.isNotEmpty() || current != null) {
            while (current != null) {
                queue.addLast(current)
                current = current.left
            }
            current = queue.removeLast()
            count++
            if (count == k) {
                return current.`val`
            }

            current = current.right
        }
        throw IllegalArgumentException("k is larger than the number of nodes in the tree")
    }

    fun balanceBST(root: TreeNode?): TreeNode? {
        fun inorderTraversal(node: TreeNode?): List<Int> {
            val result = ArrayList<Int>()
            if (node?.left != null) {
                result.addAll(inorderTraversal(node.left))
            }
            if (node != null) {
                result.add(node.`val`)
            }
            if (node?.right != null) {
                result.addAll(inorderTraversal(node.right))
            }
            return result
        }

        val array = inorderTraversal(root)

        fun buildBST(start: Int, end: Int): TreeNode? {
            if (start > end) return null
            val mid = start + (end - start) / 2
            return TreeNode(array[mid]).apply {
                left = buildBST(start, mid - 1)
                right = buildBST(mid + 1, end)
            }
        }

        return buildBST(0, array.lastIndex)
    }

    class Trie() {
        val root = LetterNode(-1)

        class LetterNode(var code: Int) {
            val children = Array<LetterNode?>(26) { null }
            var end = false
        }

        fun insert(word: String) {
            fun insert(letterIndex: Int, node: LetterNode) {
                if (letterIndex == word.length) {
                    node.end = true
                    return
                }
                val wordCode = word[letterIndex] - 'a'
                if (node.children[wordCode] == null) {
                    node.children[wordCode] = LetterNode(wordCode)
                }
                insert(letterIndex + 1, node.children[wordCode]!!)
            }
            insert(0, root)
        }

        fun search(word: String): Boolean {
            fun search(letterIndex: Int, node: LetterNode): Boolean {
                if (letterIndex == word.length) {
                    return node.end
                }
                val wordCode = word[letterIndex] - 'a'
                if (node.children[wordCode] == null) return false
                return search(letterIndex + 1, node.children[wordCode]!!)
            }
            return search(0, root)
        }

        fun startsWith(prefix: String): Boolean {
            fun startsWith(letterIndex: Int, node: LetterNode): Boolean {
                if (letterIndex >= prefix.length) return true
                val wordCode = prefix[letterIndex] - 'a'
                if (node.children[wordCode] == null) return false
                return startsWith(letterIndex + 1, node.children[wordCode]!!)
            }
            return startsWith(0, root)
        }
    }

    class WordDictionary() {
        val root = LetterNode(-1)

        class LetterNode(var code: Int) {
            val children = Array<LetterNode?>(26) { null }
            var end = false
        }

        fun addWord(word: String) {
            fun insert(letterIndex: Int, node: LetterNode) {
                if (letterIndex == word.length) {
                    node.end = true
                    return
                }
                val wordCode = word[letterIndex] - 'a'
                if (node.children[wordCode] == null) {
                    node.children[wordCode] = LetterNode(wordCode)
                }
                insert(letterIndex + 1, node.children[wordCode]!!)
            }
            insert(0, root)
        }

        fun search(word: String): Boolean {
            fun search(letterIndex: Int, node: LetterNode): Boolean {
                if (letterIndex == word.length) {
                    return node.end
                }
                if (word[letterIndex] == '.') {
                    return node.children.any { child ->
                        if (child == null) false
                        else search(letterIndex + 1, child)
                    }
                } else {
                    val wordCode = word[letterIndex] - 'a'
                    if (node.children[wordCode] == null) return false
                    return search(letterIndex + 1, node.children[wordCode]!!)
                }
            }
            return search(0, root)
        }
    }

    fun createBinaryTree(descriptions: Array<IntArray>): TreeNode? {
        // [parent, child, isLeft]
        val map = mutableMapOf<Int, TreeNode>()
        val parentMap = mutableMapOf<Int, Int>()
        for ((parent, child, isLeft) in descriptions) {
            val parentNode = map.getOrPut(parent) { TreeNode(parent) }
            val childNode = map.getOrPut(child) { TreeNode(child) }
            if (isLeft == 1) {
                parentNode.left = childNode
            } else {
                parentNode.right = childNode
            }
            parentMap[child] = parent
        }
        var root = descriptions[0][0]
        while (parentMap.containsKey(root)) {
            root = parentMap[root]!!
        }
        return map[root]
    }

//    fun getDirections(root: TreeNode?, startValue: Int, destValue: Int): String {
//        // BFS to find the paths from the start and dest to the root
//        val parentMap = mutableMapOf<Int, TreeNode?>()
//        val queue: Queue<TreeNode?> = LinkedList()
//        queue.offer(root)
//        parentMap[root?.`val` ?: 0] = null
//
//        var startNode: TreeNode? = null
//        var destNode: TreeNode? = null
//
//        while (queue.isNotEmpty() && (startNode == null || destNode == null)) {
//            val current = queue.poll()
//            if (current != null) {
//                if (current.`val` == startValue) startNode = current
//                if (current.`val` == destValue) destNode = current
//                if (current.left != null) {
//                    parentMap[current.left!!.`val`] = current
//                    queue.offer(current.left)
//                }
//                if (current.right != null) {
//                    parentMap[current.right!!.`val`] = current
//                    queue.offer(current.right)
//                }
//            }
//        }
//
//        // Function to build the path from node to root
//        fun pathToRoot(node: TreeNode?): List<TreeNode?> {
//            var current = node
//            val path = mutableListOf<TreeNode?>()
//            while (current != null) {
//                path.add(current)
//                current = parentMap[current.`val`]
//            }
//            return path
//        }
//
//        val startPath = pathToRoot(startNode)
//        val destPath = pathToRoot(destNode)
//
//        // Find the lowest common ancestor (LCA)
//        var i = startPath.size - 1
//        var j = destPath.size - 1
//        while (i >= 0 && j >= 0 && startPath[i] == destPath[j]) {
//            i--
//            j--
//        }
//        i++
//        j++
//
//        // Build the final path
//        val path = StringBuilder()
//        for (k in 0 until i) {
//            path.append('U')
//        }
//        for (k in j downTo 1) {
//            if (destPath[k - 1] == destPath[k]?.left) {
//                path.append('L')
//            } else {
//                path.append('R')
//            }
//        }
//
//        return path.toString()
//    }


    fun getDirections(root: TreeNode?, startValue: Int, destValue: Int): String {

        fun lca(root: TreeNode?, p: Int, q: Int): TreeNode? {
            if (root == null || root.`val` == p || root.`val` == q) return root

            val left = lca(root.left, p, q)
            val right = lca(root.right, p, q)

            return when {
                left != null && right != null -> root
                left != null -> left
                else -> right
            }
        }

        val lcaNode = lca(root, startValue, destValue)

        val pathToStart = StringBuilder()
        val pathToDest = StringBuilder()

        fun getPath(root: TreeNode?, target: Int, path: StringBuilder): Boolean {
            if (root == null) return false

            if (root.`val` == target) return true

            path.append('L')
            if (getPath(root.left, target, path)) return true
            path.deleteCharAt(path.length - 1)

            path.append('R')
            if (getPath(root.right, target, path)) return true
            path.deleteCharAt(path.length - 1)

            return false
        }

        getPath(lcaNode, startValue, pathToStart)
        getPath(lcaNode, destValue, pathToDest)

        val result = StringBuilder()
        for (i in pathToStart.indices) {
            result.append('U')
        }
        result.append(pathToDest)

        return result.toString()
    }

    fun delNodes(root: TreeNode?, to_delete: IntArray): List<TreeNode?> {
        if (root == null) return listOf()
        val setToDelete = to_delete.toMutableSet()
        val queue = LinkedList<TreeNode>()
        val result = mutableListOf<TreeNode>()

        val dummy = TreeNode(0).apply {
            left = root
        }
        if (root.`val` !in setToDelete) {
            result.add(root)
        }
        queue.offer(dummy)
        while (setToDelete.isNotEmpty() && queue.isNotEmpty()) {
            val current = queue.poll()!!
            current.left?.let { left ->
                if (setToDelete.remove(left.`val`)) {
                    current.left = null
                    left.left?.let {
                        if (it.`val` !in setToDelete) result.add(it)
                    }
                    left.right?.let {
                        if (it.`val` !in setToDelete) result.add(it)
                    }
                }
                queue.offer(left)
            }
            current.right?.let { right ->
                if (setToDelete.remove(right.`val`)) {
                    current.right = null
                    right.left?.let {
                        if (it.`val` !in setToDelete) result.add(it)
                    }
                    right.right?.let {
                        if (it.`val` !in setToDelete) result.add(it)
                    }
                }
                queue.offer(right)
            }
        }
        return result
    }

    fun delNodesDFS(root: TreeNode?, to_delete: IntArray): List<TreeNode?> {
        val setToDelete = to_delete.toHashSet()
        val result = mutableListOf<TreeNode>()

        fun dfs(node: TreeNode?, isRoot: Boolean): TreeNode? {
            if (node == null) return null

            val shouldDelete = node.`val` in setToDelete
            if (isRoot && !shouldDelete) {
                result.add(node)
            }

            node.left = dfs(node.left, shouldDelete)
            node.right = dfs(node.right, shouldDelete)

            return if (shouldDelete) null else node
        }

        dfs(root, true)
        return result
    }

    fun countGoodPairsOfLeaves(root: TreeNode?, distance: Int): Int {
        // leaves distance less than given distance
        val leafToRootPathMap = mutableMapOf<TreeNode, List<TreeNode>>()
        val path = mutableListOf<TreeNode>()

        fun dfs(node: TreeNode?) {
            if (node == null) return
            path.add(node)
            if (node.left == null && node.right == null) {
                leafToRootPathMap[node] = path.toList()
            }
            dfs(node.left)
            dfs(node.right)
            path.removeLast()
        }

        dfs(root)

        fun findLcaDepth(p: TreeNode, q: TreeNode): Int {
            val path1 = leafToRootPathMap[p]!!
            val path2 = leafToRootPathMap[q]!!
            val minLength = minOf(path1.size, path2.size)
            var depth = 0
            for (i in 0 until minLength) {
                if (path1[i] == path2[i]) {
                    depth++
                } else {
                    break
                }
            }
            return depth
        }

        val leaves = leafToRootPathMap.keys.toList()
        var goodCount = 0
        for (i in leaves.indices) {
            for (j in i + 1 until leaves.size) {
                val lcaDepth = findLcaDepth(leaves[i], leaves[j])
                val depth1 = leafToRootPathMap[leaves[i]]!!.size
                val depth2 = leafToRootPathMap[leaves[j]]!!.size

                if (depth1 + depth2 - 2 * lcaDepth <= distance) {
                    goodCount++
                }
            }
        }
        return goodCount
    }

    fun countGoodPairsOfLeavesDP(root: TreeNode?, distance: Int): Int {
        if (root == null) return 0
        var count = 0

        fun dfs(node: TreeNode): IntArray {
            val answer = IntArray(distance) { 0 }
            if (node.left == null && node.right == null) {
                answer[0] = 1
            } else {
                val left = if (node.left != null) {
                    dfs(node.left!!)
                } else {
                    IntArray(distance) { 0 }
                }
                val right = if (node.right != null) {
                    dfs(node.right!!)
                } else {
                    IntArray(distance) { 0 }
                }

                for (i in 0 until distance) {
                    for (j in i until distance - i - 1) {
                        count += left[i] * right[j]
                        if (i != j) {
                            count += right[i] * left[j]
                        }
                    }
                    if (i != distance - 1) {
                        answer[i + 1] = left[i] + right[i]
                    }
                }
            }
            return answer
        }

        dfs(root)
        return count
    }

    fun countNodes(root: TreeNode?): Int {
//        if (root == null) return 0
//        val queue = LinkedList<TreeNode>()
//        queue.offer(root)
//        var level = 0
//        while (queue.isNotEmpty()) {
//            val levelCount = queue.size
//            if (levelCount < (1 shl level)) {
//                return (1 shl level) - 1 + levelCount
//            }
//            for (i in 0 until levelCount) {
//                val current = queue.poll()
//                current?.left?.let { queue.offer(it) }
//                current?.right?.let { queue.offer(it) }
//            }
//            level++
//        }
//        return (1 shl level) - 1

        if (root == null) return 0

        var lh = 0
        var current = root
        while (current != null) {
            lh++
            current = current.left
        }

        var rh = 0
        current = root
        while (current != null) {
            rh++
            current = current.right
        }

        return if (rh == lh) {
            (1 shl lh) - 1
        } else {
            countNodes(root.left) + countNodes(root.right) + 1
        }
    }

    fun isSymmetricFlatten(root: TreeNode?): Boolean {
        val list = mutableListOf<Int?>()

        fun inOrder(node: TreeNode) {
            if (node.left == null) {
                if (node.right != null) {
                    list.add(null)
                }
            } else {
                inOrder(node.left!!)
            }
            list.add(node.`val`)
            if (node.right == null) {
                if (node.left != null) {
                    list.add(null)
                }
            } else {
                inOrder(node.right!!)
            }
        }
        inOrder(root!!)
        println(list.joinToString())
        var left = 0
        var right = list.lastIndex
        while (left < right) {
            if (list[left] != list[right]) return false
            left++
            right--
        }
        return true
    }

    fun levelOrder(root: TreeNode?): List<List<Int>> {
        if (root == null) return emptyList()
        val result = mutableListOf<MutableList<Int>>()
        val queue = LinkedList<TreeNode>()
        queue.offer(root)
        while (queue.isNotEmpty()) {
            val levelList = mutableListOf<Int>()
            val size = queue.size
            for (i in 0 until size) {
                val node = queue.poll()!!
                levelList.add(node.`val`)
                if (node.left != null) {
                    queue.offer(node.left)
                }
                if (node.right != null) {
                    queue.offer(node.right)
                }
            }
            result.add(levelList)
        }
        return result
    }

    fun zigzagLevelOrder(root: TreeNode?): List<List<Int>> {
        if (root == null) return emptyList()
        val result = mutableListOf<LinkedList<Int>>()
        var fromLeft = true
        val queue = LinkedList<TreeNode>()
        queue.offer(root)
        while (queue.isNotEmpty()) {
            val size = queue.size
            val currentList = LinkedList<Int>()
            for (i in 0 until size) {
                val node = queue.poll()!!
                if (fromLeft) {
                    currentList.add(node.`val`)
                } else {
                    currentList.addFirst(node.`val`)
                }
                node.left?.let { queue.offer(it) }
                node.right?.let { queue.offer(it) }
            }
            fromLeft = !fromLeft
            result.add(currentList)
        }
        return result
    }

    fun isCousins(root: TreeNode?, x: Int, y: Int): Boolean {
        if (root == null) return false
        var posX: Pair<Int, Int>? = null
        var posY: Pair<Int, Int>? = null
        val queue = LinkedList<Pair<TreeNode, Int>>() // node to parent
        queue.offer(root to 0)
        var depth = 0
        while (queue.isNotEmpty()) {
            val size = queue.size
            for (i in 0 until size) {
                val (node, parent) = queue.poll()!!
                if (node.`val` == x) {
                    posX = depth to parent
                } else if (node.`val` == y) {
                    posY = depth to parent
                }
                node.left?.let { queue.offer(it to node.`val`) }
                node.right?.let { queue.offer(it to node.`val`) }
            }
            if (posX != null && posY == null || posX == null && posY != null) { // this level find only one
                return false
            } else if (posX != null && posY != null) {
                return posX.first == posY.first && posX.second != posY.second // same level, different parent
            }
            depth++
        }
        return false
    }

    fun isSubPath(head: ListCode.ListNode?, root: TreeNode?): Boolean {

        fun search(listNode: ListCode.ListNode?, treeNode: TreeNode?): Boolean {
            if (listNode == null) return true
            if (listNode?.`val` != treeNode?.`val`) {
                return false
            }
            val found = search(listNode.next, treeNode?.left) || search(listNode.next, treeNode?.right)
            return found
        }

        val queue = LinkedList<TreeNode>()
        queue.offer(root)
        while (queue.isNotEmpty()) {
            val node = queue.poll()!!
            if (search(head, node)) {
                return true
            }
            node.left?.let { queue.offer(it) }
            node.right?.let { queue.offer(it) }
        }
        return false
    }

    fun findModeInDuplicatesBST(root: TreeNode?): IntArray {
        // mode is the most frequent value
        val result = mutableListOf<Int>()
        var maxFreq = 0
        var previous = -100001
        var count = 1

        fun dfs(node: TreeNode?) {
            if (node == null) return
            dfs(node.left)

            if (node.`val` == previous) {
                count++
            } else {
                if (previous != -100001) {
                    if (count == maxFreq) {
                        result.add(previous)
                    } else if (count > maxFreq) {
                        maxFreq = count
                        result.clear()
                        result.add(previous)
                    }
                }
                count = 1
            }
            previous = node.`val`

            dfs(node.right)
        }

        dfs(root)

        if (count == maxFreq) {
            result.add(previous)
        } else if (count > maxFreq) {
            maxFreq = count
            result.clear()
            result.add(previous)
        }

        return result.toIntArray()
    }

    fun findKthLexicographicallyNumber(n: Int, k: Int): Int {

        fun countSteps(curr: Long, next: Long): Int {
            var steps = 0
            var first = curr
            var last = next
            while (first <= n) {
                steps += minOf(n + 1, last.toInt()) - first.toInt()
                first *= 10
                last *= 10
            }
            return steps
        }

        var position = 1
        var num = 1L

        while (position < k) {
            val steps = countSteps(num, num + 1)
            if (position + steps <= k) {
                num += 1
                position += steps
            } else {
                num *= 10
                position++
            }
        }
        return num.toInt()
    }

    fun replaceValueWithCousinSum(root: TreeNode?): TreeNode? {
        if (root == null) return root
        val queue = LinkedList<TreeNode>()
        queue.offer(root)
        root.`val` = 0
        while (queue.isNotEmpty()) {
            val parents = LinkedList(queue)
            val levelSize = queue.size
            var nextSum = 0
            for (i in 0 until levelSize) {
                val node = queue.poll()!!
                node.left?.let { nextSum += it.`val` }
                node.right?.let { nextSum += it.`val` }
            }
            for (node in parents) {
                var cousinSum = nextSum
                node.left?.let {
                    cousinSum -= it.`val`
                    queue.offer(it)
                }
                node.right?.let {
                    cousinSum -= it.`val`
                    queue.offer(it)
                }
                node.left?.`val` = cousinSum
                node.right?.`val` = cousinSum
            }
        }
        return root
    }

    fun flipEquiv(root1: TreeNode?, root2: TreeNode?): Boolean {
        if (root1 == null && root2 == null) return true
        if (root1 == null || root2 == null || root1.`val` != root2.`val`) return false

        // Check if left and right children are in the same position
        val samePos = flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)

        // Check if left and right children are flipped
        val flippedPos = flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left)

        return samePos || flippedPos
    }

    fun treeQueries(root: TreeNode?, queries: IntArray): IntArray {
        val treeHeight = IntArray(100001)
        val neighboursDepth = IntArray(100001)

        fun height(node: TreeNode?): Int {
            if (node == null) return -1
            if (node.left == null && node.right == null) {
                treeHeight[node.`val`] = 0
                return 0
            }
            val leftHeight = height(node.left)
            val rightHeight = height(node.right)
            val height = maxOf(leftHeight, rightHeight) + 1
            treeHeight[node.`val`] = height
            return height
        }

        height(root)

        var depth = 0
        val queue = LinkedList<TreeNode>()
        queue.offer(root)
        while (queue.isNotEmpty()) {
            val copyOfNodes = LinkedList(queue)
            val levelSize = queue.size
            var firstHeight = -1
            var secondHeight = -1
            for (i in 0 until levelSize) {
                val node = queue.poll()!!
                if (treeHeight[node.`val`] > firstHeight) {
                    secondHeight = firstHeight
                    firstHeight = treeHeight[node.`val`]
                } else if (treeHeight[node.`val`] > secondHeight) {
                    secondHeight = treeHeight[node.`val`]
                }
            }

            for (node in copyOfNodes) {
                if (treeHeight[node.`val`] == firstHeight) {
                    neighboursDepth[node.`val`] = secondHeight + depth
                } else {
                    neighboursDepth[node.`val`] = firstHeight + depth
                }
                node.left?.let { queue.offer(it) }
                node.right?.let { queue.offer(it) }
            }
            depth++
        }

        return IntArray(queries.size) { i ->
            neighboursDepth[queries[i]]
        }
    }

    class BSTIterator(root: TreeNode?) {
        val stack = LinkedList<TreeNode>()

        private fun pushLeftNodes(node: TreeNode?) {
            var current = node
            while (current != null) {
                stack.push(current)
                current = current.left
            }
        }

        init {
            pushLeftNodes(root)
        }

        fun next(): Int {
            val node = stack.pop()!!
            pushLeftNodes(node.right) // average O(1) because any node only pushed and popped once
            return node.`val`
        }

        fun hasNext(): Boolean {
            return stack.isNotEmpty()
        }

    }


    fun reverseOddLevels(root: TreeNode?): TreeNode? {
        // BFS
//        if (root == null) return root
//        val queue = LinkedList<TreeNode>()
//        var reverse = false
//        val nodeValues = LinkedList<Int>()
//        queue.offer(root)
//        while (queue.isNotEmpty()) {
//            val size = queue.size
//            for (i in 0 until size) {
//                val current = queue.poll()!!
//                if (reverse) {
//                    current.`val` = nodeValues.pop()
//                }
//                current.left?.let {
//                    queue.offer(it)
//                    if (!reverse) nodeValues.push(it.`val`)
//                }
//                current.right?.let {
//                    queue.offer(it)
//                    if (!reverse) nodeValues.push(it.`val`)
//                }
//            }
//            reverse = !reverse
//        }
//        return root

        fun dfs(left: TreeNode?, right: TreeNode?, depth: Int) {
            if (left == null || right == null) return
            if (depth % 2 == 1) {
                val tmp = right.`val`
                right.`val` = left.`val`
                left.`val` = tmp
            }
            dfs(left.left, right.right, depth + 1)
            dfs(left.right, right.left, depth + 1)
        }

        dfs(root!!.left, root.right, 1)
        return root
    }

    fun minimumSwapToSortLevel(root: TreeNode?): Int {

        fun calculateSwaps(nums: IntArray) : Int {
            val visited = BooleanArray(nums.size)
            val sorted = nums.withIndex().sortedBy { it.value }
            var count = 0

            for (i in sorted.indices) {
                if (visited[i]) continue
                visited[i] = true
                var j = sorted[i].index
                while (!visited[j]) {
                    count++
                    visited[j] = true
                    j = sorted[j].index
                }
            }
            return count
        }

        if (root == null) return 0
        val queue = LinkedList<TreeNode>()
        queue.offer(root)
        var result = 0
        while (queue.isNotEmpty()) {
            val size = queue.size
            val nums = IntArray(size)
            for (i in 0 until size) {
                val node = queue.poll()!!
                nums[i] = node.`val`
                node.left?.let { queue.offer(it) }
                node.right?.let { queue.offer(it) }
            }
            result += calculateSwaps(nums)
        }
        return result
    }

    fun largestValuesOfLevel(root: TreeNode?): List<Int> {
        if (root == null) return listOf()
        val result = mutableListOf<Int>()

        fun dfs(node: TreeNode?, level: Int) {
            if (node == null) return
            if (result.size == level) {
                result.add(node.`val`)
            } else {
                result[level] = maxOf(result[level], node.`val`)
            }
            dfs(node.left, level + 1)
            dfs(node.right, level + 1)
        }

        dfs(root, 0)
        return result
    }

    fun maxPathSum(root: TreeNode?): Int {
        var maxSum = Int.MIN_VALUE

        fun calculateTree(node: TreeNode?): Int {
            if (node == null) return 0
            maxSum = maxOf(maxSum, node.`val`)
            val leftSum = maxOf(0, calculateTree(node.left))
            val rightSum = maxOf(0, calculateTree(node.right))
            maxSum = maxOf(maxSum, node.`val` + leftSum + rightSum)
            return node.`val` + maxOf(leftSum, rightSum)
        }

        calculateTree(root)
        return maxSum
    }
}