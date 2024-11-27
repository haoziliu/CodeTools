package com.example.codetools

import java.util.*

object ListCode {

    class ListNode(var `val`: Int) {
        var next: ListNode? = null
        override fun toString(): String {
            return this.toIntArray().joinToString()
        }
    }

    fun ListNode?.toIntArray(): Array<Int?> {
        val result = mutableListOf<Int?>()
        if (this == null) return result.toTypedArray()

        val queue: Queue<ListNode?> = LinkedList()
        queue.offer(this)

        while (queue.isNotEmpty()) {
            val node = queue.poll()
            if (node == null) {
//                result.add(null)
            } else {
                result.add(node.`val`)
                queue.offer(node.next)
            }
        }

        return result.toTypedArray()
    }

    fun mergeTwoLists(list1: ListNode?, list2: ListNode?): ListNode? {
//        if (list1 == null) return list2
//        if (list2 == null) return list1
//
//        var head: ListNode? = null
//        var tail: ListNode? = null
//        if (list1.`val` < list2.`val`) {
//            head = list1
//            tail = mergeTwoLists(list1.next, list2)
//        } else {
//            head = list2
//            tail = mergeTwoLists(list1, list2.next)
//        }
//        head.next = tail
//        return head

//        var p1 = list1
//        var p2 = list2
//
//        var current = ListNode(Int.MIN_VALUE)
//        var head: ListNode? = null
//
//        while (p1 != null && p2 != null) {
//            if (p1.`val` < p2.`val`) {
//                current.next = p1
//                p1 = p1.next
//            } else {
//                current.next = p2
//                p2 = p2.next
//            }
//            current = current.next!!
//            if (head == null) {
//                head = current
//            }
//        }
//        if (p1 != null) {
//            current.next = p1
//        }
//        if (p2 != null) {
//            current.next = p2
//        }
//        return head

        return if (list1 == null) {
            list2
        } else if (list2 == null) {
            list1
        } else if (list1.`val` < list2.`val`) {
            list1.next = mergeTwoLists(list1.next, list2)
            list1
        } else {
            list2.next = mergeTwoLists(list1, list2.next)
            list2
        }
    }

    fun removeElements(head: ListNode?, `val`: Int): ListNode? {
//        var p = head
//        while (p?.`val` == `val`) {
//            p = p.next
//        }
//        val result = p
//        while (p?.next != null) {
//            if (p.next?.`val` == `val`) {
//                p.next = p.next?.next
//            } else {
//                p = p.next
//            }
//        }
//        return result

//        val dummy = ListNode(0)
//        dummy.next = head
//        var p: ListNode? = dummy
//        while (p?.next != null) {
//            if (p.next?.`val` == `val`) {
//                p.next = p.next?.next
//            } else {
//                p = p.next
//            }
//        }
//        return dummy.next

        if (head == null) {
            // Base case: If the list is empty, return null.
            return null
        }

        // Recursively process the rest of the list.
        head.next = removeElements(head.next, `val`)

        // Check if the current node's value matches the specified value.
        return if (head.`val` == `val`) {
            // If it does, skip this node and return the next node.
            head.next
        } else {
            // Otherwise, keep this node and return it.
            head
        }
    }

    fun deleteDuplicatesLeavingOne(head: ListNode?): ListNode? {
        var curr = head
        while (curr?.next != null) {
            if (curr.next?.`val` == curr.`val`) {
                curr.next = curr.next?.next
            } else {
                curr = curr.next
            }
        }
        return head
    }

    fun removeZeroSumSublists(head: ListNode?): ListNode? {
        val dummy = ListNode(-1).apply { next = head }
        var curr = head
        val prefixSumToNodeMap = HashMap<Int, ListNode>()
        var sum = 0
        prefixSumToNodeMap[0] = dummy
        while (curr != null) {
            sum += curr.`val`
            if (prefixSumToNodeMap.containsKey(sum)) {
                // Remove entries from the map between the previous node and the current node
                var temp = prefixSumToNodeMap[sum]!!.next
                var tempSum = sum + temp!!.`val`
                while (temp != curr) {
                    prefixSumToNodeMap.remove(tempSum)
                    temp = temp!!.next
                    tempSum += temp!!.`val`
                }

                prefixSumToNodeMap[sum]!!.next = curr.next
            } else {
                prefixSumToNodeMap[sum] = curr
            }
            curr = curr.next
        }
        return dummy.next
    }

    fun mergeInBetween(list1: ListNode?, a: Int, b: Int, list2: ListNode?): ListNode? {
        if (list1 == null || list2 == null) return null
        var curr = list1
        repeat(a - 1) {
            curr = curr!!.next
        }
        var original = curr!!.next
        repeat(b - a) {
            original = original!!.next
        }
        curr!!.next = list2
        while (curr!!.next != null) {
            curr = curr!!.next
        }
        curr!!.next = original!!.next
        return list1
    }

    // reverseList:
    //        // recursively
//        // Base case: If head is null or it's the last node, return head
//        if (head?.next == null) {
//            return head
//        }
//        // Recursively reverse the sublist starting from head.next
//        val reversedHead = reverseList(head.next)
//        // Reverse the direction of the edge between head and head.next
//        head.next!!.next = head
//        // Break the link to the next node to prevent cycles
//        head.next = null
//        return reversedHead

    // stack
//        if (head == null) return null
//        val nodeStack = Stack<ListNode>()
//        nodeStack.push(head)
//        while (nodeStack.peek().next != null) {
//            // do we need to break list before push?
//            nodeStack.push(nodeStack.peek().next)
//        }
//        val tail = nodeStack.peek()
//        while (nodeStack.isNotEmpty()) {
//            nodeStack.pop().apply {
//                next = if (nodeStack.isNotEmpty()) nodeStack.peek() else null
//            }
//        }
//        return tail

    fun reverseList(head: ListNode?): ListNode? {
        // iterative: link each node backward
        var back: ListNode? = null
        var curr = head
        var next: ListNode?
        while (curr != null) {
            // temporarily store next
            next = curr.next
            // link current node backward
            curr.next = back
            // move back head
            back = curr
            // go next
            curr = next
        }
        return back
    }

    fun isPalindrome(head: ListNode?): Boolean {
//        if (head == null) return false
//        val nodeStack = Stack<Int>()
//        nodeStack.push(head.`val`)
//        var curr = head.next
//        while (curr != null) {
//            nodeStack.push(curr.`val`)
//            curr = curr.next
//        }
//        curr = head
//        while (nodeStack.isNotEmpty()) {
//            if (nodeStack.pop() != curr?.`val`) return false
//            curr = curr?.next
//        }
//        return true

        // two pointers find half. Reverse the second half.
        if (head == null) return false
        var p1 = head
        var p2 = head
        while (p2 != null) {
            p1 = p1?.next
            p2 = p2.next?.next
        }
        p2 = reverseList(p1)
        p1 = head
        while (p1 != null && p2 != null) {
            if (p1.`val` != p2.`val`) return false
            p1 = p1.next
            p2 = p2.next
        }
        return true
    }

    fun addTwoNumbers(l1: ListNode?, l2: ListNode?): ListNode? {
        var curr1 = l1
        var curr2 = l2
        val result = ListNode(0)
        var sum = 0
        var currResult = result
        while (curr1 != null || curr2 != null) {
            sum = currResult.`val` + (curr1?.`val` ?: 0) + (curr2?.`val` ?: 0)
            currResult.`val` = sum % 10
            curr1 = curr1?.next
            curr2 = curr2?.next
            if (curr1 != null || curr2 != null || sum >= 10) {
                currResult.next = ListNode(if (sum >= 10) 1 else 0)
                currResult = currResult.next!!
            }
        }
        return result

        // use l1 to return
//        var carry = 0
//        var curr1 = l1
//        var curr2 = l2
//        var prev: ListNode? = null // Pointer to keep track of the previous node in l1
//        while (curr1 != null || curr2 != null || carry != 0) {
//            val sum = (carry + (curr1?.`val` ?: 0) + (curr2?.`val` ?: 0)) // Calculate sum
//            curr1?.`val` = sum % 10 // Update value of current node in l1
//            carry = sum / 10 // Update carry
//            prev = curr1 // Move prev pointer
//            curr1 = curr1?.next // Move curr1 pointer
//            curr2 = curr2?.next // Move curr2 pointer
//            if (curr1 == null && (curr2 != null || carry != 0)) {
//                prev?.next = ListNode(0) // Append a new node to l1 if necessary
//                curr1 = prev?.next // Move curr1 pointer to the newly appended node
//            }
//        }
//        return l1 // Return the modified l1 as the result
    }

    fun reorderList(head: ListNode?): Unit {
        //L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
        // find half, reverse second half, merge
        if (head == null) return
        var p1 = head
        var p2 = head
        var tmp: ListNode? = null
        while (p2 != null) {
            tmp = p1
            p1 = p1?.next
            p2 = p2.next?.next
        }
        tmp?.next = null // break first half
        p2 = reverseList(p1)
        p1 = head

        var tmp2: ListNode? = null
        while (p1 != null) {
            tmp = p1.next
            tmp2 = p2?.next
            p1.next = p2
            p2?.next = tmp
            p1 = tmp
            p2 = tmp2
        }
        println(head)
    }

    fun deleteNode(node: ListNode?) {
        // swap one by one to the end
//        var current: ListNode? = node
//        while (current?.next != null) {
//            current.`val` = current.next!!.`val`
//            if (current.next!!.next == null) {
//                break
//            } else {
//                current = current.next
//            }
//        }
//        current?.next = null

        node?.`val` = node?.next!!.`val`
        node.next = node.next!!.next
    }

    fun removeNodesIfGreaterAtRight(head: ListNode?): ListNode? {
        // Remove every node which has a node with a greater value anywhere to the right side
        // [5,2,13,3,8] -> [13,8]
//        val dummy = ListNode(Int.MAX_VALUE)
//        dummy.next = head
//        var highestNode = head
//        var current = head
//        var previous: ListNode? = dummy
//        while (current != null) {
//            if (current.`val` > highestNode!!.`val`) {
//                // if current is highest, change head
//                dummy.next = current
//                highestNode = current
//            } else if (current.`val` > previous!!.`val`) {
//                // if current bigger than previous, seek down from highest until bigger than current
//                previous = highestNode
//                while (previous?.next != null && previous.next!!.`val` >= current.`val`) {
//                    previous = previous.next
//                }
//                previous?.next = current
//            }
//            previous = current
//            current = current.next
//        }
//        return dummy.next

        val reversedHead = reverseList(head)
        var current = reversedHead
        while (current?.next != null) {
            if (current.next!!.`val` < current.`val`) {
                current.next = current.next!!.next
            } else {
                current = current.next
            }
        }
        current?.next = null
        return reverseList(reversedHead)
    }

    fun doubleIt(head: ListNode?): ListNode? {
        val reversedHead = reverseList(head)
        var current = reversedHead
        var carry = 0
        var value = 0
        while (current != null) {
            value = current.`val` + current.`val` + carry
            carry = if (value >= 10) 1 else 0
            current.`val` = value % 10
            current = current.next
        }
        return if (carry == 1) ListNode(1).apply { next = reverseList(reversedHead) }
        else reverseList(reversedHead)
    }

    class Node(var `val`: Int) {
        var next: Node? = null
        var random: Node? = null
    }

    fun copyRandomList(node: Node?): Node? {
        if (node == null) return null
        val originalToNewMap = mutableMapOf<Node?, Node?>()
        val copiedHead = Node(node.`val`)
        originalToNewMap[node] = copiedHead
        var original: Node? = node.next
        var copied: Node? = copiedHead
        while (original != null) {
            copied?.next = Node(original.`val`)
            originalToNewMap[original] = copied?.next
            copied = copied?.next
            original = original.next
        }
        original = node
        copied = copiedHead
        while (original != null) {
            if (original.random == null) {
                copied?.random = null
            } else {
                copied?.random = originalToNewMap[original.random]
            }
            copied = copied?.next
            original = original.next
        }
        return copiedHead
    }

    fun removeNthFromEnd(head: ListNode?, n: Int): ListNode? {
//        var length = 0
//        var curr = head
//        while (curr != null) {
//            length++
//            curr = curr.next
//        }
//        val dummy = ListNode(-1).apply { next = head }
//        curr = dummy
//        repeat(length - n) {
//            curr = curr?.next
//        }
//        curr!!.next = curr!!.next?.next
//        return dummy.next
        val map = mutableMapOf<Int, ListNode>()
        var index = 0
        val dummy = ListNode(-1).apply { next = head }
        var curr: ListNode? = dummy
        while (curr != null) {
            map[index] = curr
            index++
            curr = curr.next
        }
        map[index - n - 1]?.next = map[index - n]?.next
        return dummy.next
    }

    fun rotateRight(head: ListNode?, k: Int): ListNode? {
        //        val map = mutableMapOf<Int, ListNode>()
        //        var index = 0
        //        var current = head
        //        var size = 0
        //        while (current != null) {
        //            map[index++] = current
        //            current = current.next
        //            size++
        //        }
        //        if (size == 0 || size == 1 || k == 0 || k % size == 0) return head
        //        map[size - k % size - 1]?.next = null
        //        map[size - 1]?.next = head
        //        return map[size - k % size]
        if (k == 0) return head
        var current = head
        var size = 0
        while (current != null) {
            current = current.next
            size++
        }
        if (size == 0 || size == 1 || k % size == 0) return head
        current = head
        var index = 0
        var result: ListNode? = null
        while (current != null) {
            val tmp = current.next
            if (index == size - k % size - 1) {
                current.next = null
            }
            if (index == size - k % size) {
                result = current
            }
            if (index == size - 1) {
                current.next = head
            }
            current = tmp
            index++
        }
        return result
    }

    fun mergeNodes(head: ListNode?): ListNode? {
        val result = head!!.next
        var current = head.next
        while (current != null) {
            if (current.next?.`val` != 0) {
                current.`val` += current.next!!.`val`
                current.next = current.next!!.next
            } else {
                current.next = current.next!!.next
                current = current.next
            }
        }
        return result
    }

    fun hasCycle(head: ListNode?): Boolean {
        if (head == null) return false
        // Floyd's cycle
        var slow = head
        var fast = head.next
        while (fast?.next != null) {
            if (slow == fast) return true
            slow = slow?.next
            fast = fast.next?.next
        }
        return false
    }

    fun nodesBetweenCriticalPoints(head: ListNode?): IntArray {
        if (head == null) return intArrayOf(-1, -1)
        val criticalIndices = mutableListOf<Int>()
        var current: ListNode? = head.next
        var previous: ListNode = head
        var index = 1
        var minDistance = -1
        var maxDistance = -1
        while (current != null) {
            if (current.`val` > previous.`val` && current.next != null && current.`val` > current.next!!.`val` ||
                current.`val` < previous.`val` && current.next != null && current.`val` < current.next!!.`val`
            ) {
                if (criticalIndices.isNotEmpty()) {
                    minDistance = if (minDistance == -1) {
                        index - criticalIndices.last()
                    } else {
                        minOf(minDistance, index - criticalIndices.last())
                    }
                }
                criticalIndices.add(index)
            }
            previous = current
            current = current.next
            index++
        }
        if (criticalIndices.size > 1) {
            maxDistance = criticalIndices.last() - criticalIndices.first()
        }
        return intArrayOf(minDistance, maxDistance)
    }

    fun reverseBetween(head: ListNode?, left: Int, right: Int): ListNode? {
        if (left == right) return head
        var index = 1
        var current = head
        var previous: ListNode? = null
        var beforeTail: ListNode? = null
        var reversedHead = head
        var reversedTail = head

        while (current != null && index < right + 1) {
            val tmpNext = current.next
            when (index) {
                left - 1 -> {
                    beforeTail = current
                }

                left -> {
                    reversedTail = current
                    reversedTail.next = null
                }

                in left + 1..right -> {
                    current.next = previous
                    if (index == right) {
                        reversedHead = current
                    }
                }
            }
            previous = current
            current = tmpNext
            index++
        }
        beforeTail?.next = reversedHead
        reversedTail?.next = current
        if (left == 1) return reversedHead
        return head
    }


    fun partition(head: ListNode?, x: Int): ListNode? {
        if (head == null) return head
        val firstDummy = ListNode(-1)
        val secondDummy = ListNode(-1)
        var i: ListNode? = firstDummy
        var j: ListNode? = secondDummy
        var current = head
        while (current != null) {
            if (current.`val` < x) {
                i?.next = current
                i = i?.next
            } else {
                j?.next = current
                j = j?.next
            }
            current = current.next
        }
        i?.next = secondDummy.next
        j?.next = null
        return firstDummy.next
    }

    class LRUCache(val capacity: Int) {
        class LRUNode(var key: Int, var `val`: Int) {
            var previous: LRUNode? = null
            var next: LRUNode? = null
        }

        val map = mutableMapOf<Int, LRUNode>()
        var head: LRUNode? = null
        var tail: LRUNode? = null
        var count = 0

        fun get(key: Int): Int {
            val node = map[key]
            return if (node == null) {
                -1
            } else {
                updateTail(node)
                node.`val`
            }
        }

        private fun updateTail(node: LRUNode) {
            if (node == tail) return
            if (node == head) {
                head = node.next
                head?.previous = null
            } else {
                node.previous?.next = node.next
                node.next?.previous = node.previous
            }
            node.next = null
            node.previous = tail
            tail?.next = node
            tail = node
            if (head == null) {
                head = tail
            }
        }

        fun put(key: Int, value: Int) {
            val node = map[key]
            if (node != null) {
                node.`val` = value
                updateTail(node)
            } else {
                if (count == capacity) {
                    map.remove(head!!.key)
                    head = head!!.next
                    head?.previous = null
                    count--
                }
                val newNode = LRUNode(key, value)
                if (head == null) {
                    head = newNode
                } else {
                    tail?.next = newNode
                    newNode.previous = tail
                }
                tail = newNode
                map[key] = newNode
                count++
            }
        }
    }


    fun findCircleWinner(n: Int, k: Int): Int {
        // josephus problem
        var result = 0
        for (i in 1..n) {
            result = (result + k) % i
        }
        return result + 1
//        val head = ListNode(1)
//        var previous: ListNode? = head
//        for (i in 2..n) {
//            ListNode(i).let {
//                previous?.next = it
//                previous = it
//            }
//        }
//        previous?.next = head
//        var current: ListNode? = head
//        while (current?.next != current) {
//            repeat(k - 1) {
//                previous = current
//                current = current?.next
//            }
//            previous?.next = current?.next
//            current = current?.next
//        }
//        return current?.`val` ?: -1
    }

    fun deleteDuplicatesIncluding(head: ListNode?): ListNode? {
        val dummy = ListNode(-101).apply { next = head }
        var previous = dummy
        var current = dummy.next
        while (current != null) {
            if (current.`val` == current.next?.`val`) {
                while (current?.`val` == current?.next?.`val`) {
                    current = current?.next // find all current duplicates
                }
                current = current?.next // move to next
                previous.next = current
            } else {
                previous = current
                current = current.next
            }
        }
        return dummy.next
    }

    fun splitListToParts(head: ListNode?, k: Int): Array<ListNode?> {
        val result = Array<ListNode?>(k) { null }
        var current = head
        var total = 0
        while (current != null) {
            total++
            current = current.next
        }
        val partSize = total / k
        val restSize = total % k
        var extraCount = 0
        current = head
        result[0] = head
        var groupIndex = 1
        var currentCount = 0
        while (current != null) {
            currentCount++
            val targetCount = if (extraCount < restSize) {
                partSize + 1
            } else {
                partSize
            }
            if (currentCount == targetCount && current.next != null) {
                result[groupIndex] = current.next
                current.next = null
                current = result[groupIndex]
                extraCount++
                groupIndex++
                currentCount = 0
            } else {
                current = current.next
            }
        }
        return result
    }

    fun oddEvenList(head: ListNode?): ListNode? {
        var oddPointer = head
        var evenPointer = head?.next
        val evenHead = evenPointer
        while (evenPointer?.next != null) {
            oddPointer?.next = evenPointer.next
            oddPointer = oddPointer?.next
            evenPointer.next = oddPointer?.next
            evenPointer = evenPointer.next
        }
        oddPointer?.next = evenHead
        return head
    }

    fun reverseEvenLengthGroups(head: ListNode?): ListNode? {
        if (head == null) return null
        var end = head
        var totalSize = 0
        while (end != null) {
            totalSize++
            end = end.next
        }
        var totalCount = 0
        var groupSize = 1
        var reverseFlag = false
        var start = head
        end = head
        var currentCount = 0
        var previous: ListNode? = null
        var tmp: ListNode? = null
        while (end != null) {
            if (reverseFlag) {
                tmp = end.next
                if (currentCount != 0) {
                    end.next = previous
                } else {
                    end.next = null
                }
                previous = end
                end = tmp

                if (currentCount == groupSize - 1 || end == null) {
                    start?.next?.next = end
                    tmp = start?.next
                    start?.next = previous
                    previous = tmp
                    start = tmp
                }
            } else {
                previous = end
                start = end
                end = end.next
            }

            currentCount++
            totalCount++
            if (currentCount == groupSize) {
                currentCount = 0
                groupSize++
                val rest = totalSize - totalCount
                reverseFlag = if (rest >= groupSize) {
                    // have enough nodes, toggle flag normally
                    !reverseFlag
                } else {
                    // reverse if even
                    rest % 2 == 0
                }
            }
        }
        return head
    }

    fun reverseEvenLengthGroupsExtraSpace(head: ListNode?): ListNode? {
        if (head == null) return null
        var groupSize = 1
        var current = head
        var count = 0
        var lastEven = false
        val list = mutableListOf<ListNode>()
        while (current != null) {
            count++
            if (count == 1) {
                list.add(current)
            }
            if (count == groupSize) {
                val tmp = current.next
                current.next = null
                current = tmp
                if (current != null) {
                    count = 0
                    groupSize++
                }
            } else {
                current = current.next
            }
        }
        lastEven = count % 2 == 0
        for (i in list.indices) {
            if (i == list.lastIndex) {
                if (lastEven) {
                    list[i] = reverseList(list[i])!!
                }
            } else if (i == list.lastIndex && lastEven || i % 2 == 1) {
                list[i] = reverseList(list[i])!!
            }
        }
        for (i in 0 until list.size - 1) {
            current = list[i]
            while (current?.next != null) {
                current = current.next
            }
            current?.next = list[i + 1]
        }
        return head
    }

    fun insertGreatestCommonDivisors(head: ListNode?): ListNode? {
        fun gcd(a: Int, b: Int): Int {
            return if (b == 0) a else gcd(b, a % b)
        }

        var current = head
        while (current?.next != null) {
            val gcd = gcd(current.`val`, current.next!!.`val`)
            val newNode = ListNode(gcd).apply { next = current!!.next }
            current.next = newNode
            current = newNode.next
        }
        return head
    }

    fun findMiddle(head: ListNode?): ListNode? {
        var slow = head
        var fast = head?.next
        while (fast?.next != null) {
            slow = slow?.next
            fast = fast.next?.next
        }
        return slow
    }

    fun quickSortList(head: ListNode?): ListNode? {
        if (head == null) return null

        fun swap(node1: ListNode, node2: ListNode) {
            val tmp = node1.`val`
            node1.`val` = node2.`val`
            node2.`val` = tmp
        }

        fun partition(left: ListNode, rightExclusive: ListNode?): ListNode? {
            var last = left
            var curr = last.next
            while (curr != rightExclusive) {
                if (curr != null && curr.`val` < left.`val`) {
                    swap(curr, last.next!!)
                    last = last.next!!
                }
                curr = curr?.next
            }
            swap(left, last)
            return last
        }

        fun quickSort(left: ListNode, rightExclusive: ListNode?): ListNode {
            if (left != rightExclusive) {
                val pivot = partition(left, rightExclusive)
                quickSort(left, pivot)
                if (pivot?.next != null) {
                    quickSort(pivot.next!!, rightExclusive)
                }
            }
            return left
        }
        quickSort(head, null)
        return head
    }

    fun mergeSortList(head: ListNode?): ListNode? {
        if (head?.next == null) return head

        fun merge(head1: ListNode?, head2: ListNode?): ListNode? {
            val dummy = ListNode(0)
            var curr = dummy
            var left = head1
            var right = head2
            while (left != null && right != null) {
                if (left.`val` < right.`val`) {
                    curr.next = left
                    left = left.next
                } else {
                    curr.next = right
                    right = right.next
                }
                curr = curr.next!!
            }
            curr.next = left ?: right // rest
            return dummy.next
        }

        val middle = findMiddle(head)
        val nextToMiddle = middle?.next
        middle?.next = null
        return merge(mergeSortList(head), mergeSortList(nextToMiddle))
    }
}