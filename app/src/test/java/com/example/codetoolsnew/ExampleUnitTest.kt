package com.example.codetoolsnew

import com.example.codetools.ArrayCode
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class ExampleUnitTest {
    @Test
    fun addition_isCorrect() {
        assertEquals(4, 2 + 2)
    }

    private fun parseToIntArray(input: String): Array<IntArray> {
        // 去除首尾的方括号
        val trimmedInput = input.trim().removePrefix("[").removeSuffix("]")
        // 按每一行进行分割
        return trimmedInput.split("],[").map { row ->
            // 处理每一行，去除方括号，并将字符串分割为数字
            row.replace("[", "").replace("]", "")
                .split(",").map { it.trim().toInt() }
                .toIntArray() // 转为 IntArray
        }.toTypedArray() // 转为 Array<IntArray>
    }

    private fun parseToCharArray(input: String): Array<CharArray> {
        // 去除首尾的方括号
        val trimmedInput = input.trim().removePrefix("[").removeSuffix("]")
        // 按每一行进行分割
        return trimmedInput.split("],[").map { row ->
            // 处理每一行，去除方括号，并将字符串分割为数字
            row.replace("[", "").replace("]", "")
                .split(",").map { it.trim()[0] }
                .toCharArray()
        }.toTypedArray()
    }

    private fun parseToList(input: String): List<List<Int>> {
        // 去除首尾的方括号
        val trimmedInput = input.trim().removePrefix("[").removeSuffix("]")
        // 按每一行进行分割
        return trimmedInput.split("],[").map { row ->
            // 处理每一行，去除方括号，并将字符串分割为数字
            row.replace("[", "").replace("]", "")
                .split(",").map { it.trim().toInt() }
                .toList()
        }.toList()
    }

    @Test
    fun test() {
//        val tree1 = TreeCode.TreeNode(3).apply {
//            left = TreeCode.TreeNode(5).apply {
//                left = TreeCode.TreeNode(6).apply {
//                }
//                right = TreeCode.TreeNode(2).apply {
//                    left = TreeCode.TreeNode(7).apply {
//                    }
//                    right = TreeCode.TreeNode(4).apply {
//                    }
//                }
//            }
//            right = TreeCode.TreeNode(1).apply {
//                left = TreeCode.TreeNode(0).apply {
//                }
//                right = TreeCode.TreeNode(8).apply {
//                }
//            }
//        }
//        println(TreeCode.lcaDeepestLeaves(tree1))

//
//        val head = ListCode.ListNode(4).apply {
//            next = ListCode.ListNode(2).apply {
//                next = ListCode.ListNode(1).apply {
//                    next = ListCode.ListNode(3).apply {
//                    }
//                }
//            }
//        }
//        println(ListCode.sortList(head).toIntArray().joinToString())
//

//        println(
//            HardArrayCode.lenOfVDiagonal(
//                parseToIntArray("[[1]]")
////                parseToIntArray("[[2,2,1,2,2],[2,0,2,2,0],[2,0,1,1,0],[1,0,2,2,2],[2,0,0,2,2]]")
//            )
//        )
        println(
            ArrayCode.makeTheIntegerZero(3, -2)
        )

//        println(
//            TreeCode.findKthLexicographicallyNumber(1000, 990)
//        )

//        println(
//            BitCode.kthCharacter(101)
//        )

//        println(HardStringCode.longestSubsequenceRepeatedK("letsleetcode", 2))
//        println(
//            StringCode.longestSubsequence("0001101010", 5)
//        )
//        println(MathCode.checkArithmeticSubarrays(intArrayOf(4,6,5,9,3,7), intArrayOf(0,0,2), intArrayOf(2,3,5)))

//        println(
//            GraphCode.maxWeight(4, parseToIntArray("[[0,1,4],[0,2,3],[1,2,9],[2,3,5],[0,3,5]]"), 2, 11)
//        )

//        println(MatrixCode.findDiagonalOrder(parseToIntArray("[[1,2,3],[4,5,6],[7,8,9]]\n")))

//        println(
//            SortCode.radixSort(intArrayOf(170, 45, 75, 90, 802, 24, 2, 66))
//        )

//        val mutex = Mutex()
//        runBlocking {
//            var count = 0
//            withContext(Dispatchers.Default) {
//                repeat(1000) {
//                    launch {
//                        count++
//                    }
//                }
//            }
//            delay(1000)
//            println("final count $count")
//            count = 0
//            withContext(Dispatchers.Default) {
//                repeat(1000) {
//                    launch {
//                        mutex.withLock {
//                            count++
//                        }
//                    }
//                }
//            }
//            delay(1000)
//            println("final count $count")
//            count = 0
//            repeat(1000) {
//                launch {
//                    count++
//                }
//            }
//            delay(1000)
//            println("final count $count")
//        }
    }
}