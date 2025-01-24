package com.example.codetoolsnew

import com.example.codetools.ArrayCode
import com.example.codetools.hard.HardArrayCode
import org.junit.Test

import org.junit.Assert.*

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
//        val tree1 = TreeCode.TreeNode(1).apply {
//            left = TreeCode.TreeNode(4).apply {
//                left = TreeCode.TreeNode(7).apply {
//                }
//                right = TreeCode.TreeNode(6).apply {
//                }
//            }
//            right = TreeCode.TreeNode(3).apply {
//                left = TreeCode.TreeNode(8).apply {
//                    left = TreeCode.TreeNode(9)
//                }
//                right = TreeCode.TreeNode(5).apply {
//                    left = TreeCode.TreeNode(10)
//                }
//            }
//        }
//        println(TreeCode.largestValues(tree1))

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
//            HardArrayCode.minCostToReachLast(
//                parseToIntArray("[[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2]]")
//            )
//        )
        println(
            ArrayCode.maxSubarraySumCircular(intArrayOf(-3,-2,-3))
        )

//        println(
//            BitCode.maxEqualRowsAfterFlips(parseToArray("[[0,0,0],[0,0,1],[1,1,0]]"))
//        )

//        println(HardStringCode.allSamePrefixSuffix("aaabaaab"))
//        println(
//            StringCode.countPalindromicSubsequence("bbcbaba")
//        )

//        println(MathCode.fractionAddition("-1/3-1/2"))

//        println(
//            GraphCode.findChampion(4, parseToIntArray("[[0,2],[1,3],[1,2]]")
//            )
//        )

//        println(MatrixCode.totalNQueens(4))
    }
}