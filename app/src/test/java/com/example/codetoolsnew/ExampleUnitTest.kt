package com.example.codetoolsnew

import com.example.codetools.ArrayCode
import com.example.codetools.MathCode
import com.example.codetools.StringCode
import com.example.codetools.TreeCode
import com.example.codetools.hard.HardArrayCode
import com.example.codetools.hard.HardStringCode
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
//            HardArrayCode.largestPathValue("hhqhuqhqff", parseToIntArray("[[0,1],[0,2],[2,3],[3,4],[3,5],[5,6],[2,7],[6,7],[7,8],[3,8],[5,8],[8,9],[3,9],[6,9]]"))
//        )
//        println(
//            ArrayCode.minimizeMaxDiffInPairs(intArrayOf(1,3,4,5), 2)
//        )

//        println(
//            TreeCode.findKthLexicographicallyNumber(1000, 990)
//        )

//        println(
//            BitCode.maxEqualRowsAfterFlips(parseToArray("[[0,0,0],[0,0,1],[1,1,0]]"))
//        )

//        println(HardStringCode.maxDifference("1122211", 3))
//        println(
//            StringCode.removeOccurrences("daabcbaabcbc", "abc")
//        )
        println(MathCode.minMaxDifference(98749211))

//        println(
//            GraphCode.maxWeight(4, parseToIntArray("[[0,1,4],[0,2,3],[1,2,9],[2,3,5],[0,3,5]]"), 2, 11)
//        )

//        println(MatrixCode.sortMatrix(parseToIntArray("[[1,7,3],[9,8,2],[4,5,6]]")))
    }
}