package com.example.codetoolsnew

import com.example.codetools.ArrayCode
import com.example.codetools.GraphCode
import com.example.codetools.MatrixCode
import com.example.codetools.StringCode
import com.example.codetools.TreeCode
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
//            left = TreeCode.TreeNode(2).apply {
//                left = TreeCode.TreeNode(4).apply {
//                    left = TreeCode.TreeNode(8)
//                    right = TreeCode.TreeNode(9)
//                }
//                right = TreeCode.TreeNode(5).apply {
//                    left = TreeCode.TreeNode(10)
//                    right = TreeCode.TreeNode(11)
//                }
//            }
//            right = TreeCode.TreeNode(3).apply {
//                left = TreeCode.TreeNode(6).apply {
//                    left = TreeCode.TreeNode(12)
//                    right = TreeCode.TreeNode(13)
//                }
//                right = TreeCode.TreeNode(7).apply {
//                    left = TreeCode.TreeNode(14)
//                    right = TreeCode.TreeNode(15)
//                }
//            }
//        }
//        println(TreeCode.reverseOddLevels(tree1))

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

        println(
            HardArrayCode.maxKDivisibleComponents(7, parseToIntArray("[[2,0],[0,4],[2,5],[1,2],[5,3],[4,6]]"), intArrayOf(12154284,649536765,974051464,821507385,392654193,770357917,37707285), 11)
        )
//        println(
//            ArrayCode.finalPrices(intArrayOf(10,2,5,2,8))
//        )

//        println(
//            BitCode.maxEqualRowsAfterFlips(parseToArray("[[0,0,0],[0,0,1],[1,1,0]]"))
//        )

//        println(HardStringCode.calculate("-(-(2 + 3))"))
//        println(
//            StringCode.repeatLimitedString("mzptvecynjfohbwduxown", 1)
//        )

//        println(MathCode.fractionAddition("-1/3-1/2"))

//        println(
//            GraphCode.findChampion(4, parseToIntArray("[[0,2],[1,3],[1,2]]")
//            )
//        )

//        println(MatrixCode.totalNQueens(4))
    }
}