package com.example.codetoolsnew

import com.example.codetools.ArrayCode
import com.example.codetools.MathCode
import com.example.codetools.MatrixCode
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
//        val tree1 = TreeCode.TreeNode(1).apply {
//            left = TreeCode.TreeNode(2).apply {
//            }
//            right = TreeCode.TreeNode(3).apply {
//            }
//        }
//        println(TreeCode.maxPathSum(tree1))

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
//            HardArrayCode.largestIsland(
//                parseToIntArray("[[1,1,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]]")
//            )
//        )
        println(
            ArrayCode.findDifferentBinaryString(arrayOf("00","01"))
        )

//        println(
//            BitCode.maxEqualRowsAfterFlips(parseToArray("[[0,0,0],[0,0,1],[1,1,0]]"))
//        )

//        println(HardStringCode.shortestMatchingSubstring("madlogic", "*adlogi*"))
//        println(
//            StringCode.removeOccurrences("daabcbaabcbc", "abc")
//        )

//        println(MathCode.separateSquares(parseToIntArray("[[522261215,954313664,461744743],[628661372,718610752,21844764],[619734768,941310679,91724451],[352367502,656774918,591943726],[860247066,905800565,853111524],[817098516,868361139,817623995],[580894327,654069233,691552059],[182377086,256660052,911357],[151104008,908768329,890809906],[983970552,992192635,462847045]]")))

//        println(
//            GraphCode.findChampion(4, parseToIntArray("[[0,2],[1,3],[1,2]]")
//            )
//        )

//        println(MatrixCode.sortMatrix(parseToIntArray("[[1,7,3],[9,8,2],[4,5,6]]")))
    }
}