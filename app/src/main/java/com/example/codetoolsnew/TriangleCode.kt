package com.example.codetools

object TriangleCode {

    fun generate(numRows: Int): List<List<Int>> {
        val result = ArrayList<ArrayList<Int>>(numRows)
        var newLine: ArrayList<Int>? = null
        for (n in 0 until numRows) {
            newLine = ArrayList<Int>(n + 1)
            newLine.add(1)
            for (i in 1 until n) {
                newLine.add(result[result.size - 1][i - 1] + result[result.size - 1][i])
            }
            if (n != 0) {
                newLine.add(1)
            }
            result.add(newLine)
        }
        return result
    }

    fun getRow(rowIndex: Int): List<Int> {
        var newLine = ArrayList<Int>(0)
        var lastLine = ArrayList<Int>(0)
        for (n in 0..rowIndex) {
            newLine = ArrayList<Int>(n + 1)
            newLine.add(1)
            for (i in 1 until n) {
                newLine.add(lastLine[i - 1] + lastLine[i])
            }
            if (n != 0) {
                newLine.add(1)
            }
            if (n == rowIndex) break
            else lastLine = newLine
        }
        return newLine
    }
}