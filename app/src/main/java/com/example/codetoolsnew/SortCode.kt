package com.example.codetools


object SortCode {

    fun bubbleSort(nums: IntArray): IntArray {
        val len = nums.size
        for (i in 0 until len - 1) {
            for (j in 0 until len - 1 - i) {
                if (nums[j] > nums[j + 1]) {
                    val temp = nums[j + 1]
                    nums[j + 1] = nums[j]
                    nums[j] = temp
                }
            }
        }
        return nums
    }

    fun selectionSort(nums: IntArray): IntArray {
        for (i in nums.indices) {
            var minIndex = i
            for (j in i until nums.size) {
                if (nums[j] < nums[minIndex]) {
                    minIndex = j
                }
            }
            val tmp = nums[i]
            nums[i] = nums[minIndex]
            nums[minIndex] = tmp
        }
        return nums
    }

    fun insertionSort(nums: IntArray): IntArray {
        for (i in 1 until nums.size) {
            val current = nums[i]
            var preIndex = i - 1
            while (preIndex >= 0 && current < nums[preIndex]) {
                nums[preIndex + 1] = nums[preIndex]
                preIndex--
            }
            nums[preIndex + 1] = current
        }
        return nums
    }

    fun shellSort(nums: IntArray): IntArray {
        val length = nums.size
        var temp: Int
        var step = length / 2
        while (step >= 1) {
            for (i in step until length) {
                temp = nums[i]
                var preIndex = i - step
                while (preIndex >= 0 && nums[preIndex] > temp) {
                    nums[preIndex + step] = nums[preIndex]
                    preIndex -= step
                }
                nums[preIndex + step] = temp
            }
            step /= 2
        }
        return nums
    }

    fun sortJumbled(mapping: IntArray, nums: IntArray): IntArray {
        var maxNumber = nums.maxOrNull() ?: 0
        var digitCount = 0
        while (maxNumber != 0) {
            digitCount++
            maxNumber /= 10
        }

        fun getMappedDigit(value: Int, digitPosition: Int): Int {
            if (value == 0) return mapping[value]
            var scale = 1
            repeat(digitPosition - 1) {
                scale *= 10
            }
            val digit = value / scale
            return if (digit == 0) 0 else mapping[digit % 10]
        }

        fun radixSort(input: IntArray): IntArray {
            var source = input
            var destination = IntArray(source.size)
            val bucketSize = 10
            val bucket = IntArray(bucketSize)
            var count: Int
            var temp: Int

            for (position in 1..digitCount) {
                bucket.fill(0)
                source.forEach { number ->
                    val digit = getMappedDigit(number, position)
                    bucket[digit]++
                }
                count = 0
                for (i in 0 until bucketSize) {
                    temp = bucket[i]
                    bucket[i] = count
                    count += temp
                }
                source.forEach { number ->
                    val digit = getMappedDigit(number, position)
                    destination[bucket[digit]] = number
                    bucket[digit]++
                }
                val tempArray = source
                source = destination
                destination = tempArray
            }
            return source
        }

        return radixSort(nums)
    }


    fun mergeSort(nums: IntArray): IntArray {
        fun merge(arr1: IntArray, arr2: IntArray): IntArray {
            val length1 = arr1.size
            val length2 = arr2.size
            val result = IntArray(length1 + length2)
            var i = 0
            var j = 0
            while (i < length1 && j < length2) {
                result[i + j] = if (arr1[i] < arr2[j]) {
                    arr1[i++]
                } else {
                    arr2[j++]
                }
            }
            while (i < length1) {
                result[i + j] = arr1[i++]
            }
            while (j < length2) {
                result[i + j] = arr2[j++]
            }
            return result
        }

        val length = nums.size
        if (length < 2) {
            return nums
        }
        val middle = length / 2
        val left = nums.sliceArray(IntRange(0, middle - 1))
        val right = nums.sliceArray(IntRange(middle, length - 1))
        return merge(mergeSort(left), mergeSort(right))
    }

    fun quickSort(nums: IntArray): IntArray {
        fun swap(arr: IntArray, i: Int, j: Int) {
            val temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
        }

        fun partition(arr: IntArray, left: Int, right: Int): Int {
            var index = left + 1 // index is the first bigger than pivot
            for (i in index..right) {
                if (arr[i] < arr[left]) {
                    swap(arr, i, index)
                    index++
                }
            }
            swap(arr, left, index - 1)
            return index - 1
        }

        fun quickSort(arr: IntArray, left: Int, right: Int): IntArray {
            if (left < right) {
                val partitionIndex = partition(arr, left, right)
                quickSort(arr, left, partitionIndex - 1)
                quickSort(arr, partitionIndex + 1, right)
            }
            return arr
        }

        return quickSort(nums, 0, nums.lastIndex)
    }

    fun heapSort(arr: IntArray, start: Int, end: Int) {
        // 计算子数组长度
        val n = end - start + 1

        fun heapify(arr: IntArray, n: Int, i: Int, start: Int, end: Int) {
            val left = 2 * (i - start) + 1 + start
            val right = 2 * (i - start) + 2 + start
            var largest = i

            // 检查左子节点是否存在且大于当前节点
            if (left <= end && arr[left] > arr[largest]) {
                largest = left
            }

            // 检查右子节点是否存在且大于当前最大值
            if (right <= end && arr[right] > arr[largest]) {
                largest = right
            }

            // 如果需要交换
            if (largest != i) {
                arr[i] = arr[largest].also { arr[largest] = arr[i] }

                // 递归调整子堆
                heapify(arr, n, largest, start, end)
            }
        }

        // 构建最大堆
        for (i in (n / 2 - 1) downTo 0) {
            heapify(arr, n, i + start, start, end)
        }

        // 一个一个取出元素，并调整堆
        for (i in end downTo start + 1) {
            // 将堆顶元素与数组末尾交换
            arr[start] = arr[i].also { arr[i] = arr[start] }

            // 调整堆
            heapify(arr, i - start, start, start, i - 1)
        }
    }

}