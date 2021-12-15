package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.graphics.Bitmap
import android.graphics.Matrix
import java.nio.ByteBuffer


abstract class whochari {
    companion object {
        private fun buffer2patch(
            maskBitmap: Bitmap,
            width: Int,
            height: Int,
            widthRatio: Int,
            heightRatio: Int
        ) {
            val arr: MutableList<ByteBuffer>
            //4:3
            val widthpatchlength: Int = width / widthRatio
            val heightpatchlength: Int = height / heightRatio
            for (z in 0 until widthRatio * heightRatio) {
                val arr_element : ByteBuffer =
                    ByteBuffer.allocateDirect(1 * widthpatchlength * heightpatchlength * ImageSegmentationModelExecutor.NUM_CLASSES * 4)
                for (y in 0 until heightpatchlength) {
                    for (x in 0 until widthpatchlength) {
                    }
                        var maxVal = 0f

                    }
                }
            }
        }
}