/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Environment
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.core.graphics.ColorUtils
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.random.Random
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.examples.imagesegmentation.utils.ImageUtils
import org.tensorflow.lite.gpu.GpuDelegate
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.imagesegmentation.tflite.UseMaskInform
import java.io.*

/**
 * Class responsible to run the Image Segmentation model. more information about the DeepLab model
 * being used can be found here:
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://www.tensorflow.org/lite/models/segmentation/overview
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 *
 * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
 * 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
 * 'sofa', 'train', 'tv'
 */
class ImageSegmentationModelExecutor(context: Context, private var useGPU: Boolean = false) {
  private var gpuDelegate: GpuDelegate? = null
  private val segmentationMasks: ByteBuffer
  private val interpreter: Interpreter
  private var fullTimeExecutionTime = 0L
  private var preprocessTime = 0L
  private var imageSegmentationTime = 0L
  private var maskFlatteningTime = 0L

  private var statusLog : String = ""
  private var numberThreads = 8
  private val matInput: Mat? = null
  private val matResult: Mat? = null
  private var demoIndex = 0
  private var demoIndexStart = 0
  private var demoIndexEnd = 4150

  private var _context:Context = context
  init {

    interpreter = getInterpreter(context, imageSegmentationModel, useGPU)
    //segmentationMasks = ByteBuffer.allocateDirect(1 * imageSize * imageSize * NUM_CLASSES * 4)
    segmentationMasks = ByteBuffer.allocateDirect(1 * output_width * output_height * NUM_CLASSES * 4)
    segmentationMasks.order(ByteOrder.nativeOrder())
    //demoIndex = (demoIndexStart..demoIndexEnd).random()
    demoIndex = 2732
  }


  fun getBitmapFromAsset(filePath: String): Bitmap {
    val assetManager: AssetManager = this._context.assets
    var istr = assetManager.open(filePath)
    var bitmap = BitmapFactory.decodeStream(istr)
    return bitmap
  }


  fun execute(data: Bitmap, ttsclass : UseMaskInform, demoButton_flag : Boolean): ModelExecutionResult {
    try {
      fullTimeExecutionTime = SystemClock.uptimeMillis()
      preprocessTime = SystemClock.uptimeMillis()

      var scaledBitmap:Bitmap? = null
      if(demoButton_flag)
      {
        if(demoIndex >= demoIndexEnd)
        {
          demoIndex = (demoIndexStart..demoIndexEnd).random()
        }
        var path = "test_images/" + "%05d".format(demoIndex) + ".png"
        var demoImage = getBitmapFromAsset(path)
        scaledBitmap = ImageUtils.resizeBitmap(demoImage, width, height, false)
        demoIndex++
      }
      else
      {
        scaledBitmap = ImageUtils.resizeBitmap(data, width, height, false)
      }

      val contentArray =
        ImageUtils.bitmapToByteBuffer(scaledBitmap, width, height, IMAGE_MEAN, IMAGE_STD)
      preprocessTime = SystemClock.uptimeMillis() - preprocessTime

      imageSegmentationTime = SystemClock.uptimeMillis()
      interpreter.run(contentArray, segmentationMasks)
      imageSegmentationTime = SystemClock.uptimeMillis() - imageSegmentationTime
      Log.d(TAG, "Time to run the model $imageSegmentationTime")

      maskFlatteningTime = SystemClock.uptimeMillis()
      val (maskImageApplied, patch, itemsFound) =
        convertBytebufferMaskToBitmap(
          segmentationMasks,
          output_width,
          output_height,
          width/2,
          height/2,
          scaledBitmap,
          segmentColors
        )
      val patchbitmap = Bitmap.createBitmap(output_width, output_height, Bitmap.Config.ARGB_8888)
      for (j in 0 until output_height) {
        for (i in 0 until output_width) {
          patchbitmap.setPixel(i, j, segmentColors[patch[i][j]])
        }
      }
      val gridBitmap = ImageUtils.resizeBitmap(patchbitmap, width, height, false)
      statusLog = ttsclass.executeTTS(patch)

      maskFlatteningTime = SystemClock.uptimeMillis() - maskFlatteningTime
      Log.d(TAG, "Time to flatten the mask result $maskFlatteningTime")

      fullTimeExecutionTime = SystemClock.uptimeMillis() - fullTimeExecutionTime
      Log.d(TAG, "Total time execution $fullTimeExecutionTime")

      return ModelExecutionResult(
        gridBitmap,
        maskImageApplied,
        scaledBitmap,
        formatExecutionLog(),
        itemsFound,
        fullTimeExecutionTime
      )
    } catch (e: Exception) {
      val exceptionLog = "something went wrong: ${e.message}"
      Log.d(TAG, exceptionLog)

      val emptyBitmap = ImageUtils.createEmptyBitmap(width, height)
      return ModelExecutionResult(
        emptyBitmap,
        emptyBitmap,
        emptyBitmap,
        exceptionLog,
        HashMap<String, Int>(),
        0
      )
    }
  }

  private fun JudgeGridBitmapColor(data: Bitmap)
  {

  }

  // base:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java
  @Throws(IOException::class)
  private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(modelFile)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    fileDescriptor.close()
    return retFile
  }

  @Throws(IOException::class)
  private fun getInterpreter(
    context: Context,
    modelName: String,
    useGpu: Boolean = false
  ): Interpreter {
    val tfliteOptions = Interpreter.Options()
    tfliteOptions.setNumThreads(numberThreads)

    gpuDelegate = null
    if (useGpu) {
//      gpuDelegate = GpuDelegate()
//      tfliteOptions.addDelegate(gpuDelegate)
      var nnApiDelegate = NnApiDelegate()
      tfliteOptions.addDelegate(nnApiDelegate)
    }

    return Interpreter(loadModelFile(context, modelName), tfliteOptions)
  }

  private fun formatExecutionLog(): String {
    val sb = StringBuilder()
    sb.append("Input Image Size: $width x $height\n")
    sb.append("GPU enabled: $useGPU\n")
    sb.append("Number of threads: $numberThreads\n")
    sb.append("Pre-process execution time: $preprocessTime ms\n")
    sb.append("Model execution time: $imageSegmentationTime ms\n")
    sb.append("Mask flatten time: $maskFlatteningTime ms\n")
    sb.append("Full execution time: $fullTimeExecutionTime ms\n")
    sb.append(statusLog)
    return sb.toString()
  }

  fun close() {
    interpreter.close()
    if (gpuDelegate != null) {
      gpuDelegate!!.close()
    }
  }

  private fun convertBytebufferMaskToBitmap(
    inputBuffer: ByteBuffer,
    imageWidth: Int,
    imageHeight: Int,
    backgroundWidth: Int,
    backgroundHeight: Int,
    backgroundImage: Bitmap,
    colors: IntArray
  ): Triple<Bitmap, Array<Array<Int>>, Map<String, Int>> {
    val conf = Bitmap.Config.ARGB_8888
    val maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
    val scaledBackgroundImage =
      ImageUtils.resizeBitmap(backgroundImage, backgroundWidth, backgroundHeight, false)
    val mSegmentBits = Array(imageWidth) { Array(imageHeight) { _ -> 0 } }
    val itemsFound = HashMap<String, Int>()
    inputBuffer.rewind()
    var nextClass = imageWidth * imageHeight
    for (y in 0 until imageHeight) {
      for (x in 0 until imageWidth) {
        var maxVal = 0f
        mSegmentBits[x][y] = 0

        for (c in 0 until NUM_CLASSES) {
          val value = inputBuffer.getFloat((y * imageWidth + c * nextClass + x ) * 4)
          if (c == 0 || value > maxVal) {
            maxVal = value
            mSegmentBits[x][y] = c
          }
        }

        val label = labelsArrays[mSegmentBits[x][y]]
        val color = colors[mSegmentBits[x][y]]
        itemsFound.put(label, color)

        val newPixelColor =
          ColorUtils.compositeColors(
            colors[mSegmentBits[x][y]],
            scaledBackgroundImage.getPixel(x, y)
          )
        //resultBitmap.setPixel(x, y, newPixelColor)
        maskBitmap.setPixel(x, y, colors[mSegmentBits[x][y]])
      }
    }

    val patchbitmap = Bitmap.createBitmap(output_width, output_height, conf)
    for (j in 0 until output_height) {
      for (i in 0 until output_width) {
        patchbitmap.setPixel(i, j, segmentColors[mSegmentBits[i][j]])
      }
    }

    var maskOnly = ImageUtils.resizeBitmap(patchbitmap, backgroundWidth, backgroundHeight, false)
    val resultBitmap = Bitmap.createBitmap(backgroundWidth, backgroundHeight, conf)
    for (y in 0 until backgroundHeight) {
      for (x in 0 until backgroundWidth) {
        val newPixelColor =
          ColorUtils.compositeColors(
            maskOnly.getPixel(x, y),
            scaledBackgroundImage.getPixel(x, y)
          )
        resultBitmap.setPixel(x, y, newPixelColor)
      }
    }

    return Triple(resultBitmap, mSegmentBits, itemsFound)
  }

  companion object {

    public const val TAG = "SegmentationInterpreter"
    private const val imageSegmentationModel = "dlv3_mbv3-small_grid_reduced.211222.tflite"
    private const val width = 320
    private const val height = 240
    const val NUM_CLASSES = 22
    private const val IMAGE_MEAN = 127.5f
    private const val IMAGE_STD = 255.0f
    private const val output_width = 20
    private const val output_height = 15
    val segmentColors = IntArray(NUM_CLASSES)
    val labelsArrays =
      arrayOf(
        "background", "alley_crosswalk", "alley_damaged", "alley_normal",
        "alley_speed_bump", "bike_lane", "braille_guide_blocks_damaged",
        "braille_guide_blocks_normal", "caution_zone_grating",
        "caution_zone_manhole", "caution_zone_repair_zone", "caution_zone_stairs",
        "caution_zone_tree_zone", "roadway_crosswalk", "roadway_normal",
        "sidewalk_asphalt", "sidewalk_blocks", "sidewalk_cement",
        "sidewalk_damaged", "sidewalk_other", "sidewalk_soil_stone",
        "sidewalk_urethane"
      )


    init {
      val random = Random(System.currentTimeMillis())
      segmentColors[0] = Color.TRANSPARENT
      for (i in 1 until NUM_CLASSES) {
        segmentColors[i] =
          Color.argb(
            (128),
            getRandomRGBInt(random),
            getRandomRGBInt(random),
            getRandomRGBInt(random)
          )
      }
    }

    private fun getRandomRGBInt(random: Random) = (255 * random.nextFloat()).toInt()
  }
}
