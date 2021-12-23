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

package org.tensorflow.lite.examples.imagesegmentation

import android.speech.tts.TextToSpeech
import androidx.lifecycle.ViewModel
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import kotlinx.coroutines.*
import java.io.File
import org.tensorflow.lite.examples.imagesegmentation.tflite.ImageSegmentationModelExecutor
import org.tensorflow.lite.examples.imagesegmentation.tflite.ModelExecutionResult
import org.tensorflow.lite.examples.imagesegmentation.tflite.UseMaskInform
import org.tensorflow.lite.examples.imagesegmentation.utils.ImageUtils

private const val TAG = "MLExecutionViewModel"

class MLExecutionViewModel : ViewModel() {

  private val _resultingBitmap = MutableLiveData<ModelExecutionResult>()

  val resultingBitmap: LiveData<ModelExecutionResult>
    get() = _resultingBitmap

  private val viewModelJob = Job()
  private val viewModelScope = CoroutineScope(viewModelJob)

  // the execution of the model has to be on the same thread where the interpreter
  // was created
  fun onApplyModel(
    filePath: String,
    imageSegmentationModel: ImageSegmentationModelExecutor?,
    inferenceThread: ExecutorCoroutineDispatcher,
    ttsclass : UseMaskInform,
    demoButton_flag : Boolean
  ): Deferred<Long> {
      val t = viewModelScope.async(inferenceThread) {
        val contentImage = ImageUtils.decodeBitmap(File(filePath))
        try {
          val result = imageSegmentationModel?.execute(contentImage, ttsclass, demoButton_flag)
          _resultingBitmap.postValue(result)
          result!!.executiontime

        } catch (e: Exception) {
          Log.e(TAG, "Fail to execute ImageSegmentationModelExecutor: ${e.message}")
          _resultingBitmap.postValue(null)
          0
        }
      }
    return t
  }


}
