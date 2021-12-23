
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

import android.Manifest
import androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Color
import android.hardware.camera2.CameraCharacteristics
import android.os.Build
import android.os.Bundle
import android.os.Process
import android.speech.tts.TextToSpeech
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import android.util.Log
import android.util.TypedValue
import android.view.animation.AnimationUtils
import android.view.animation.BounceInterpolator
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.Observer
import com.bumptech.glide.Glide
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import kotlinx.coroutines.*
import java.io.File
import java.util.concurrent.Executors
import org.tensorflow.lite.examples.imagesegmentation.camera.CameraFragment
import org.tensorflow.lite.examples.imagesegmentation.tflite.ImageSegmentationModelExecutor
import org.tensorflow.lite.examples.imagesegmentation.tflite.ModelExecutionResult
import org.tensorflow.lite.examples.imagesegmentation.tflite.UseMaskInform
import java.util.*
import kotlin.collections.HashMap


// This is an arbitrary number we are using to keep tab of the permission
// request. Where an app has multiple context for requesting permission,
// this can help differentiate the different contexts
private const val REQUEST_CODE_PERMISSIONS = 10

// This is an array of all the permission specified in the manifest
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

private const val TAG = "MainActivity"

class MainActivity : AppCompatActivity(), CameraFragment.OnCaptureFinished{

  private lateinit var cameraFragment: CameraFragment
  private var TTSCallBack = UseMaskInform()
  private lateinit var viewModel: MLExecutionViewModel
  private lateinit var viewFinder: FrameLayout
  private lateinit var gridImageView: ImageView
  private lateinit var resultImageView: ImageView
  private lateinit var originalImageView: ImageView
  private lateinit var chipsGroup: ChipGroup
  private lateinit var rerunButton: Button
  private lateinit var captureButton: ImageButton
  private lateinit var demoButton: Button
  var tts: TextToSpeech? = null
  private var lastSavedFile = ""
  private var useGPU = false
  //  private var imageSegmentationModel: ImageSegmentationModelExecutor? = null
  private var imageSegmentationModel: ImageSegmentationModelExecutor? = null
  private val inferenceThread = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
  private val mainScope = MainScope()
  private var captureButton_flag : Boolean = false
  private var demoButton_flag : Boolean = false
  private var lensFacing = CameraCharacteristics.LENS_FACING_BACK
  private var usetime : Long = 0
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.tfe_is_activity_main)

    val toolbar: Toolbar = findViewById(R.id.toolbar)
    setSupportActionBar(toolbar)
    supportActionBar?.setDisplayShowTitleEnabled(false)

    viewFinder = findViewById(R.id.view_finder)
    gridImageView = findViewById(R.id.grid_imageview)
    resultImageView = findViewById(R.id.result_imageview)
    originalImageView = findViewById(R.id.original_imageview)
    chipsGroup = findViewById(R.id.chips_group)
    captureButton = findViewById(R.id.capture_button)
    demoButton = findViewById(R.id.demo_button)

    val useGpuSwitch: Switch = findViewById(R.id.switch_use_gpu)
    // Request camera permissions
    if (allPermissionsGranted()) {
      addCameraFragment()
    } else {
      ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
    }

    viewModel = AndroidViewModelFactory(application).create(MLExecutionViewModel::class.java)
    viewModel.resultingBitmap.observe(
      this,
      Observer { resultImage ->
        if (resultImage != null) {
          updateUIWithResults(resultImage)
        }
        enableControls(true)
      }
    )

    demoButton.setBackgroundColor(Color.RED)
    demoButton.setOnClickListener  {
      if(demoButton_flag)
      {
        demoButton_flag = false
        demoButton.setBackgroundColor(Color.RED)
        demoButton.setText("Disable Demo Mode")
      }
      else
      {
        demoButton_flag = true
        demoButton.setBackgroundColor(Color.GREEN)
        demoButton.setText("Enable Demo Mode")
      }
    }

    createModelExecutor(useGPU)
    useGpuSwitch.setOnCheckedChangeListener { _, isChecked ->
      useGPU = isChecked
      mainScope.async(inferenceThread) { createModelExecutor(useGPU) }
    }
    initTextToSpeech()
    rerunButton = findViewById(R.id.rerun_button)
    rerunButton.setOnClickListener {
      if (lastSavedFile.isNotEmpty()) {
        enableControls(false)
        viewModel.onApplyModel(lastSavedFile, imageSegmentationModel, inferenceThread, TTSCallBack, demoButton_flag)
      }
    }

    animateCameraButton()
    setChipsToLogView(HashMap<String, Int>())
    setupControls(usetime)
    enableControls(true)
  }

  private fun ttsSpeak(tts : TextToSpeech, str : String){
    tts.speak(str, TextToSpeech.QUEUE_ADD, null, null)
    tts.playSilentUtterance(300, TextToSpeech.QUEUE_ADD, null)
  }


  private fun initTextToSpeech() {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
      return
    }
    tts = TextToSpeech(this) {
    }
    sendTTS(tts!!)
  }

  private fun createModelExecutor(useGPU: Boolean) {
    if (imageSegmentationModel != null) {
      imageSegmentationModel!!.close()
      imageSegmentationModel = null
    }
    try {
      imageSegmentationModel = ImageSegmentationModelExecutor(this, useGPU)
    } catch (e: Exception) {
      Log.e(TAG, "Fail to create ImageSegmentationModelExecutor: ${e.message}")
      val logText: TextView = findViewById(R.id.log_view)
      logText.text = e.message
    }
  }

  private fun animateCameraButton() {
    val animation = AnimationUtils.loadAnimation(this, R.anim.scale_anim)
    animation.interpolator = BounceInterpolator()
    captureButton.animation = animation
    captureButton.animation.start()
  }



  private fun setChipsToLogView(itemsFound: Map<String, Int>) {
    chipsGroup.removeAllViews()

    val paddingDp =
      TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 10F, resources.displayMetrics).toInt()

    for ((label, color) in itemsFound) {
      val chip = Chip(this)
      chip.text = label
      chip.chipBackgroundColor = getColorStateListForChip(color)
      chip.isClickable = false
      chip.setPadding(0, paddingDp, 0, paddingDp)
      chipsGroup.addView(chip)
    }
    val labelsFoundTextView: TextView = findViewById(R.id.tfe_is_labels_found)
    if (chipsGroup.childCount == 0) {
      labelsFoundTextView.text = getString(R.string.tfe_is_no_labels_found)
    } else {
      labelsFoundTextView.text = getString(R.string.tfe_is_labels_found)
    }
    chipsGroup.parent.requestLayout()
  }

  private fun getColorStateListForChip(color: Int): ColorStateList {
    val states =
      arrayOf(
        intArrayOf(android.R.attr.state_enabled), // enabled
        intArrayOf(android.R.attr.state_pressed) // pressed
      )

    val colors = intArrayOf(color, color)
    return ColorStateList(states, colors)
  }

  private fun setImageView(imageView: ImageView, image: Bitmap) {
    Glide.with(baseContext).load(image).override(512, 512).fitCenter().into(imageView)
  }

  private fun updateUIWithResults(modelExecutionResult: ModelExecutionResult) {
    setImageView(gridImageView, modelExecutionResult.gridResult)
    setImageView(resultImageView, modelExecutionResult.bitmapResult)
    setImageView(originalImageView, modelExecutionResult.bitmapOriginal)
    val logText: TextView = findViewById(R.id.log_view)
    logText.text = modelExecutionResult.executionLog

    setChipsToLogView(modelExecutionResult.itemsFound)
    enableControls(true)
  }

  private fun enableControls(enable: Boolean) {
    rerunButton.isEnabled = enable && lastSavedFile.isNotEmpty()
    captureButton.isEnabled = enable
  }

  private fun setupControls(time : Long) {
    captureButton.setOnClickListener {
      captureButton_flag = !captureButton_flag
      it.clearAnimation()
      runBlocking {
        if (captureButton_flag) {
          TTSCallBack.startTTS()
          cameraFragment.takePicture(captureButton_flag)
        } else {
          TTSCallBack.stopTTS()
          cameraFragment.onDestroy()
          addCameraFragment()
        }
      }
    }

    findViewById<ImageButton>(R.id.toggle_button).setOnClickListener {
      lensFacing =
        if (lensFacing == CameraCharacteristics.LENS_FACING_BACK) {
          CameraCharacteristics.LENS_FACING_FRONT
        } else {
          CameraCharacteristics.LENS_FACING_BACK
        }
      cameraFragment.setFacingCamera(lensFacing)
      addCameraFragment()
    }
  }

  /**
   * Process result from permission request dialog box, has the request been granted? If yes, start
   * Camera. Otherwise display a toast
   */
  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<String>,
    grantResults: IntArray
  ) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode == REQUEST_CODE_PERMISSIONS) {
      if (allPermissionsGranted()) {
        addCameraFragment()
        viewFinder.post { setupControls(0) }
      } else {
        Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show()
        finish()
      }
    }
  }


  private fun addCameraFragment() {
    cameraFragment = CameraFragment.newInstance()
    cameraFragment.setFacingCamera(lensFacing)
    supportFragmentManager.popBackStack()
    supportFragmentManager.beginTransaction().replace(R.id.view_finder, cameraFragment).commit()
  }

  /** Check if all permission specified in the manifest have been granted */
  private fun allPermissionsGranted() =
    REQUIRED_PERMISSIONS.all {
      checkPermission(it, Process.myPid(), Process.myUid()) == PackageManager.PERMISSION_GRANTED
    }

  fun sendTTS(tts : TextToSpeech) {
    val scope = GlobalScope.launch {
      while (true) {
        TTSCallBack.TTScallback = { str ->
          Log.d("arrive message", "$str")
          ttsSpeak(tts!!, str)
        }
      }
    }
  }


  override fun onCaptureFinished(file: File): Deferred<Long> {
    val msg = "Photo capture succeeded: ${file.absolutePath}"
    Log.d(TAG, msg)

    lastSavedFile = file.absolutePath
    enableControls(false)
    val time : Deferred<Long> = viewModel.onApplyModel(file.absolutePath, imageSegmentationModel, inferenceThread, TTSCallBack, demoButton_flag)
    return time
  }
}
