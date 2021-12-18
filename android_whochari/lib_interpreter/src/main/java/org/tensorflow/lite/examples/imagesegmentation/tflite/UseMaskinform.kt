package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import java.nio.ByteBuffer
import android.speech.*
import android.speech.tts.TextToSpeech
import android.util.Log
import org.w3c.dom.Text
import java.util.*
import androidx.fragment.app.Fragment
import java.io.File


class UseMaskinform(){
    var TTScallback: ((String) -> Unit)? = null


    private fun initSetStatus(str : String) : String{
        return str
    }
    fun startTTS(){
        this.TTScallback?.invoke("인퍼런스를 시작합니다.응애")
    }
    fun stopTTS(){
        this.TTScallback?.invoke("안내를 중지합니다.")
    }

    fun execute_TTS(patch: Array<Array<Int>>){
        //TODO
        // patch : 6X8형태의 class(string 형태의)
        // 각 하나의 patch는 convertBytebufferMaskToBitmap함수의 standard X standard 픽셀의 최빈값
        // ex [[alley, alley, alley, sidewalk, sidewalk, sidewalk],
        //    [alley, alley, alley, sidewalk, sidewalk, sidewalk]...]]
        if (tempstatus == null){
            tempstatus = initSetStatus("alley")
            this.TTScallback?.invoke(map[tempstatus]!!)
        }
        count++
        Log.d("count", "${count}")
    }

    companion object {
        private var tempstatus : String? = null
        private var map : Map<String, String>
                = mapOf("alley" to "차도",
            "sidewalk" to "인도",
            "caution" to "주의구역",
            "braille" to "점자블록"
        )
        private var count : Int = 0
    }
}