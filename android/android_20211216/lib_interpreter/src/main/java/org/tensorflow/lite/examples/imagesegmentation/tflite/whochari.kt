package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import java.nio.ByteBuffer
import android.speech.*
import android.speech.tts.TextToSpeech
import org.w3c.dom.Text
import java.util.*

class whochari() {
    //8 X
    fun ttsSpeak(tts : TextToSpeech, str : String){
        tts.speak(str, TextToSpeech.QUEUE_ADD, null, null)
        tts.playSilentUtterance(1000, TextToSpeech.QUEUE_ADD, null)
    }

    fun whochari(patch: Array<Array<String>>){
        //TODO
        // patch : 6X8형태의 class(string 형태의)
        // 각 하나의 patch는 convertBytebufferMaskToBitmap함수의 standard X standard 픽셀의 최빈값
        // ex [[alley, alley, alley, sidewalk, sidewalk, sidewalk],
        //    [alley, alley, alley, sidewalk, sidewalk, sidewalk]...]]
        
    }
}