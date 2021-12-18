package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.util.Log
import java.util.*
import java.util.Collections

class UseMaskinform(){
    var TTScallback: ((String) -> Unit)? = null


    fun startTTS(){
        this.TTScallback?.invoke("안내를 시작합니다.")
    }
    fun stopTTS(){
        this.TTScallback?.invoke("안내를 중지합니다.")
    }

    fun statusDistinguish(patch: Array<Array<Int>>) : String{
        val centerj = intArrayOf(5, 6, 7, 8, 9, 10, 11, 12, 13, 14) // width
        val centeri = intArrayOf(11, 12, 13, 14) // height
        var status: String =  Distinguish(patch, centeri, centerj)
        return status
    }

    fun Distinguish(patch : Array<Array<Int>>,
                    IArray : IntArray,
                    JArray : IntArray)
        : String{
        var tempmap : HashMap<String, Int> = hashMapOf("alley" to 0,
                                                    "sidewalk" to 0,
                                                    "caution" to 0,
                                                    "braille" to 0,
                                                    "background" to 0,
                                                    "roadway" to 0,
                                                    "crosswalk" to 0)
        for(i in IArray.indices){
            for(j in JArray.indices){
                if (labelsArrays[patch[i][j]].lowercase(Locale.getDefault()) != "roadway_crosswalk") {
                    tempmap[labelsArrays[patch[i][j]].split("_")[0]]!!.plus(1)
                }
                else{
                    tempmap["crosswalk"]!!.plus(1)
                }
            }
        }
        var maxCnt : Int = 0
        var maxKey : String = "background"
        tempmap.forEach{
            k, v ->
            if (v > maxCnt){
                maxKey = k
            }
            else if ( v == maxCnt){
                if (rankRoad[maxKey]!! > rankRoad[k]!!){
                    maxKey = k
                }
            }
        }

        return maxKey
    }

    fun execute_TTS(patch: Array<Array<Int>>){
        //TODO
        // patch : 20X15형태의 class(string 형태의)
        // 각 하나의 patch는 convertBytebufferMaskToBitmap함수의 standard X standard 픽셀의 최빈값
        // ex [[alley, alley, alley, sidewalk, sidewalk, sidewalk],
        //    [alley, alley, alley, sidewalk, sidewalk, sidewalk]...]]

        if (tempstatus == null){
            tempstatus = statusDistinguish(patch)
            this.TTScallback?.invoke(Koreanclass[tempstatus]!!)
        }
        count++
        Log.d("count", "${count}")
    }


    companion object {
        private var tempstatus : String? = null

        private var Koreanclass : Map<String, String>
                = mapOf("alley" to "차도",
            "sidewalk" to "인도",
            "caution" to "주의구역",
            "braille" to "점자블록",
            "bike" to "자전거",
            "background" to "장애물",
            "crosswalk" to "횡단보도"
        )

        private val labelsArrays : Array<String> = arrayOf(
        "background", "alley_crosswalk", "alley_damaged", "alley_normal",
        "alley_speed_bump", "bike_lane", "braille_guide_blocks_damaged",
        "braille_guide_blocks_normal", "caution_zone_grating",
        "caution_zone_manhole", "caution_zone_repair_zone", "caution_zone_stairs",
        "caution_zone_tree_zone", "roadway_crosswalk", "roadway_normal",
        "sidewalk_asphalt", "sidewalk_blocks", "sidewalk_cement",
        "sidewalk_damaged", "sidewalk_other", "sidewalk_soil_stone",
        "sidewalk_urethane"
        )


        val rankRoad : Map<String, Int> = mapOf("alley" to 4,
            "sidewalk" to 7,
            "caution" to 5,
            "braille" to 1,
            "background" to 6,
            "roadway" to 3,
            "crosswalk" to 2) // 서열(낮을수록 높은것)

        private var count : Int = 0
    }
}