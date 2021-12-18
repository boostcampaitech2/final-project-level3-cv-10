package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.util.Log
import java.util.*
import java.util.Collections

class UseMaskinform(){
    var TTScallback: ((String) -> Unit)? = null


    fun startTTS(){
        this.TTScallback?.invoke("안내를 시작합니다.")
        frontStack = 0
        count = 0
    }
    fun stopTTS(){
        this.TTScallback?.invoke("안내를 중지합니다.")
    }

    private fun statusDistinguish(patch: Array<Array<Int>>) : String{
        val centerj = intArrayOf(5, 6, 7, 8, 9, 10, 11, 12, 13, 14) // width
        val centeri = IntArray(5) { it -> it + 10 } // height
        var status: String =  maskDistinguish(patch, centeri, centerj)
        return status
    }

    private fun frontDistinguish(patch: Array<Array<Int>>) : String{
        val centerj = IntArray(10) { it -> it + 5 } // width
        val centeri = IntArray(7) { it -> it} // height
        var status: String =  maskDistinguish(patch, centeri, centerj)
        return status
    }

    private fun maskDistinguish(patch : Array<Array<Int>>,
                                IArray : IntArray,
                                JArray : IntArray)
        : String{
        var tempMap : MutableMap<String, Int> = mutableMapOf("alley" to 0,
                                                    "sidewalk" to 0,
                                                    "caution" to 0,
                                                    "braille" to 0,
                                                    "background" to 0,
                                                    "roadway" to 0,
                                                    "crosswalk" to 0)
        for(i in IArray.indices){
            for(j in JArray.indices){
                var label = labelsArrays[patch[i][j]].split("_")[0].lowercase(Locale.getDefault())
                if (label != "roadway_crosswalk") {
                    tempMap[label] = tempMap[label]!! + 1
                }
                else{
                    tempMap["crosswalk"]!!.plus(1)
                }
            }
        }
        var maxCnt : Int = 0
        var maxKey : String = "background"
        for ((k, v) in tempMap){
            if (v > maxCnt){
                maxKey = k
                maxCnt = v
            }
            else if (v == maxCnt){
                if (rankRoad[maxKey]!! > rankRoad[k]!!){
                    maxKey = k
                }
            }
        }
        return maxKey
    }

    fun executeTTS(patch: Array<Array<Int>>){
        //TODO
        // patch : 20X15형태의 class(string 형태의)
        // 각 하나의 patch는 convertBytebufferMaskToBitmap함수의 standard X standard 픽셀의 최빈값
        // ex [[alley, alley, alley, sidewalk, sidewalk, sidewalk],
        //    [alley, alley, alley, sidewalk, sidewalk, sidewalk]...]]
        if (tempStatus == null){
            tempStatus = statusDistinguish(patch)
            tempFront[1] = frontDistinguish(patch)
            this.TTScallback?.invoke("현재 위치에 ${KoreanClass[tempStatus]!!}에 있습니다.")

        }
        tempFront[0] = frontDistinguish(patch)
        var frontPair = checkStack(tempFront, frontStack)
        tempFront = frontPair.first
        frontStack = frontPair.second
        Log.d("count", "${count}")
    }

    private fun checkStack(temp : MutableList<String?>, stack : Int)
        :Pair<MutableList<String?>, Int>{
        if (temp[0] == temp[1]) {
            return Pair(temp, 0)
        }
        if (temp[0] != temp[1]) {
            stack.plus(1)
        }
        if (stack == count){
            this.TTScallback?.invoke("전방에 ${KoreanClass[temp[0]]!!} 있습니다.")
            temp[1] = temp[0]
            stack.minus(count)
        }
        Log.d("stack", "$stack")
        return Pair(temp, stack)
    }

    companion object {
        private var tempStatus : String? = null
        private var tempFront : MutableList<String?> = mutableListOf(null, null) //temp, prev
        private var KoreanClass : Map<String, String>
                = mapOf("alley" to "차도가",
            "sidewalk" to "인도가",
            "caution" to "주의구역이",
            "braille" to "점자블록이",
            "bike" to "자전거도로가",
            "background" to "장애물이",
            "crosswalk" to "횡단보도가",
            "roadway" to "차도가"
        )
        private var frontStack : Int = 0
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
            "sidewalk" to 8,
            "bike" to 5,
            "caution" to 7,
            "braille" to 1,
            "background" to 6,
            "roadway" to 3,
            "crosswalk" to 2) // 서열(낮을수록 높은것)

        private var count : Int = 7
    }
}