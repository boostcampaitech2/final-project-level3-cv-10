package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.util.Log

class UseMaskinform(){
    var TTScallback: ((String) -> Unit)? = null


    fun startTTS(){
        this.TTScallback?.invoke("안내를 시작합니다.")
    }
    fun stopTTS(){
        this.TTScallback?.invoke("안내를 중지합니다.")
        tempStatus = mutableListOf(null, null) //temp, prev
        tempFront = mutableListOf(null, null) //temp, prev
        tempLeftUp = mutableListOf(null, null) //temp, prev
        tempLeftDown = mutableListOf(null, null) //temp, prev
        tempRightUp = mutableListOf(null, null) //temp, prev
        tempRightDown = mutableListOf(null, null) //temp, prev
        statusStack = 0
        frontStack = 0
        leftUpStack = 0
        leftDownStack = 0
        rightUpStack = 0
        rightDownStack = 0
    }

    // arr는 좌우반전, i와 j값이 뒤바뀐 상태임을 주의해야한다!

    private fun statusDistinguish(patch: Array<Array<Int>>) : String{
        // 현재 위치
        val centeri = IntArray(6) {i -> i + 7}// width
        val centerj = IntArray(5) {i -> i + 10} // height
        var status: String =  maskDistinguish(patch, centeri, centerj)
        return status
    }

    private fun frontDistinguish(patch: Array<Array<Int>>) : String{
        // 전방 위치
        val centeri = IntArray(12) {i -> i + 4} // width
        val centerj = IntArray(7) {i -> i} // height
        var status: String =  maskDistinguish(patch, centeri, centerj)
        return status
    }

    private fun leftUpDistinguish(patch: Array<Array<Int>>) : String{
        // 전방 위치
        val centeri = IntArray(8) {i -> i} // width
        val centerj = IntArray(7) {i ->i} // height
        var status: String =  maskDistinguish(patch, centeri, centerj)
        return status
    }


    private fun rightUpDistinguish(patch: Array<Array<Int>>) : String{
        // 전방 위치
        val centeri = IntArray(8) {i -> i + 12} // width
        val centerj = IntArray(7) {i -> i} // height
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
            "bike" to 0,
            "background" to 0,
            "roadway" to 0,
            "crosswalk" to 0)
        for(i in IArray){
            for(j in JArray){
                var label = labelsArrays[patch[i][j]]
                //Log.d("status", "$i, $j, ${patch[i][j]}")
                if (label == "background"){
                    tempMap[label] = tempMap[label]!! + 1
                }
                else if (label == "roadway_crosswalk" || label == "alley_crosswalk"){
                    tempMap["crosswalk"] = tempMap["crosswalk"]!! + 1
                }
                else{
                    label = label.split("_")[0]
                    tempMap[label] = tempMap[label]!! + 1
                }
            }
            //Log.d("mapis ", "${tempMap}")
        }
        var maxCnt = 0
        var maxKey = "background"

        //dict search(max)
        for ((k, v) in tempMap){
            //Log.d("map is ", "$k, $v")
            if (k == "braille" && v > 5){
                maxKey = k
                break
            }
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

    fun executeTTS(patch: Array<Array<Int>>): String {
        //TODO
        // patch : 20X15형태의 class
        // ex [[1, 2, 3, ..., 1, 1, 1],
        //    [1, 2, 3, ..., 1, 1, 1]...]]

        /*patch.forEach {
            var str = ""
            it.forEach {
                str += it.toString()
            }
            Log.d("arris ", "${str}")
        }*/


        if (tempStatus[1] == null) {
            var str = ""
            tempStatus[1] = statusDistinguish(patch)
            tempFront[1] = frontDistinguish(patch)
            tempLeftUp[1] = leftUpDistinguish(patch)
            //tempLeftDown[1] = leftDownDistinguish(patch)
            tempRightUp[1] = rightUpDistinguish(patch)
            //tempRightDown[1] = rightDownDistinguish(patch)
            TTScallback(Triple(" "," "," "), Triple(tempFront[1]!!, tempLeftUp[1]!!, tempRightUp[1]!!))
        }
        var prev : Triple<String, String, String> = Triple(tempFront[1]!!, tempLeftUp[1]!!, tempRightUp[1]!!) //전방, 좌측, 우측
        var statusPair = checkStack(tempStatus, statusStack, statusDistinguish(patch), 0)
        tempStatus = statusPair.first
        statusStack = statusPair.second

        var frontPair = checkStack(tempFront, frontStack, frontDistinguish(patch), 1)
        tempFront = frontPair.first
        frontStack = frontPair.second

        var leftUpPair = checkStack(tempLeftUp, leftUpStack, leftUpDistinguish(patch), 2)
        tempLeftUp = leftUpPair.first
        leftUpStack = leftUpPair.second


        var rightUpPair = checkStack(tempRightUp, rightUpStack, rightUpDistinguish(patch), 4)
        tempRightUp = rightUpPair.first
        rightUpStack = rightUpPair.second
        var temp : Triple<String, String, String> = Triple(tempFront[1]!!, tempLeftUp[1]!!, tempRightUp[1]!!)

        TTScallback(prev, temp)

        return "status ${tempStatus[1].toString()} $statusStack\n" +
                "front ${tempFront[1].toString()} $frontStack\n" +
                " leftup ${tempLeftUp[1].toString()} $leftUpStack\n" +
                " rightup ${tempRightUp[1].toString()} $rightUpStack\n"
    }

    private fun checkStack(temp : MutableList<String?>, stack: Int, str : String, location: Int)
            :Pair<MutableList<String?>, Int>{
        var tempStack : Int = stack
        if (str == temp[1]) {
            return Pair(temp, 0)
        }
        temp[0] = str
        tempStack += 1

        if (tempStack == count){
            temp[1] = temp[0]
            tempStack -= count
        }
        Log.d("stack", "$tempStack")
        return Pair(temp, tempStack)
    }

    private fun TTScallback(prev : Triple<String, String, String>, temp : Triple<String, String, String>){
        val (prevF, prevL, prevR) = prev
        val (tempF, tempL, tempR) = temp
        val locationLabel : Map<Int, String> = mapOf(1 to "전방에",
            2 to "좌측에",
            3 to "전방 및 좌측에",
            4 to "우측에",
            5 to "전방 및 우측에",
            6 to "좌측과 우측에",
            7 to "전방 및 좌측과 우측에")
        if (prevF == tempF && prevL == tempL && prevR == tempR){
            return
        }
        else{
            var stack : Int = 0
            var strList : MutableList<String> = mutableListOf()
            if (tempF == tempL && tempL == tempR) {
                this.TTScallback?.invoke("전방에 넓게 ${KoreanClass[tempR]!!}${partClass[tempR]} 있습니다.")
                return
            }
            if (tempF != prevF){
                stack += 1
                strList.add(tempF)
            }
            if (tempL != prevL){
                stack += 2
                strList.add(tempF)
            }
            if (tempR != prevR){
                stack += 4
                strList.add(tempF)
            }
            var str : String = " "
            var strpart : String = ""

            for (i in 0 until strList.size){
                if (i == 0 || (i >= 1 && strList[i] != strList[i - 1])){
                    str += KoreanClass[strList[i]]
                }
                if (i == strList.size - 1){
                    strpart += partClass[strList[i]]
                }
            }

            this.TTScallback?.invoke("${locationLabel[stack]}${str}${strpart} 있습니다.")
        }
        return
    }

    companion object {
        private var tempStatus : MutableList<String?> = mutableListOf(null, null)//temp, prev
        private var tempFront : MutableList<String?> = mutableListOf(null, null)  //temp, prev
        private var tempLeftUp : MutableList<String?> = mutableListOf(null, null)//temp, prev
        private var tempLeftDown : MutableList<String?> = mutableListOf(null, null) //temp, prev
        private var tempRightUp : MutableList<String?>  = mutableListOf(null, null)//temp, prev
        private var tempRightDown : MutableList<String?>  = mutableListOf(null, null) //temp, prev
        private var KoreanClass : Map<String, String>
                = mapOf("alley" to "차도",
            "sidewalk" to "인도",
            "caution" to "주의구역",
            "braille" to "점자블록",
            "bike" to "자전거도로",
            "background" to "장애물",
            "crosswalk" to "횡단보도",
            "roadway" to "차도"
        )
        private var partClass : Map<String, String>
                = mapOf("alley" to "가",
            "sidewalk" to "가",
            "caution" to "이",
            "braille" to "이",
            "bike" to "가",
            "background" to "이",
            "crosswalk" to "가",
            "roadway" to "가"
        )
        private var statusStack : Int = 0
        private var frontStack : Int = 0
        private var leftUpStack : Int = 0
        private var leftDownStack : Int = 0
        private var rightUpStack : Int = 0
        private var rightDownStack : Int = 0
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
        val goodLabels = arrayOf(
            "alley_crosswalk", "alley_normal",
            "sidewalk_asphalt", "sidewalk_blocks", "sidewalk_cement",
            "sidewalk_other", "sidewalk_soil_stone","sidewalk_urethane"
        )
        val badLabels = arrayOf(
            "alley_speed_bump", "caution_zone_grating",
            "caution_zone_manhole", "caution_zone_repair_zone",
            "caution_zone_tree_zone", "roadway_crosswalk", "roadway_normal"
        )
        val noticeLabels = arrayOf(
            "alley_crosswalk", "alley_speed_bump", "caution_zone_grating",
            "caution_zone_manhole", "caution_zone_repair_zone",
            "caution_zone_tree_zone", "roadway_crosswalk", "roadway_normal"
        )
        val disregardLabels = arrayOf(
            "background", "alley_damaged", "braille_guide_blocks_damaged",
            "caution_zone_stairs", "sidewalk_damaged"
        )

        val rankRoad : Map<String, Int> = mapOf("alley" to 4,
            "sidewalk" to 8,
            "bike" to 5,
            "caution" to 7,
            "braille" to 1,
            "background" to 6,
            "roadway" to 3,
            "crosswalk" to 2) // 서열(낮을수록 높은것)

        private val count : Int = 3
    }
}