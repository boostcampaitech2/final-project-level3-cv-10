package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.util.Log
import java.util.*

class UseMaskinform(){
    var TTScallback: ((String) -> Unit)? = null


    fun startTTS(){
        this.TTScallback?.invoke("안내를 시작합니다.")
    }
    fun stopTTS(){
        this.TTScallback?.invoke("안내를 중지합니다.")
        tempStatus = mutableListOf(null, null) //temp, prev
        tempFront = mutableListOf(null, null) //temp, prev
        statusStack = 0
        frontStack = 0
    }

    // arr는 좌우반전, i와 j값이 뒤바뀐 상태임을 주의해야한다!

    private fun statusDistinguish(patch: Array<Array<Int>>) : String{
        // 현재 위치
        val centeri = IntArray(6) {i -> i + 7}// width
        val centerj = IntArray(5) {i -> i + 10} // height
        return maskDistinguish(patch, centeri, centerj)
    }

    private fun frontDistinguish(patch: Array<Array<Int>>) : String{
        // 전방 위치
        val centeri = IntArray(12) {i -> i + 4} // width
        val centerj = IntArray(7) {i -> i + 2} // height
        return maskDistinguish(patch, centeri, centerj)
    }

    private fun findBraille(patch: Array<Array<Int>>, flag : BooleanArray) : BooleanArray{
        var centeri = IntArray(6) {i -> i}// width
        var centerj =IntArray(7) {i -> i + 8} // height
        if (maskDistinguish(patch, centeri, centerj) == "braille") {
            if (!flag[0]) {
                flag[0] = true
                this.TTScallback?.invoke("좌측에 점자블럭이 있습니다.")
            }
        }
        else{
            flag[0] = false
        }

        centeri = IntArray(6) {i -> i + 14}// width
        centerj = IntArray(7) {i -> i + 8} // height
        if (maskDistinguish(patch, centeri, centerj) == "braille"){
            if (!flag[2]) {
                flag[2] = true
                this.TTScallback?.invoke("우측에 점자블럭이 있습니다.")
            }
        }
        else{
            flag[2] = false
        }
        return flag
    }


    private fun findObstacle(patch: Array<Array<Int>>) : Boolean{
        // 전방 위치
       val centeri = IntArray(10) {i -> i + 5} // width
       val centerj = IntArray(7) {i -> i + 4} // height
       var count : Int = 0
       for(i in centeri){
            for(j in centerj){
                if (judgeArrays[patch[i][j]] == 5){
                    count += 1
                }
                if (count > 10){
                    return true
                }
            }
        }
        return false
    }

    private fun findNotice(patch: Array<Array<Int>>) : String{
        // 전방 위치
        // 5, 8, 9
        val centeri = IntArray(10) {i -> i + 5} // width
        val centerj = IntArray(7) {i -> i + 4} // height
        var noticeList = IntArray(10) {0}
        var count : Int = 0
        var noticeMap = mapOf(4 to "bump", 8 to "grating", 9 to "manhole")
        for(i in centeri){
            for(j in centerj){
                if (judgeArrays[patch[i][j]] == 3){
                    count += 1
                    if (patch[i][j] == 4 || patch[i][j] == 8 || patch[i][j] == 9){
                        noticeList[patch[i][j]] += 1
                    }
                }
                if (count > 10){
                    val maxNum : Int = noticeList.indexOf(noticeList.maxOrNull()!!)
                    return noticeMap[maxNum]!!
                }
            }
        }
        return "false"
    }


    private fun maskDistinguish(patch : Array<Array<Int>>,
                                IArray : IntArray,
                                JArray : IntArray)
            : String{
        var tempMap : MutableMap<Int, Int> = mutableMapOf()
        for(i in IArray){
            for(j in JArray){
                var label : Int = judgeArrays[patch[i][j]]
                if(tempMap.containsKey(label)){
                    tempMap[label] = tempMap[label]!! + 1
                }
                else{
                    tempMap[label] = 1
                }
            }
        }
        if (tempMap.containsKey(5)){
            if (findObstacle(patch))
                return "obstacle"
            }
        if (tempMap.containsKey(3)){
            val flag : String = findNotice(patch)
            if (flag != "false") {
                return flag
            }
        }

        //Log.d("mapis ", "${tempMap}")

        return findMaxPixel(tempMap)
    }

    fun findMaxPixel(pixelMap : MutableMap<Int, Int>) : String{
        var maxCnt = 0
        var maxKey = "background"

        //dict search(max)
        for ((k, v) in pixelMap){
            //Log.d("map is ", "$k, $v")
            val road : String = hash2judge[k]
            if (road == "braille" && v > 5){
                maxKey = road
                break
            }
            if (v > maxCnt){
                maxKey = road
                maxCnt = v
            }
            else if (v == maxCnt){
                if (rankRoad[maxKey]!! > rankRoad[road]!!){
                    maxKey = road
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
            TTScallback(" ", tempFront[1]!!)
        }

        var prev = tempFront[1]!! //전방
        var statusPair = checkStack(tempStatus, statusStack, statusDistinguish(patch))
        tempStatus = statusPair.first
        statusStack = statusPair.second


        var frontPair = checkStack(tempFront, frontStack, frontDistinguish(patch))
        tempFront = frontPair.first
        frontStack = frontPair.second

        var temp = tempFront[1]!!

        if (temp == "braille") {
            brailleFlag = findBraille(patch, brailleFlag)

        }
        if (prev != temp) {
            if (temp == "obstacle"){
                this.TTScallback?.invoke("전방에 장애물이 있습니다. 다른 방향을 확인해주세요")
            }
            else{
                TTScallback(prev, temp)
            }
        }

        return "status ${tempStatus[1].toString()} $statusStack\n" +
                "front ${tempFront[1].toString()} $frontStack\n"
    }

    private fun checkStack(temp : MutableList<String?>, stack: Int, str : String)
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

    private fun TTSbraille(flag : Int) : Int{
        when(flag){
            4 -> {}
            1 -> this.TTScallback?.invoke("곧 좌측에 점자블록이 있습니다.")
            3 -> this.TTScallback?.invoke("전방에 점자블록이 있습니다.")
            2 -> this.TTScallback?.invoke("곧 우측에 점자블록이 있습니다.")
        }
        return flag
    }
    private fun TTScallback(prev : String, temp : String){
        val str : String = KoreanClass[temp] + partClass[temp]
        if (temp == "roadway" || temp == "alley" || temp == "bike"){
            this.TTScallback?.invoke("전방에 $str 있습니다. 조심해주세요")
        }
        else{
            this.TTScallback?.invoke("전방에 $str 있습니다.")
        }
        return
    }

    companion object {
        private var tempStatus : MutableList<String?> = mutableListOf(null, null)//temp, prev
        private var tempFront : MutableList<String?> = mutableListOf(null, null)  //temp, prev

        private var KoreanClass : Map<String, String>
                = mapOf("alley" to "이면도로",
            "manhole" to "맨홀",
            "bump" to "과속방지턱",
            "grating" to "배수구",
            "sidewalk" to "인도",
            "notice" to "주의구역",
            "braille" to "점자블록",
            "bike" to "자전거도로",
            "obstacle" to "장애물",
            "crosswalk" to "횡단보도",
            "roadway" to "차도"
        )
        private var partClass : Map<String, String>
                = mapOf("alley" to "가",
            "manhole" to "이",
            "bump" to "이",
            "grating" to "가",
            "sidewalk" to "가",
            "notice" to "이",
            "braille" to "이",
            "bike" to "가",
            "obstacle" to "이",
            "crosswalk" to "가",
            "roadway" to "가"
        )
        private var statusStack : Int = 0
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

        private val judgeArrays : Array<Int> = arrayOf(
            5, 7, 7, 7,
            3, 6, 4,
            4, 3,
            3, 5, 0,
            2, 8, 2,
            1, 1, 1,
            0, 1, 1,
            1
        )
        private val hash2judge : Array<String> = arrayOf("disregard", "sidewalk", "roadway",
                                                         "notice", "braille", "obstacle",
                                                         "bike", "alley", "crosswalk")

        private val rankRoad : Map<String, Int> = mapOf("alley" to 7,
            "sidewalk" to 8,
            "bike" to 4,
            "notice" to 6,
            "braille" to 1,
            "obstacle" to 5,
            "roadway" to 3,
            "crosswalk" to 2) // 서열(낮을수록 높은것)
        private var brailleFlag : BooleanArray = booleanArrayOf(false, false, false)
        private val count : Int = 3
    }
}