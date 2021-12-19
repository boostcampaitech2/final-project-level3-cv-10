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
        val centeri = IntArray(10) {i -> i + 5} // width
        val centerj = IntArray(5) {i -> i + 2} // height
        var status: String =  maskDistinguish(patch, centeri, centerj)
        return status
    }

    private fun leftUpDistinguish(patch: Array<Array<Int>>) : String{
        // 전방 위치
        val centeri = IntArray(5) {i -> i} // width
        val centerj = IntArray(15) {i ->i} // height
        var status: String =  maskDistinguish(patch, centeri, centerj)
        return status
    }


    private fun rightUpDistinguish(patch: Array<Array<Int>>) : String{
        // 전방 위치
        val centeri = IntArray(5) {i -> i + 15} // width
        val centerj = IntArray(15) {i -> i} // height
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
                else if (label == "roadway_crosswalk"){
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
            tempStatus[1] = statusDistinguish(patch)
            tempFront[1] = frontDistinguish(patch)
            tempLeftUp[1] = leftUpDistinguish(patch)
            //tempLeftDown[1] = leftDownDistinguish(patch)
            tempRightUp[1] = rightUpDistinguish(patch)
            //tempRightDown[1] = rightDownDistinguish(patch)
            this.TTScallback?.invoke("현재 위치에 ${KoreanClass[tempStatus[1]]!!} 있습니다.")
        }
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
            if (location == 1 || location == 2 || location == 4) {
                TTScallback(location, temp[0])
            }
            temp[1] = temp[0]
            tempStack -= count
        }
        Log.d("stack", "$tempStack")
        return Pair(temp, tempStack)
    }

    private fun TTScallback(location : Int, label : String?){
        val locationlabel = mapOf(1 to "전방에",
            2 to "좌측에",
            4 to "우측에")

        this.TTScallback?.invoke("${locationlabel[location]} ${KoreanClass[label]!!} 있습니다.")
    }
    companion object {
        private var tempStatus : MutableList<String?> = mutableListOf(null, null)//temp, prev
        private var tempFront : MutableList<String?> = mutableListOf(null, null)  //temp, prev
        private var tempLeftUp : MutableList<String?> = mutableListOf(null, null)//temp, prev
        private var tempLeftDown : MutableList<String?> = mutableListOf(null, null) //temp, prev
        private var tempRightUp : MutableList<String?>  = mutableListOf(null, null)//temp, prev
        private var tempRightDown : MutableList<String?>  = mutableListOf(null, null) //temp, prev
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


        val rankRoad : Map<String, Int> = mapOf("alley" to 4,
            "sidewalk" to 8,
            "bike" to 5,
            "caution" to 7,
            "braille" to 1,
            "background" to 6,
            "roadway" to 3,
            "crosswalk" to 2) // 서열(낮을수록 높은것)

        private val count : Int = 5
    }
}