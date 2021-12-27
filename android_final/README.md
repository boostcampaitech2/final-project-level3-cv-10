# Android Studio, tflite를 이용한 어플리케이션
***

## 실험 기기
galaxy tab s6
사용 배터리 : 최소 밝기 기준, 약 5분에 70mAh<br>
사용 메모리 : 최대 270MB
***

## 후처리
- **전방 판별 기준** : width(0을 가장 왼쪽)을 4\~15, height을(0을 사용자와 가장 먼 기준)2\~9 에 해당하는 grid


- **사용자 위협 거리 판별기준** : width(0을 가장 왼쪽)을 5\~14, height을(0을 사용자와 가장 먼 기준)4\~11 에 해당하는 grid


- 전방을 기준으로 상황이 바뀌면 다음과 같이 동작합니다
  - 점자블록(braille)
    -  전방에 점자블록에 해당하는 grid가 5개 이상 존재할 시 전방에 점자블록이 있음을 판별
  - 장애물(Obstacle)
    - 장애물에 대한 label의 없는 관계로, "background"를 장애물로 판별
    ![image](https://user-images.githubusercontent.com/71861842/147430047-98ec6f62-298f-4dea-be1b-c6125486065e.png)

    - 전방에 장애물에 해당하는 grid가 존재할시 최빈값이 장애물일 때 혹은 **사용자 위협거리**에 해당하는 grid가 10개 이상일경우 장애물 판별
  - 주의구역(findNotice)
    - Notice에 해당하는 label : 과속방지턱(bump), 배수구(grating), 맨홀(manhole), 계단(stair), 나무가 존재하는 지역(tree)
    - 전방에 주의구역에 해당하는 grid가 존재할시 
      - 사용자 위협거리에 과속방지턱, 맨홀, 계단, 나무가 존재하는 지역이 존재하는지 각각 판별
      - 해당하는 grid가 10개 이상 존재할 시 해당하는 label을 반환
      - 존재하지 않을시 notice label을 무시
  - 이외의 label은 최빈값을 이용해 계산합니다.


- 전방이 현재 label과 다르고, 2번이상 연속된 label이 등장할 시에 지정된 문구를 TTS로 읽어주고, 현재 상황을 등장한 label로 변경합니다.
- TTS는 Android의 TextToSpeech library를 사용하였습니다. https://developer.android.com/reference/android/speech/tts/TextToSpeech
***

## reference
https://www.tensorflow.org/lite/examples/segmentation/overview
