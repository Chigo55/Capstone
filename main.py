import time
import cv2
import RPi.GPIO as GPIO

from collections import deque
from threading import Thread

from module import speechToText, chatGPT, textToSpeech, faceDetect, timeToGreeting
from prefferedsoundplayer import playsound

if __name__ == "__main__":
    try:
        
        # 플레그
        is_greeting = False

        # api_key 설정
        stt_api = "./speech-to-text.json"
        gpt_api = "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"
        tts_api = "./text-to-speech.json"

        # face_detect classifier 설정
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        
        # 데크 정의
        scripts = deque()
        
        # 모터 제어 설정
        GPIO.setmode(GPIO.BOARD) #Use Board numerotation mode
        GPIO.setwarnings(False) #Disable warnings   
        
        # 서브 스레드 생성
        thread_face = Thread(target=faceDetect, args=(face_classifier,), daemon=True) 
        thread_stt = Thread(target=speechToText, args=(scripts, stt_api,), daemon=True)
        
        # 서브 스레드 시작
        thread_face.start()
        thread_stt.start()
        
        # 메인 스레드 시작
        while True:
            if scripts:
                text = scripts.popleft()
                if text.strip() == "안녕 다비":
                    print("stt")
                    is_greeting = True

                    times = time.localtime().tm_hour
                    timeToGreeting(times, is_greeting)
                        
                elif is_greeting:
                    if text.strip() == "잘가 다비":
                        is_greeting = False
                        
                        times = time.localtime().tm_hour
                        timeToGreeting(times, is_greeting)
                        break
                    
                    gpt_answer = chatGPT(message=text.strip(), api_key=gpt_api)
                    textToSpeech(text=gpt_answer, api_key=tts_api)
                    time.sleep(3)
                    playsound("gpt-answer.mp3")

    except KeyboardInterrupt:
        pass