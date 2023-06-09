from __future__ import division
from six.moves import queue

from google.cloud import speech
from google.cloud import texttospeech

# 아래 두개 라이브러리는 라즈베리파이에서만 동작
# The following two libraries only work on Raspberry Pi.
import RPi.GPIO as GPIO
from prefferedsoundplayer import playsound

import re
import sys
import os
import time

import cv2
import pyaudio
import openai

# 음성 -> 텍스트 합성 클레스
class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

# 음성 -> 텍스트 출력 함수
def listen_print_loop(responses, deque):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0
            deque.append(transcript + overwrite_chars)

# Speech To Text 함수
def speechToText(deque, api_key):
    credential_path_stt = api_key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path_stt
    
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "ko-KR"

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(16000, int(16000 / 10)) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        listen_print_loop(responses, deque=deque)

# 시간대별 인사 함수
def timeToGreeting(time, greeting):
    if greeting:
        if time >= 5 and time <= 9:
            # 아침인사
            playsound("./audio/morning.mp3")
        elif time >= 12 and time <= 13:
            # 점심 인사
            playsound("./audio/lunch.mp3")
        elif time >= 19:
            # 저녁 인사
            playsound("./audio/dinner.mp3")
        else:
            playsound("./audio/hello.mp3")
    else:
        playsound("./audio/bye.mp3")

# GPT 함수
def chatGPT(message, api_key):
    openai.api_key = api_key
    
    messages = [{"role": "user", "content":  message}]
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        
    reply = chat.choices[0].message.content
    print(f"ChatGPT: {reply}")
    return reply

# Text to Speech 함수
def textToSpeech(text, api_key):
    credential_path_tts = api_key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path_tts

    """Synthesizes speech from the input string of text."""
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        # language_code="en-US",
        language_code="ko-KR",
        name="ko-KR-Neural2-C",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    # Note: you can pass in multiple effects_profile_id. They will be applied
    # in the same order they are provided.
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("./audio/gpt-answer.wav", "wb") as out:
        out.write(response.audio_content)

# 카메라 동작 및 얼굴 찾기 함수      
def faceDetect(classifier):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    while True: 
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        faces = classifier.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        
        for x, y, w, h in faces:
            # TODO: 카메라와 얼굴의 각도 계산 코드 작성 필요
            # motorControl(angle=, motor_pin=)
            pass

# 각도 pwm 변환 함수
def angleToPercent (angle):
    if angle > 180 or angle < 0 :
        return False

    start = 4
    end = 12.5
    ratio = (end - start)/180 #Calcul ratio from angle to percent

    angle_as_percent = angle * ratio

    return start + angle_as_percent

# 모터 제어 함수
def motorControl(angle, motor_pin):

    GPIO.setup(motor_pin, GPIO.OUT)
    pwm = GPIO.PWM(motor_pin, 50)

    pwm.ChangeDutyCycle(angleToPercent(angle=angle))
    time.sleep(1)

    #Close GPIO & cleanup
    pwm.stop()
    GPIO.cleanup()