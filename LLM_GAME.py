import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from st_clickable_images import clickable_images
import speech_recognition as sr
import base64, os, time
from pathlib import Path

# 가로로 화면 넓히기
st.set_page_config(layout="wide")

load_dotenv()
client= OpenAI()
recognizer= sr.Recognizer()

SMART_IMAGE_PATH=("C:\\SKN-_24\\과제(LLM game)\\LLM_GAME\\IMAGES\\start_image.PNG")
NEXT_IMAGE_PATH=("C:\\SKN-_24\\과제(LLM game)\\LLM_GAME\\IMAGES\\next_image.PNG")
ANRY_KING_PATH=("C:\\SKN-_24\\과제(LLM game)\\LLM_GAME\\IMAGES\\angry_king.mp4")
SMILE_KING_PATH=("C:\\SKN-_24\\과제(LLM game)\\LLM_GAME\\IMAGES\\smile_king.mp4")

# 로컬에 있는 IMAGES 파일 사용
#def get_b64_img(path):
#    with open(path, "rb")

st.title('LLM GAME - 세종대왕님과 표준말 배틀')
st.subheader('--🔍--<Game Rule>--🔍--')
st.text('1️⃣ 세종대왕님 앞에서 외래어, 줄임말, 신조어를 쓰지 마세요. ')
st.text('2️⃣ 만약 우리말을 헤치는 단어를 사용하는 경우 -10점 받습니다!')
st.text('3️⃣ 완벽히 우리말을 지킨다면 + 5점 얻습니다!')

# 단락 나누기
# st.divider()
# 구분선 넣기
st.markdown("---")

# 시작 이미지 넣기(화면)
st.image(SMART_IMAGE_PATH, use_column_width=True)

message=st.text_area('문장 입력')
st.text(f"문장:{message}")

#TTS
def speak(text: str, voice: str, king_id:str):
    filename=f'tts_{king_id}.mp3'
    #Audio Speech API - TTS 음성 생성
    with client.audio.speech.with_streaming_response.create(
        model='gpt-4o-mini-tts',
        voice=voice,
        input=text
    ) as response:
        response.stream_to_file(filename)
    return filename

#SST
def listen():
    with sr.Microphone() as source:
        print('세종대왕님께 예를 갖추어 말하세요.')
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=5)
    return recognizer.recognize_google(audio, language='ko-KR')


#GPT
def chat(user_text:str, dim_id:str):
    dim=SEJONG_KING[dim_id]
    messages=[                                          #history: 지금까지의 대화 기록을 빈 리스트에서 쌓이도록 함
        {'role':'system','content':dim['system']}]+dim['history']+[{'role':'user','content':user_text}]
        #role: system(GPT 성격 및 역할 지정), user(사용자 입력)                         # 지금 내가 한 말 뒤에 붙음
        #Chat Completions API - GPT 텍스트 답변
    response=client.chat.completions.create(
        model='gpt-4o', #채팅 모델
        messages=messages,
        max_tokens=300
    )
    reply=response.choices[0].message.content

    # 다른 차원 '나'들의 history는 각각 저장
    dim['history'].append({'role':'user','content':user_text})
    dim['history'].append({'role':'assistant','content':reply}) #assistant:이전 GPT 답변 내용
    return reply

# 첫 대화
# 말하면 str로 인식 -> GPT:str 대답 -> 오디바 출력 및 파일 저장
def talk_to(king_id: str):
    king = SEJONG_KING[king_id]
    print(f'☎️ {king["label"]} (목소리: {king["voice"]}) ☎️ ')

    user_text = listen()
    if user_text in ['종료', '그만']:
        return False

    print(f'나: {user_text}')
    reply = chat(user_text, king_id)
    print(f'{king["label"]}: {reply}')

    filename = speak(reply, king['voice'], king_id)
    display(Audio(filename, autoplay=True))   # 파일과 동일
    return True

def  get_img(king_id: str):
    if word >= 80():
        print(st.video('ANRY_KING_PATH'))
    else:
        print(st.video('SMILE_KING_PATH'))

# 색 선택
#color = st.color_picker('색 선택')
#st.text(color)

