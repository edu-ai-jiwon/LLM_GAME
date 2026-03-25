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

# 색 선택
#color = st.color_picker('색 선택')
#st.text(color)

#SEGONG={
#    []
#}

