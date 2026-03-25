import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from st_clickable_images import clickable_images
import os, json, base64

# RAG 관련
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

#노란 박스 숨기기
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# 기본 설정
st.set_page_config(layout="wide", page_title="세종대왕과 표준말 배틀")

load_dotenv()
client = OpenAI()


# 이미지, 영상, doc, db 파일 경로
BASE_DIR= os.path.dirname(os.path.abspath(__file__))
SMART_IMAGE_PATH= os.path.join(BASE_DIR,"IMAGES","start_image.PNG")
NEXT_IMAGE_PATH= os.path.join(BASE_DIR,"IMAGES","next_image.PNG")
ANRY_KING_PATH = os.path.join(BASE_DIR,"IMAGES","angry_king.mp4")
SMILE_KING_PATH= os.path.join(BASE_DIR,"IMAGES","smile_king.mp4")
SEJONG_JSON_PATH= os.path.join(BASE_DIR,"SEJONG_DOCS.json")
CHROMA_DIR= os.path.join(BASE_DIR,"chroma_db")

# 세션 상태 초기화 (화면 전환용)
if "page" not in st.session_state:
    st.session_state.page = "start"   # "start" -> "game"


# 이미지를 base64로 변환 (st_clickable_images 요구사항)
def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"


# RAG 초기화 (앱 시작 시 1회만 실행)
@st.cache_resource
def init_rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists(CHROMA_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    else:
        with open(SEJONG_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    return retriever, llm
retriever, llm = init_rag()


# 세종대왕 채점 함수 (RAG 연결)
SEJONG_PROMPT = ChatPromptTemplate.from_template("""
당신은 조선 4대왕 세종대왕입니다.
백성을 사랑하고 우리말을 소중히 여기는 성군입니다.
사용자가 신조어, 외래어, 줄임말을 쓰면 크게 혼을 내고 올바른 우리말로 고쳐주어야 합니다.
말투는 ~하노라, ~하거늘, ~하옵니다 등 고어체를 쓰도록 합니다.
아래 실록 기록을 참고하여 답변하세요.

[점수 기준]
외래어/신조어/줄임말 0개 -> 100점
외래어/신조어/줄임말 1개 -> 80점
외래어/신조어/줄임말 2개 -> 60점
외래어/신조어/줄임말 3개 -> 40점
외래어/신조어/줄임말 4개 -> 20점
외래어/신조어/줄임말 5개 이상 -> 0점

[참고 기록]
{context}

[사용자 입력]
{question}

아래 JSON 형식으로만 답변하라 (다른 말 절대 금지):
{{"score": 점수, "answer": "세종대왕 답변"}}
""")

def sejong_answer(user_input: str) -> dict:
    docs = retriever.invoke(user_input)
    context = "\n".join([d.page_content for d in docs])
    chain = SEJONG_PROMPT | llm
    response = chain.invoke({"context": context, "question": user_input})
    raw = response.content.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# TTS: 세종대왕 음성 생성 후 Streamlit에서 재생
def speak_and_play(text: str):
    """답변 텍스트 → mp3 생성 → st.audio()로 자동 재생"""
    filename = os.path.join(BASE_DIR, "tts_sejong.mp3")
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",          # 중후한 남성 목소리 (세종대왕 느낌)
        input=text
    ) as response:
        response.stream_to_file(filename)
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)


# 영상 표시 (버그 수정 버전)
def show_king_reaction(score: int):
    if score < 80:
        if os.path.exists(ANRY_KING_PATH):
            st.video(ANRY_KING_PATH)
        else:
            st.error("😡: " + ANRY_KING_PATH)
    else:
        if os.path.exists(SMILE_KING_PATH):
            st.video(SMILE_KING_PATH)
        else:
            st.success("😊: " + SMILE_KING_PATH)


# 화면 1: 시작 화면 (이미지 클릭 -> 게임 화면으로 이동)
def page_start():
    st.title("👑 LLM GAME - 세종대왕님과 표준말 배틀")
    st.markdown("---")

    if os.path.exists(SMART_IMAGE_PATH):
        # st_clickable_images: 이미지 어디를 클릭해도 index(0) 반환
        clicked = clickable_images(
            [img_to_b64(SMART_IMAGE_PATH)],
            titles=["게임 시작"],
            div_style={
                "display": "flex",
                "justify-content": "center",
                "cursor": "pointer"
            },
            img_style={
                "width": "100%",
                "max-width": "800px"
            }
        )
        if clicked == 0:
            st.session_state.page = "game"
            st.rerun()

        st.markdown(
            "<p style='text-align:center; color:gray; font-size:14px;'>"
            "이미지를 클릭하면 게임이 시작됩니다 🖱️</p>",
            unsafe_allow_html=True
        )
    else:
        st.warning("시작 이미지를 찾을 수 없습니다: " + SMART_IMAGE_PATH)
        if st.button("게임 시작", type="primary"):
            st.session_state.page = "game"
            st.rerun()


# 화면 2: 게임 화면
def page_game():
    st.title("👑 LLM GAME - 세종대왕님과 표준말 배틀")
    st.subheader("--🔍-- Game Rule --🔍--")
    st.text("1️⃣ 세종대왕님 앞에서 외래어, 줄임말, 신조어를 쓰지 마세요.")
    st.text("2️⃣ 우리말을 헤치는 단어를 사용하면 점수가 깎입니다!")
    st.text("3️⃣ 완벽히 우리말을 지킨다면 100점 만점!")
    st.markdown("---")

    if os.path.exists(NEXT_IMAGE_PATH):
        st.image(NEXT_IMAGE_PATH, use_container_width=True)
        st.markdown("---")

    user_input = st.text_area("📝 세종대왕님께 드릴 말씀을 입력하세요", height=100)

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        battle = st.button("⚔️ 배틀 시작!", type="primary")
    with col_btn2:
        if st.button("🔙 처음으로"):
            st.session_state.page = "start"
            st.rerun()

    if battle:
        if not user_input.strip():
            st.warning("문장을 입력해주세요!")
        else:
            with st.spinner("세종대왕님께서 검토 중이시옵니다... 👑"):
                try:
                    result = sejong_answer(user_input)
                    score  = result["score"]
                    answer = result["answer"]

                    st.markdown("---")

                    # 1. 영상 먼저 표시 (싱크를 위해 음성보다 먼저 배치)
                    # show_king_reaction 함수 내부에 st.video(..., autoplay=True, muted=True)가 있어야 함
                    show_king_reaction(score)
                    st.markdown("---")

                    # 2. 결과 출력 레이아웃 (점수와 대사 박스 정렬)
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # 점수 박스
                        score_container = st.container(border=True)
                        score_container.metric(
                            label="📊 점수",
                            value=f"{score}점",
                            delta="만점!" if score == 100 else f"-{100 - score}점 감점"
                        )
                        if score == 100:
                            st.success("🏆 완벽하도다!")
                        elif score >= 60:
                            st.warning("⚠️ 부족하도다...")
                        else:
                            st.error("💢 노여움을 샀다!")

                    with col2:
                        st.subheader("👑 세종대왕의 말씀")
                        st.info(answer)
                        
                        # 음성 재생을 대사 바로 아래에 배치
                        st.markdown("🔊 **세종대왕의 음성**")
                        speak_and_play(answer)

                except json.JSONDecodeError:
                    st.error("응답 파싱 오류입니다. 다시 시도해주세요.")
                except Exception as e:
                    st.error(f"오류 발생: {e}")

# 메인: page 상태에 따라 화면 분기
if st.session_state.page == "start":
    page_start()
elif st.session_state.page == "game":
    page_game()
