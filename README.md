# 👑 LLM GAME - 세종대왕과 표준말 배틀

세종대왕 캐릭터 기반의 한국어 순화 게임입니다.  
사용자가 입력한 문장에 외래어·신조어·줄임말이 얼마나 포함됐는지 GPT-4o가 채점하고,  
세종실록 기반 RAG를 통해 세종대왕 고어체로 피드백을 제공합니다.

---

## ⚙️ 환경 설정

### 1. 패키지 설치

```bash
pip install streamlit openai python-dotenv
pip install langchain-openai langchain-chroma langchain-core
pip install st-clickable-images
pip install chromadb
```

### 2. .env 파일 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래 키를 입력합니다.

```
OPENAI_API_KEY=sk-...
KOREAN_DICT_KEY=...
```

> **API 키 발급처**
> - OpenAI : https://platform.openai.com
> - 국립국어원 표준국어대사전 : https://stdict.korean.go.kr/openapi/openApiInfo.do
> - 한국고전종합DB (별도 키 없이 사용 가능) : http://db.itkc.or.kr

### 3. 앱 실행

```bash
streamlit run LLM_GAME1.py
```

---

## 🎮 게임 흐름

1. [시작 화면]start_image.PNG 클릭
        ↓
2. [게임 화면] 문장 입력 → ⚔️ 배틀 시작! 버튼 클릭
        ↓
3. RAG: 세종실록에서 관련 문서 3개 검색
        ↓
4. GPT-4o: 외래어·신조어·줄임말 개수 채점
        + 세종대왕 고어체 답변 생성 (JSON 반환)
        ↓
5. 점수에 따른 영상 표시 (영상 먼저)
   - 점수 < 80  → angry_king.mp4
   - 점수 >= 80 → smile_king.mp4
        ↓
6. 점수 박스 + 세종대왕 답변 텍스트
7. 세종대왕 음성 자동 재생 (TTS)
        ↓
8. 🔙 처음으로 버튼 → 시작 화면으로 복귀


---

## 📊 점수 기준

| 외래어·신조어·줄임말 개수 | 점수 | 피드백 |
|:---:|:---:|:---:|
| 0개 | 100점 | 🏆 완벽하도다! |
| 1개 | 80점  | 🏆 완벽하도다! |
| 2~3개 | 40~60점 | ⚠️ 부족하도다... |
| 4개 이상 | 0~20점 | 💢 노여움을 샀다! |

---

## 🔧 주요 기능 설명

### 1. 외부 API 연동 (데이터 수집 단계)

쥬피터 노트북(`LLM_gmae.ipynb`)에서 아래 2가지 API로 데이터를 수집

| API | 용도 | 비고 |
|---|---|---|
| 국립국어원 표준국어대사전 | 표준어 단어 목록 수집 → `KOREAN_DICT.json` | API 키 필요 |
| 한국고전종합DB (세종실록) | 세종실록 본문 수집 → `SEJONG_DOCS.json` | 키 불필요 |

**국립국어원 API 호출 예시:**
```python
response = requests.get("https://stdict.korean.go.kr/api/search.do", params={
    "key": KOREAN_DICT_KEY,
    "q": "검색어",
    "req_type": "json",
    "num": 100,
    "start": 1
})
```

**한국고전종합DB (세종실록) 호출 예시:**
```python
# 수집 키워드: 훈민정음, 집현전, 백성, 한글, 세종, 대왕
response = requests.get("http://db.itkc.or.kr/openapi/search", params={
    "keyword": "훈민정음",
    "secId": "JR_BD",   # 신역 조선왕조실록 본문
    "rows": 20
})
# 수집된 데이터: SEJONG_DOCS.json 저장 (총 277개 Document)
```

---

### 2. RAG (Retrieval-Augmented Generation)

세종실록 문서를 벡터DB에 저장하고, 사용자 입력과 의미적으로 유사한 문서를 검색해 GPT의 답변 근거로 제공합니다.

```python
@st.cache_resource          # 앱 시작 시 1회만 실행
def init_rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(CHROMA_DIR):
        vectorstore = Chroma(persist_directory=CHROMA_DIR, ...)  # 기존 DB 로드
    else:
        # SEJONG_DOCS.json → Document 변환 → Chroma DB 생성
        vectorstore = Chroma.from_documents(docs, embedding=embeddings, ...)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 상위 3개 검색
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    return retriever, llm
```

---

### 3. 세종대왕 프롬프트 설계

프롬프팅 기본 구성 **(Instruction / Context / Input / Output)** 과 **ReAct 기법**을 적용했습니다.

```
[Instruction]  역할: 조선 4대왕 세종대왕
               목표: 외래어·신조어·줄임말 채점 후 고어체로 피드백
               제약: JSON 형식으로만 출력

[Context]      세종실록 관련 기록 (RAG로 검색된 3개 문서)

[Input]        사용자가 입력한 문장

[Output]       {"score": 점수, "answer": "세종대왕 답변"}
```

```python
SEJONG_PROMPT = ChatPromptTemplate.from_template("""
당신은 조선 4대왕 세종대왕입니다.
...
[참고 기록]
{context}

[사용자 입력]
{question}

아래 JSON 형식으로만 답변하라 (다른 말 절대 금지):
{{"score": 점수, "answer": "세종대왕 답변"}}
""")
```

---

### 4. TTS (Text-to-Speech)

세종대왕의 답변 텍스트를 음성으로 변환해 자동 재생
`세종대왕의 말씀` 텍스트 바로 아래에 음성 플레이어가 배치

```python
def speak_and_play(text: str):
    filename = os.path.join(BASE_DIR, "tts_sejong.mp3")
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",       # 중후한 남성 목소리
        input=text
    ) as response:
        response.stream_to_file(filename)
    with open(filename, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
```

---

### 5. 화면 전환 (st_clickable_images + session_state)

`st.session_state`로 현재 페이지 상태를 관리  
시작 이미지를 클릭하면 게임 화면으로 전환

```python
# 세션 초기화
if "page" not in st.session_state:
    st.session_state.page = "start"

# 이미지 클릭 감지 (어디를 클릭해도 clicked=0 반환)
clicked = clickable_images([img_to_b64(SMART_IMAGE_PATH)], ...)
if clicked == 0:
    st.session_state.page = "game"
    st.rerun()

# 페이지 분기
if st.session_state.page == "start":
    page_start()
elif st.session_state.page == "game":
    page_game()
```

---

## 🖥️ 결과 화면 레이아웃

배틀 결과는 아래 순서로 표시

```
① 세종대왕 영상  (점수에 따라 화난 왕 / 웃는 왕)
────────────────────────────────
② [ 점수 박스    |  세종대왕의 말씀 (텍스트)  ]
   [ 피드백 메시지 |  🔊 세종대왕의 음성 (TTS) ]
```

`st.columns([1, 2])` 로 점수와 대사를 나란히 배치
`st.container(border=True)` 로 점수 박스에 테두리를 추가

---

## 🐛 수정된 버그 내역

| 위치 | 버그 | 수정 내용 |
|---|---|---|
| `show_king_reaction()` | `word >= 80()` 숫자에 괄호 | `score < 80` 으로 수정 |
| `show_king_reaction()` | `print(st.video(...))` | `st.video(...)` 로 수정 |
| `show_king_reaction()` | `'ANRY_KING_PATH'` 따옴표로 문자열 취급 | `ANRY_KING_PATH` 변수로 수정 |
| `page_game()` | `use_column_width=True` deprecated 경고 | `use_container_width=True` 로 수정 |
| 전체 | Streamlit UserWarning 노란 박스 출력 | `warnings.filterwarnings('ignore')` 추가 |

---

## 🔄 주요 변경 이력

| 버전 | 변경 내용 |
|---|---|
| 초기 | 기본 Streamlit UI (텍스트 입력, 이미지 표시만) |
| v2 | 국립국어원 API / 세종실록 API 연동, `SEJONG_DOCS.json` 생성 |
| v3 | Chroma 벡터DB 구성, RAG 파이프라인 완성, 세종대왕 프롬프트 + 점수 시스템 완성 |
| v4 | `LLM_GAME.py` 에 RAG 통합, TTS 음성 재생, 이미지 클릭 화면 전환 연결 |
| **v5 (최종)** | 영상을 결과보다 먼저 표시, `st.container(border=True)` 점수 박스, `use_container_width` 경고 수정, `warnings` 노란 박스 제거 |

---

## 📚 사용 기술 스택

| 항목 | 내용 |
|---|---|
| LLM | OpenAI GPT-4o |
| TTS | OpenAI gpt-4o-mini-tts (`onyx` 목소리) |
| 임베딩 | OpenAI text-embedding-3-small |
| 벡터DB | Chroma (로컬 저장) |
| RAG 프레임워크 | LangChain (langchain-openai, langchain-chroma) |
| 웹 프레임워크 | Streamlit |
| 이미지 클릭 | st-clickable-images |
| 데이터 수집 | 국립국어원 API, 한국고전종합DB API |
