"""
Advanced RAG 검색 방법 비교기 (Streamlit) — Session 05 실습
==========================================================
같은 질문을 여러 검색 방법에 동시에 던져 결과를 나란히 비교합니다.

사용법:
    1) cd streamlit_app
    2) pip install -r requirements.txt
    3) streamlit run app.py

비교 대상:
    - 기본 RAG      : 의미(벡터) 검색만
    - BM25          : 키워드 검색만
    - Ensemble      : 위 둘을 RRF로 합침
    - Reranking     : 넓게 찾고 Cross-encoder로 순서 다시 매김
    - Parent Doc    : 작은 조각으로 찾고 큰 조각을 돌려줌
    - HyDE          : LLM이 가짜 답변을 만들고 그걸로 검색 (API 키 필요)

API 키가 없어도 HyDE와 답변 생성만 빠지고 나머지는 모두 동작합니다.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore

load_dotenv()

# === 설정 =========================================================
EMBEDDING_MODEL = "BM-K/KoSimCSE-roberta-multitask"   # 노트북과 동일
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"            # 노트북과 동일

st.set_page_config(page_title="Advanced RAG 비교기", page_icon="🔍", layout="wide")


# 검색된 문서만 근거로 답하게 하는 프롬프트.
# "없으면 없다고 하라"는 줄이 없으면 LLM이 아는 지식으로 지어낸다(환각).
ANSWER_PROMPT = """아래 [참고 문서]만 사용해 질문에 한국어로 두 문장 이내로 답하세요.
문서에 답이 없으면 반드시 "제공된 문서에서 찾을 수 없습니다" 라고만 답하세요.
추측하거나 아는 지식을 덧붙이지 마세요.

[참고 문서]
{context}

질문: {question}
답변:"""


# === 샘플 문서 (PDF 없이 바로 써보기) ===============================
SAMPLE_DOCS = [
    ("소형 언어 모델", """소형 언어 모델(sLLM)은 1B에서 7B 규모의 경량 언어 모델입니다.
Phi-3, Gemma, Qwen2.5 등이 대표적이며 온디바이스 추론과 저비용 서빙에 적합합니다.
큰 모델 대비 성능 손실이 크지 않아 특정 도메인에서는 충분한 대안이 됩니다."""),
    ("PEFT", """LoRA는 저랭크 행렬 분해로 학습 파라미터 수를 줄이는 PEFT 기법입니다.
QLoRA는 4bit 양자화와 LoRA를 결합해 RTX 4060 같은 소비자급 GPU에서도 7B 모델 학습을 가능하게 합니다.
어댑터, 프리픽스 튜닝, IA3 등도 대표적인 PEFT 계열입니다."""),
    ("양자화", """양자화는 모델 가중치를 저정밀도로 바꿔 메모리를 줄이는 기법입니다.
GPTQ와 AWQ는 대표적인 사후 양자화 방식이며, GGUF는 CPU 추론에 널리 쓰이는 파일 포맷입니다.
비트를 낮출수록 메모리는 줄지만 정확도 손실이 생길 수 있습니다."""),
    ("벡터 DB", """벡터 데이터베이스는 고차원 벡터를 저장하고 유사도 검색을 수행합니다.
ChromaDB는 오픈소스 임베디드 벡터 DB이고, FAISS는 Meta가 만든 고속 검색 라이브러리입니다.
Weaviate는 GraphQL 기반 검색 엔진이며, Pinecone은 관리형 클라우드 서비스입니다."""),
    ("ANN 인덱스", """HNSW는 계층 그래프를 타고 이웃을 탐색하는 근사 최근접 이웃 알고리즘입니다.
IVF는 벡터를 클러스터로 나눈 뒤 가까운 클러스터만 탐색하며, nprobe로 정확도를 조절합니다.
PQ는 벡터를 압축해 메모리를 절감하지만 정확도를 일부 희생합니다."""),
    ("정렬", """RLHF는 사람의 선호를 보상 모델로 학습해 정책에 반영하는 정렬 기법입니다.
DPO는 보상 모델 없이 선호 쌍으로 직접 최적화합니다.
이런 정렬 기법은 모델의 안전성과 유용성을 높이는 데 사용됩니다."""),
    ("강화학습", """PPO는 정책이 한 번에 크게 변하지 않도록 제한해 안정적인 업데이트를 보장합니다.
GRPO는 DeepSeek이 제안한 효율적인 정책 최적화 방법입니다.
GRPO는 가치 함수 없이 그룹 내 상대 비교로 이점을 추정해 메모리를 아낍니다."""),
    ("프롬프트", """Chain-of-Thought는 모델이 단계별 추론 과정을 서술하도록 유도하는 기법입니다.
복잡한 수리·논리 문제에서 정답률을 크게 끌어올립니다.
Self-Consistency는 여러 추론 경로를 뽑아 다수결로 답을 정합니다."""),
    ("어텐션", """셀프 어텐션은 Query, Key, Value 세 행렬의 곱으로 토큰 간 관계를 계산합니다.
멀티헤드 어텐션은 여러 부분공간에서 병렬로 어텐션을 수행합니다.
Flash Attention은 메모리 접근을 최적화해 긴 시퀀스 학습 속도를 크게 높입니다.
그룹 쿼리 어텐션(GQA)은 Key와 Value 헤드를 공유해 추론 메모리를 줄입니다."""),
    ("RAG 기본", """RAG는 외부 지식을 검색해 LLM 답변에 근거를 제공하는 기술입니다.
문서 로딩, 청킹, 임베딩, 검색, 생성의 다섯 단계로 구성됩니다.
파인튜닝 없이 최신 정보를 반영할 수 있다는 것이 장점입니다.
검색이 실패하면 아무리 좋은 LLM도 옳은 답을 낼 수 없습니다."""),
    ("하이브리드 검색", """하이브리드 검색은 BM25 같은 키워드 검색과 벡터 검색을 결합합니다.
고유명사나 코드처럼 정확한 토큰 일치가 중요한 질의에서는 키워드 검색이 강합니다.
RRF는 두 순위 목록을 상호 순위 역수로 융합하는 방법입니다."""),
    ("임베딩", """임베딩은 텍스트를 고차원 벡터로 바꿔 의미를 수치화하는 과정입니다.
의미가 비슷한 문장은 벡터 공간에서 가까이 놓입니다.
한국어에는 KoSimCSE, multilingual-e5 같은 한국어 지원 모델이 적합합니다.
영어 전용 모델을 한국어에 쓰면 검색 품질이 급격히 무너집니다."""),
    ("RAG 평가", """RAGAS는 RAG 파이프라인을 지표화해 평가하는 프레임워크입니다.
Faithfulness는 답변이 근거 문서에 충실한지를 봅니다.
검색 품질은 정답 문서를 얼마나 잘 올렸는지로 따로 재야 합니다."""),
    ("에이전트", """LLM 에이전트는 도구를 호출하며 여러 단계로 문제를 해결하는 시스템입니다.
ReAct는 추론과 행동을 번갈아 수행하는 대표적인 패턴입니다.
MCP는 모델과 외부 도구를 연결하는 표준 프로토콜입니다."""),
]


# === 모델 로딩 (앱 실행 중 한 번만) ==================================
@st.cache_resource(show_spinner="임베딩 모델 로딩 중... (최초 1회)")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Reranker 로딩 중... (최초 1회, 약 2.2GB)")
def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL, max_length=512)


# === 인덱스 구축 ===================================================
@st.cache_resource(show_spinner="문서 인덱싱 중...")
def build_index(file_key, chunk_size: int, chunk_overlap: int, _docs: list[Document]):
    """검색에 필요한 것들을 한 번에 만든다. file_key는 캐시 키 역할."""
    embeddings = load_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(_docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=f"cmp_{abs(hash(file_key)) % 10**8}",
        collection_metadata={"hnsw:space": "cosine"},
    )

    bm25 = BM25Retriever.from_documents(splits)

    # Parent Document — 작은 조각으로 찾고 큰 조각을 돌려준다
    parent_vs = Chroma(
        collection_name=f"par_{abs(hash(file_key)) % 10**8}",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    parent = ParentDocumentRetriever(
        vectorstore=parent_vs,
        docstore=InMemoryStore(),
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20),
        parent_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50),
    )
    parent.add_documents(_docs)

    return {"splits": splits, "vectorstore": vectorstore,
            "bm25": bm25, "parent": parent}


# === 검색 방법들 ===================================================
def search_basic(idx, query, k):
    return idx["vectorstore"].similarity_search(query, k=k)


def search_bm25(idx, query, k):
    idx["bm25"].k = k
    return idx["bm25"].invoke(query)


def search_ensemble(idx, query, k):
    idx["bm25"].k = k
    ens = EnsembleRetriever(
        retrievers=[idx["bm25"], idx["vectorstore"].as_retriever(search_kwargs={"k": k})],
        weights=[0.4, 0.6],      # 0.5/0.5로 두면 동점이 생겨 순서에 좌우된다
    )
    return ens.invoke(query)[:k]


def search_rerank(idx, query, k, fetch_k=None):
    """1단계로 넓게 뽑고, 2단계로 다시 채점한다.

    진단을 위해 후보 전체의 (1단계 등수, 리랭커 점수)를 세션에 남긴다.
    리랭킹이 이상하게 동작할 때 이 표를 보면 원인을 바로 알 수 있다.
    """
    fetch_k = fetch_k or st.session_state.get("fetch_k", 10)
    candidates = idx["vectorstore"].similarity_search(query, k=fetch_k)
    if not candidates:
        return []
    scores = load_reranker().predict([[query, d.page_content] for d in candidates])
    dense_rank = {id(d): i for i, d in enumerate(candidates, 1)}
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    st.session_state["rerank_trace"] = [
        {"리랭 등수": i, "1단계 등수": dense_rank[id(d)],
         "리랭커 점수": float(sc), "출처": d.metadata.get("source", "?"),
         "내용": d.page_content[:60].replace("\n", " ")}
        for i, (d, sc) in enumerate(ranked, 1)
    ]
    return [d for d, _ in ranked[:k]]


def search_parent(idx, query, k):
    idx["parent"].search_kwargs = {"k": k}
    return idx["parent"].invoke(query)[:k]


def make_hyde_search(llm):
    def search_hyde(idx, query, k):
        prompt = (f"다음 질문에 대한 답변을 기술 문서의 한 문단처럼 3문장 이내로 작성하세요.\n"
                  f"사실 여부는 중요하지 않습니다.\n\n질문: {query}\n\n답변 문단:")
        fake = llm.invoke(prompt)
        fake = fake if isinstance(fake, str) else fake.content
        st.session_state["last_hyde"] = fake.strip()
        return idx["vectorstore"].similarity_search(fake, k=k)
    return search_hyde


# === 사이드바 ======================================================
with st.sidebar:
    st.header("⚙️ 설정")

    st.subheader("📄 문서")
    source = st.radio("무엇으로 검색할까요?",
                      ["샘플 문서 8개 (바로 시작)", "내 PDF 올리기"],
                      label_visibility="collapsed")

    uploaded = None
    if source == "내 PDF 올리기":
        uploaded = st.file_uploader("PDF 선택", type=["pdf"], accept_multiple_files=True)

    st.divider()
    st.subheader("✂️ 조각내기")
    chunk_size = st.slider("조각 크기 (자)", 100, 1000, 200, 50)
    chunk_overlap = st.slider("겹치는 부분 (자)", 0, 200, 30, 10)
    top_k = st.slider("가져올 개수", 1, 5, 3)
    st.slider("Reranking 1단계 후보 수", 5, 30, 10, 5, key="fetch_k",
              help="여기서 뽑은 후보 안에서만 순서를 다시 매깁니다. "
                   "후보에 정답이 없으면 리랭킹으로는 절대 찾을 수 없습니다.")

    st.divider()
    st.subheader("🔍 비교할 방법")
    api_key = os.getenv("OPENAI_API_KEY")
    has_key = bool(api_key) and not api_key.startswith("sk-your")

    picks = {
        "기본 RAG": st.checkbox("기본 RAG (의미 검색)", value=True),
        "BM25": st.checkbox("BM25 (키워드 검색)", value=False),
        "Ensemble": st.checkbox("Ensemble (BM25+의미)", value=True),
        "Reranking": st.checkbox("Reranking (2단계)", value=True),
        "Parent Doc": st.checkbox("Parent Document", value=False),
        "HyDE": st.checkbox("HyDE (API 키 필요)", value=False, disabled=not has_key),
    }

    st.divider()
    st.subheader("💬 답변 생성")
    make_answer = st.checkbox(
        "검색 결과로 답변까지 만들기", value=has_key, disabled=not has_key,
        help="검색된 문서만 근거로 LLM이 답변을 만듭니다. OpenAI API 키가 필요합니다.",
    )
    st.caption("문서에 답이 없으면 '찾을 수 없습니다'라고 답하도록 지시합니다.")

    st.divider()
    st.caption(f"임베딩: `{EMBEDDING_MODEL}`")
    st.caption(f"Reranker: `{RERANKER_MODEL}`")
    if has_key:
        st.success("OpenAI API 키 확인됨")
    else:
        st.info("API 키 없음 — HyDE와 답변 생성은 비활성화됩니다")


# === 본문 ==========================================================
st.title("🔍 Advanced RAG 검색 방법 비교기")
st.caption("같은 질문을 여러 검색 방법에 동시에 던져, 무엇을 가져오는지 나란히 봅니다.")

# 문서 준비
if source == "내 PDF 올리기":
    if not uploaded:
        st.info("👈 왼쪽에서 PDF를 올리거나 '샘플 문서'를 선택하세요.")
        st.stop()
    docs, key_parts = [], []
    for f in uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
            for p in pages:
                p.metadata["source"] = f.name
            docs.extend(pages)
        finally:
            os.unlink(tmp_path)
        key_parts.append(f"{f.name}:{len(f.getvalue())}")
    file_key = "|".join(key_parts)
else:
    docs = [Document(page_content=body, metadata={"source": f"{name}.txt"})
            for name, body in SAMPLE_DOCS]
    file_key = "sample-v2"

idx = build_index(file_key, chunk_size, chunk_overlap, docs)
st.success(f"문서 {len(docs)}개 → 조각 {len(idx['splits'])}개 준비 완료")

# 검색 방법 모으기
llm = None
if has_key:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

METHODS = {
    "기본 RAG": search_basic,
    "BM25": search_bm25,
    "Ensemble": search_ensemble,
    "Reranking": search_rerank,
    "Parent Doc": search_parent,
}
if llm is not None:
    METHODS["HyDE"] = make_hyde_search(llm)

selected = [name for name, on in picks.items() if on and name in METHODS]

# 질문 입력
examples = ["GRPO가 뭐야?", "RTX 4060으로도 학습이 되나?",
            "CPU에서 모델 돌릴 때 쓰는 파일 포맷은?", "단계별로 생각하게 시키면 정답률이 오르나?"]
st.write("**예시 질문** (누르면 입력됩니다)")
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["query"] = ex

query = st.text_input("질문", key="query", placeholder="문서에 대해 물어보세요…")

if not query:
    st.stop()
if not selected:
    st.warning("👈 비교할 방법을 하나 이상 선택하세요.")
    st.stop()

# 무거운 모델은 측정 전에 미리 로딩한다.
# (로딩 시간이 검색 시간에 섞이면 Reranking이 수천 ms로 보인다)
if "Reranking" in selected:
    load_reranker()

# 검색 실행
results, timings = {}, {}
for name in selected:
    start = time.time()
    try:
        results[name] = METHODS[name](idx, query, top_k)
        timings[name] = (time.time() - start) * 1000
    except Exception as e:
        results[name] = []
        timings[name] = 0
        st.error(f"{name} 실패: {e}")

# 속도 요약
st.subheader("⏱️ 검색 시간")
tcols = st.columns(len(selected))
fastest = min(timings.values()) if timings else 1
for col, name in zip(tcols, selected):
    ms = timings[name]
    ratio = ms / fastest if fastest else 1
    col.metric(name, f"{ms:.0f}ms",
               delta=None if ratio < 1.5 else f"{ratio:.0f}배 느림",
               delta_color="inverse")

if "HyDE" in selected and st.session_state.get("last_hyde"):
    with st.expander("🔮 HyDE가 지어낸 가짜 문서 보기"):
        st.write(st.session_state["last_hyde"])
        st.caption("이 문서로 검색합니다. 엉뚱한 분야를 지어내면 검색도 함께 틀어집니다.")

# 결과 나란히
st.subheader("📄 가져온 문서")
rcols = st.columns(len(selected))
for col, name in zip(rcols, selected):
    with col:
        st.markdown(f"### {name}")
        docs_found = results[name]
        if not docs_found:
            st.write("결과 없음")
            continue
        for rank, d in enumerate(docs_found, 1):
            src = d.metadata.get("source", "?")
            st.markdown(f"**{rank}위** · `{src}`")
            st.caption(d.page_content[:200].replace("\n", " ") +
                       ("..." if len(d.page_content) > 200 else ""))

# Reranking 진단 — 왜 그 문서를 골랐는지 보여준다
if "Reranking" in selected and st.session_state.get("rerank_trace"):
    trace = st.session_state["rerank_trace"]
    top_score = trace[0]["리랭커 점수"]
    with st.expander(f"🔬 Reranking 진단 — 1단계 후보 {len(trace)}개를 어떻게 재정렬했나", expanded=False):
        if top_score < 0.1:
            st.warning(
                f"리랭커 1위 점수가 **{top_score:.4f}** 로 매우 낮습니다. "
                "후보 중 질문에 맞는 문서가 없다고 판단한 것입니다.\n\n"
                "왼쪽에서 **1단계 후보 수**를 늘려보세요. 그래도 낮으면 "
                "청킹이나 임베딩 단계의 문제일 가능성이 큽니다."
            )
        st.dataframe(trace, hide_index=True)
        st.caption(
            "**읽는 법** — 리랭커 점수는 보통 정답 하나만 0.9 이상이고 나머지는 0에 가깝습니다. "
            "즉 리랭킹은 **1등에 몰아주는** 방식이라, 1등을 틀리면 2·3등도 같이 쓸모없어집니다. "
            "반면 기본 검색은 3개에 고르게 나눠 걸기 때문에, 1등을 틀려도 2·3등에서 건질 때가 있습니다. "
            "리랭킹이 평균으로는 이겨도 특정 질문에서는 기본 검색에 질 수 있는 이유입니다."
        )

# 몇 개가 겹치는지
if len(selected) > 1:
    st.subheader("🔗 방법끼리 얼마나 같은 문서를 가져왔나")
    base_name = selected[0]
    base_set = {d.page_content for d in results[base_name]}
    lines = []
    for name in selected[1:]:
        overlap = len(base_set & {d.page_content for d in results[name]})
        lines.append(f"- **{base_name}** vs **{name}** — {top_k}개 중 **{overlap}개** 같음")
    st.markdown("\n".join(lines))
    st.caption("겹치는 게 적을수록 두 방법이 서로 다른 문서를 본다는 뜻입니다.")

# 검색한 문서로 답변 만들기
if llm is not None and make_answer:
    st.subheader("💬 방법별 답변")
    st.caption("검색된 문서만 근거로 만든 답변입니다. "
               "검색이 실패하면 '찾을 수 없습니다'가 나옵니다 — 그것이 정상 동작입니다.")

    acols = st.columns(len(selected))
    성공 = []
    for col, name in zip(acols, selected):
        with col:
            st.markdown(f"### {name}")
            docs_found = results[name]
            if not docs_found:
                st.warning("검색 결과가 없어 답할 수 없습니다.")
                continue
            context = "\n\n".join(f"[문서 {i}] {d.page_content}"
                                  for i, d in enumerate(docs_found, 1))
            with st.spinner("생성 중..."):
                out = llm.invoke(ANSWER_PROMPT.format(context=context, question=query))
                text = (out if isinstance(out, str) else out.content).strip()
            if "찾을 수 없습니다" in text:
                st.error(text)
            else:
                st.success(text)
                성공.append(name)
            with st.expander("이 답변의 근거"):
                for rank, d in enumerate(docs_found, 1):
                    st.caption(f"{rank}. `{d.metadata.get('source','?')}` — "
                               f"{d.page_content[:120]}...")

    if len(selected) > 1:
        실패 = [n for n in selected if n not in 성공]
        if 실패 and 성공:
            st.info(f"**답변 성공**: {', '.join(성공)}  |  "
                    f"**답변 실패**: {', '.join(실패)}\n\n"
                    f"실패한 쪽은 LLM이 나빠서가 아니라 **검색이 근거 문서를 못 가져왔기 때문**입니다. "
                    f"검색을 고치지 않으면 LLM을 바꿔도 해결되지 않습니다.")
elif llm is None:
    st.caption("💬 답변 생성은 OpenAI API 키가 있을 때만 동작합니다. "
               "검색 비교는 키 없이도 전부 사용할 수 있습니다.")
