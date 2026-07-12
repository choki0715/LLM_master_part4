# LLM 마스터 과정 특별 교육 시리즈 — Part 4

> **지식증강 — 벡터 RAG & 그래프 RAG & 온톨로지 RAG** — LLM의 지식 한계를 외부 지식으로 보강하는 검색 증강 생성(RAG)을 밑바닥부터 고급 기법까지 다루는 실습 중심 과정

본 저장소는 **LLM 마스터 과정 특별 교육 시리즈(총 5개 파트)** 중 **Part 4**의 실습 자료입니다.
RAG 파이프라인의 기본(임베딩·시맨틱 서치·ChromaDB)에서 출발해 벡터 DB 비교, LangChain RAG 애플리케이션 구현, HyDE·Reranking·Ensemble 등 Advanced RAG, 지식 그래프 기반 그래프 RAG, RDF·OWL 추론의 온톨로지 RAG, 그리고 RAGAS·LLM-as-a-Judge 평가까지를 2일 과정으로 다룹니다.

---

## 강의 정보

| 항목 | 내용 |
|---|---|
| **과정명** | (특별교육) [인공지능 : LLM 마스터] — **Part 4** (5개 파트 중 네 번째) |
| **주제** | 지식증강 — 벡터 RAG & 그래프 RAG & 온톨로지 RAG |
| **일정** | 2026-09-12 ~ 09-13 · **2일 / 총 12시간** (Day 1: 6시간, Day 2: 6시간) |
| **난이도** | 고급 |
| **강사** | **김의중** (아이덴티파이 대표) |
| **교재** | **딥러닝 개념과 활용** (김의중 저) |
| **형식** | 이론 + Jupyter Notebook 실습 병행 |

---

## 학습 목표

- RAG의 개념과 파이프라인(임베딩·시맨틱 서치·생성)을 이해하고 ChromaDB로 직접 구현할 수 있다.
- 주요 벡터 DB(Chroma·FAISS 등)의 특성을 비교하고 용도에 맞게 선택할 수 있다.
- LangChain으로 PDF 기반 RAG 챗봇(Streamlit) 애플리케이션을 구성할 수 있다.
- HyDE·Reranking·Ensemble Retriever 등 Advanced RAG 기법으로 검색 품질을 높일 수 있다.
- 지식 그래프(Neo4j·NetworkX) 기반 그래프 RAG를 구축할 수 있다.
- RDF·OWL 온톨로지와 추론(rdflib)으로 명시되지 않은 사실까지 도출하는 온톨로지 RAG를 이해한다.
- RAGAS와 LLM-as-a-Judge로 RAG 시스템의 품질을 정량 평가할 수 있다.

---

## 커리큘럼

### 📅 Day 1 (6시간) — 벡터 RAG의 기초와 구현

| # | 세션 | 노트북 |
|---|---|---|
| 00 | 실습 환경 점검 | [setup_check.ipynb](setup_check.ipynb) |
| 01 | 자연어 처리 기초 — 인코딩, 토큰화, 임베딩(Word2Vec) | [01_nlp_encoding_tokenization.ipynb](01_nlp_encoding_tokenization.ipynb) |
| 02 | RAG 파이프라인 — ChromaDB & 시맨틱 서치 | [02_rag_fundamentals.ipynb](02_rag_fundamentals.ipynb) |
| 03 | 벡터 DB 심층 비교 분석 및 실습 | [03_vector_db_comparison.ipynb](03_vector_db_comparison.ipynb) |
| 04 | LangChain RAG 어플리케이션 구현 (PDF 챗봇) | [04_rag_practice.ipynb](04_rag_practice.ipynb) |

### 📅 Day 2 (6시간) — Advanced · Graph · Ontology RAG와 평가

| # | 세션 | 노트북 |
|---|---|---|
| 05 | Advanced RAG — HyDE, Reranking, Ensemble Retriever | [05_advanced_rag_base.ipynb](05_advanced_rag_base.ipynb) |
| 06 | 그래프 RAG — 지식 그래프 기반 검색 증강 | [06_graph_rag.ipynb](06_graph_rag.ipynb) |
| 07 | 온톨로지 RAG — RDF · OWL 추론 (rdflib) | [07_ontology_rag.ipynb](07_ontology_rag.ipynb) |
| 08 | RAG 평가 — RAGAS & LLM-as-a-Judge | [08_rag_evaluation.ipynb](08_rag_evaluation.ipynb) |

---

## 실습 환경 설정

먼저 저장소를 복제합니다.

```bash
git clone https://github.com/choki0715/LLM_master_part4.git
cd LLM_master_part4
```

이후 깡통(clean) Ubuntu 상태에서 다음 스크립트로 실습 환경을 자동 구성합니다. `sudo` 없이 `uv`로 독립형 **Python 3.11**을 설치하고 가상환경을 생성합니다.

```bash
bash setup.sh
```

생성되는 가상환경:

| 환경 | 용도 |
|---|---|
| `venv` | 메인 환경 (torch 2.11 / langchain 0.3 / chromadb / sentence-transformers / neo4j / rdflib / ragas 등) — 모든 Part 4 노트북에서 사용 |
| `venv-quant` | 양자화 전용 (torch 2.2 / transformers 4.46 / auto-gptq / autoawq) — 선택 사항 |

> Python 3.11을 고정하는 이유: 최신 Python(3.14)에서는 gensim·auto-gptq·autoawq 등 다수 ML 패키지의 사전 빌드 휠이 없어 설치가 깨지기 때문입니다.

> 💡 대부분의 RAG 실습은 GPU 없이 CPU로 실행 가능합니다. 일부 세션은 외부 서비스가 필요합니다: 그래프 RAG(Session 06)는 **Neo4j**(Docker 또는 Neo4j Aura 무료 인스턴스), 생성·평가 단계는 **OpenAI API 키**(`OPENAI_API_KEY`)가 있으면 더 풍부하게 실습할 수 있습니다.

### 설치 확인

환경 구성 후 [setup_check.ipynb](setup_check.ipynb)를 실행해 주요 패키지와 GPU/CUDA 인식 여부를 점검하세요.

```bash
source venv/bin/activate
jupyter notebook   # 또는 VS Code / JupyterLab 사용
```

---

## 시리즈 구성

**LLM 마스터 과정 특별 교육 시리즈**는 총 5개 파트로 구성되며, 각 파트는 2일 / 12시간 과정입니다. (난이도: 고급)

| 파트 | 일정 | 주제 |
|---|---|---|
| Part 1 | 2026-08-08 ~ 08-09 | LLM 아키텍처 분석 및 HuggingFace, LangChain 활용 |
| Part 2 | 2026-08-15 ~ 08-16 | 모델 경량화와 추론 최적화 |
| Part 3 | 2026-08-22 ~ 08-23 | 강화학습을 통한 LLM 정렬 파인튜닝 |
| **Part 4** *(본 저장소)* | 2026-09-12 ~ 09-13 | 지식증강 — 벡터 RAG & 그래프 RAG & 온톨로지 RAG |
| Part 5 | 2026-09-19 ~ 09-20 | 바이브 코딩을 이용한 Agentic AI와 Harness 설계 |

---

## 라이선스 및 저작권

본 교육 자료의 모든 노트북과 코드는 **© AIDENTIFY. All rights reserved.**
교육 목적 외 무단 복제·배포를 금합니다.
