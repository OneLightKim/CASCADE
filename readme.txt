# CASCADE: Clinically Aware Strategy Controlled AI Dialogue Engine

**[HCLT 2025] 2025년도 한글 및 한국어 정보처리 & 한국코퍼스언어학회 공동 학술대회 게재 논문**

## Introduction (소개)

**CASCADE**는 정신 건강 상담을 위한 AI 기반 프레임워크로, 대규모 언어 모델(LLM)이 임상적 맥락을 반영하지 못하거나 환각(Hallucination) 현상을 일으키는 문제를 해결하기 위해 제안되었습니다.

본 연구는 다음과 같은 다단계 프로세스를 통해 상담의 안정성과 신뢰도를 향상시킵니다:

1. **Disease Classification (질환 분류):** 사용자의 게시물을 분석하여 주요 정신 질환(우울, 불안, 양극성 장애 등)을 분류합니다.

2. **Strategy Matching (전략 매칭):** 분류된 질환에 적합한 임상적 대응 전략을 매칭합니다.

3. **Evidence Retrieval (근거 검색):** 의료 지식 그래프(PrimeKG)와 PubMed 문헌 검색(RAG)을 결합하여 최신 의학적 근거를 확보합니다.

실험 결과, CASCADE는 **공감성(Empathy), 논리적 일관성(Logical Coherence), 지도력(Guidance)** 측면에서 기존 베이스라인 모델 대비 우수한 성능을 입증하였습니다.

---

## Framework Architecture

### 1. Overall Pipeline

CASCADE는 단일 턴(Single-turn) 프레임워크로 작동하며, `질환 분류(L) -> 전략 매칭(S) -> 근거 검색(K&R)` 과정을 거쳐 최종적으로 LLM이 임상적으로 타당한 응답을 생성합니다.

![Overall Framework](OVERALL CASACADE Framework.png)

*Figure 1: CASCADE 프레임워크의 전체 구조*

### 2. Triple & Evidence Ranking Pipeline

임상적 신뢰도를 확보하기 위해 PrimeKG에서 추출한 트리플(Triplet)과 PubMed 문헌을 재정렬(Re-ranking)하는 하이브리드 검색 파이프라인을 사용합니다.

![Ranking Pipeline](PIEPLINE OF Traiple Ranking and evidence based Pubmed documents Ranking.png)

*Figure 2: (A) 트리플 랭킹과 (B) PubMed 기반 근거 문헌 랭킹 파이프라인*

---

## Directory Structure

본 저장소의 디렉터리 구조는 다음과 같습니다.

```bash
CASCADE/
├── data/          # Dataset directory (Empty due to license, see below)
├── output_dir/    # Directory for model outputs and logs
├── src/           # Source code for the CASCADE framework
├── .gitignore
├── OVERALL CASACADE Framework.png
├── PIEPLINE OF Traiple Ranking and evidence based Pubmed documents Ranking.png
└── readme.txt
```

---

## Data Availability (PsyQA)

### 데이터 제공 관련 안내 (PsyQA)

본 저장소는 재현을 위한 코드와 디렉터리 구조만 포함하며, `data/` 폴더는 의도적으로 비워 두었습니다.
PsyQA 데이터셋은 해당 프로젝트의 배포/저작권 정책에 따르므로, 공식 저장소에서 허가를 받은 뒤 직접 다운로드해야 합니다.

- **공식 PsyQA 저장소:** https://github.com/thu-coai/PsyQA
- 본 저장소는 PsyQA 원본 데이터 및 파생 파일을 **재배포하지 않습니다.**

### Data Availability Notice

This repository includes code and directory structure only; the `data/` directory is intentionally left empty.
The PsyQA dataset is distributed under its own license/policy, so you must obtain permission from the official repository and download it yourself.

- **Official PsyQA repo:** https://github.com/thu-coai/PsyQA
- This repository **does not redistribute** the original PsyQA data or any derived files.

---

## Citation

이 연구가 유용하셨다면, 아래의 BibTeX 형식을 사용하여 인용해 주시기 바랍니다.

```bibtex
@inproceedings{kim2025cascade,
  title={CASCADE: 임상 인식 기반 전략 제어형 AI 상담 엔진 (CASCADE: Clinically Aware Strategy Controlled AI Dialogue Engine)},
  author={Kim, Kwang-Il and Kim, Seul-Gi and Kim, Hark-Soo},
  booktitle={Proceedings of the 2025 Joint Conference on Human and Cognitive Language Technology and Korean Association for Corpus Linguistics (HCLT-KACL 2025)},
  year={2025},
  address={Daejeon, South Korea}
}
```

---

## Contact

- **Affiliation:** Konkuk University, Department of Artificial Intelligence
- **Authors:**
  - Kwang-Il Kim (dksk1090@konkuk.ac.kr)