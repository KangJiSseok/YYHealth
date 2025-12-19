## 📌 문제 정의 (Problem Definition)
### 1. 문제 배경

본 프로젝트에서는 학술 PDF 문서를 pdfplumber를 이용해 텍스트로 추출한 뒤,
레이아웃 블록 단위로 분해하고 토큰 기준 청크로 분할하여 임베딩한 후
벡터 DB(Qdrant 등)에 저장하여 RAG 기반 질의응답을 수행한다.

이 과정에서 extract_text(layout=True) 옵션을 사용하여
PDF의 레이아웃을 최대한 보존한 텍스트 추출을 수행하고 있다.

### 2. 문제 현상

PDF에서 추출된 텍스트를 청크 단위로 확인한 결과,
문장이 비정상적으로 끊어진 상태로 저장되는 현상이 반복적으로 발생한다.

주요 현상

줄바꿈(line break)으로 인해 단어가 중간에서 분리됨

컬럼 폭 또는 페이지 레이아웃으로 인해 단어가 잘림

의미적으로 하나의 문장 또는 문단이 여러 개의 조각으로 분리됨

예시
```aiignore
medic
therapy play a pivotal role in the treatment of diabetes

fatty
acid supplements is not recommended

Medical    nutrition    therapy
```


위와 같이,

하나의 단어가 줄바꿈을 기준으로 분리되거나

불필요한 공백이 다수 삽입된 상태로

하나의 청크에 저장되는 문제가 발생한다.

### 3. 문제의 영향

이 문제는 단순한 텍스트 가독성 문제를 넘어,
RAG 시스템 전반의 품질 저하로 직결된다.

### 🔴 주요 영향

임베딩 품질 저하
하나의 벡터가 불완전한 의미 단위만 포함
문맥 정보 손실로 의미 유사도 계산 정확도 하락
검색 정확도 및 신뢰도 하락
질문과 정확히 일치하는 문단이 검색되지 않음
의미적으로 어색한 청크가 상위 결과로 노출
LLM 응답 품질 저하
불완전한 문장을 기반으로 응답 생성
답변의 일관성 저하 및 환각(hallucination) 가능성 증가
청크 단위 의미 붕괴
청크가 더 이상 “완결된 의미 단위”가 아님
청크 사이즈 조정(min/max tokens)으로 해결 불가


### 4. 근본 원인 분석 (Root Cause)
PDF는 본질적으로 문단 기반 포맷이 아닌 좌표(x, y) 기반 포맷
extract_text(layout=True)는
줄 단위 텍스트를 그대로 반환
줄 끝에서 발생하는 단어 절단(word split)을 자동으로 복원하지 않음
텍스트 추출 이후 단계에서
줄바꿈·단어 절단을 복원하는 전처리 과정이 존재하지 않음
👉 즉, 문제의 본질은 청크 사이즈 설정 문제가 아니라
PDF 텍스트 정규화 전처리 누락 문제이다.


## 🛠 해결 방안 (Solution)
### 1. 해결 전략 개요

청킹 이전 단계에서 PDF 텍스트를 정규화(normalization)하여
줄바꿈(line break) 및 단어 절단(word split)을 복원한 후
의미 단위 청킹을 수행한다.

본 해결 방안은 청크 사이즈를 조정하는 방식이 아니라,
PDF 텍스트 추출 결과 자체를 사람이 읽는 문장 단위로 복원하는 것에 초점을 둔다.

개선된 텍스트 처리 파이프라인은 다음과 같다.

[기존]
```
PDF → extract_text → chunk → embedding
```

[개선]
```
PDF → extract_text
    → text normalization (line break / word split 복원)
    → paragraph split
    → semantic chunking
    → embedding
```

### 2. 해결 방안 ① 줄바꿈·단어 절단 복원 전처리 도입 (권장 ⭐⭐⭐⭐⭐)
🔹 접근 방식

PDF 텍스트 추출 직후, 청킹 이전 단계에서
다음과 같은 정규화 로직을 적용한다.

```def normalize_text(text: str) -> str:
    """
    Clean PDF-extracted text before paragraph splitting:
    - join hyphenated line breaks for Latin/digit words
    - replace single line breaks with spaces (keep paragraph breaks)
    - collapse excessive spaces
    """
    if not text:
        return ""
    cleaned = re.sub(r"([A-Za-z0-9])-\s*\n([A-Za-z0-9])", r"\1\2", text)
    cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = cleaned.replace(" \n", "\n").strip()
    return cleaned
```

🔹 처리 내용 설명

하이픈(-)으로 잘린 단어 복원

PDF 줄 끝에서 발생한 단어 분리를 결합

예:

nutri-
tion  →  nutrition


단일 줄바꿈 제거

문단 구분(\n\n)은 유지

문장 내부 줄바꿈만 공백으로 치환

예:

fatty
acid  →  fatty acid


불필요한 다중 공백 정리

PDF 레이아웃 특성으로 발생한 비정상 공백 제거

예:

Medical    nutrition    therapy
→ Medical nutrition therapy

🔹 적용 위치 (중요)

해당 정규화 로직은 반드시 청킹 이전,
문단 분리(paragraph split) 이전 단계에 적용된다.

page_text = subpage.extract_text(layout=True) or ""
page_text = normalize_text(page_text)
paragraphs = split_paragraphs(page_text)


👉 이 위치에 적용하지 않으면
깨진 문장이 그대로 청킹·임베딩되어 효과가 없다.

3. 해결 방안 ② 2단 컬럼 구조 인식과 병행 처리

텍스트 정규화는 컬럼 혼합 문제와 함께 사용될 때 효과가 극대화된다.

🔹 병행 적용 로직

_detect_two_columns(page)를 통해 2단 컬럼 여부 판단

좌측 / 우측 컬럼을 분리하여 텍스트 추출

각 컬럼에 대해 동일한 normalize_text() 적용

이를 통해 다음을 동시에 해결한다.

좌·우 컬럼 문장 병합 문제

컬럼 내부 줄바꿈·단어 절단 문제

4. 기대 효과
✅ 텍스트 품질 개선

문장이 원래의 의미 단위로 복원됨

단어 단절 및 비정상 공백 제거

✅ 청크 품질 안정화

하나의 청크가 완결된 문맥을 포함

기존 청크 사이즈(120~220 tokens) 유지 가능

✅ 임베딩 및 RAG 성능 향상

의미 밀도가 높은 벡터 생성

검색 정확도(recall / precision) 개선

LLM 응답의 일관성 및 신뢰도 향상

5. 청크 사이즈 조정 불필요

본 해결 방안은 청크 크기 문제를 전제로 하지 않는다.

기존 설정:

min 120 ~ max 220 tokens

문제 원인은 텍스트 정규화 누락이므로

청크 사이즈는 유지하고 전처리 품질만 개선

✅ 결론

본 해결 방안은 PDF 텍스트 처리 과정에서 발생하는
줄바꿈 및 단어 절단 문제를 구조적으로 해결하기 위해
청킹 이전 단계에 텍스트 정규화(normalization)를 도입하는 접근이다.

이를 통해 청크 단위의 의미 연속성을 회복하고,
RAG 시스템 전반의 임베딩 품질과 응답 정확도를 효과적으로 개선할 수 있다.