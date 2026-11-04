# 선행연구 위치 (1차, 2026-11-04) — 새로움 주장 범위

## 우리 문제는 어느 분야인가
판정이 확정될 때까지 비용(파형 읽기)을 최소로 쓰며 변수를 하나씩 열어 보는 문제다. 이는 **확률적 불리언 함수 평가(Stochastic Boolean Function Evaluation, SBFE)** 또는 운영과학에서 말하는 **순차 검사(sequential testing)**의 한 사례다.
- 변수 하나 = 파형 하나. 한 번 읽으면 9개 규칙의 라벨이 동시에 나오고, 이 라벨들은 서로 포함 관계(단조)에 있다.
- 판정하는 함수 = "어떤 규칙 하나에 대해, 모든 열의 절편 구간이 공통점을 가지는가"를 9개 규칙에 걸쳐 OR한 것이다. 각 열의 구간은 '처음으로 양성이 둘 연속되는 위치'로 정해진다.
- 표준 SBFE와 다른 점: 변수마다 확률 모형을 주지 않는다(라벨만 쓰는 조건). 또 함수가 기하 구조(기울기 −2 법칙, 구간 교차)를 가진다.

## 알려진 일반 방법
- **Deshpande·Hellerstein·Kletenik (SODA 2014):** SBFE를 확률적 부분모듈 집합 덮개로 바꾸고 적응형 탐욕법을 쓴다. 다만 선형 문턱식이나 read-once DNF 같은 함수에서는 이 방식으로 좋은 근사비를 얻을 수 없다는 한계도 보였다. https://ar5iv.arxiv.org/html/1310.3673
- **Allen·Hellerstein·Kletenik·Ünlüyurt (Algorithmica 2017):** 단조 k-DNF 평가에서 비용을 *기대 인증서 비용* 기준으로 묶는 근사 결과. https://www.sci.brooklyn.cuny.edu/~kletenik/publications/dnfeval.pdf
- **대칭 불리언 함수 SBFE(Gkenosis 외)**, **적응성 격차(Hellerstein·Kletenik·Liu·Witter)** 등 후속 연구가 있다. https://ar5iv.labs.arxiv.org/html/2111.08793 · https://par.nsf.gov/servlets/purl/10435780
- 11/03 공개 정책(seed)은 "한 번 더 읽으면 구간이 얼마나 좁아지나"를 보는 한 단계 탐욕 예측이다. 즉 위 적응형 탐욕 계열에 해당한다.

## 이번 방법(t2 / distilled_a.py)의 위치
- 원리: 규칙마다 **인증서의 모양을 먼저 정하고(commit), 그 인증서를 검증하는 데 필요한 변수만 연다.** 인증서는 '분리되는 두 열'이고, 검증은 높은 열의 앞부분 0 쓸기 + 낮은 열의 양성 쌍 하나다. 뒤집히는 위치는 이분 탐색으로 찾는다. AND/OR 구조에서 증명 하나를 정해 놓고 파고드는 **증명수 탐색(proof-number search, Allis 1994)**과 성격이 닮았다. "인증서 비용"을 직접 줄이려 한다는 점에서는 위의 인증서 비용 관점과도 통한다.
- search 분할 기준: seed 9,158 → t2 3,321 → 원리만 남긴 버전 3,453. 기존 방법을 빌려 쓰는 부분은 마지막 대체용으로만 남겼다.
- **주장할 수 있는 범위:** "이 기하 판정 문제에 맞춘 인증서 지향 순차 검사 정책이, 공개된 한 단계 탐욕 정책보다 읽기 횟수를 크게 줄인다(처음 보는 폴더에서 −60%)." 새로운 *일반* SBFE 알고리즘이라고 주장하지 않는다.
- **새로움을 더 주장하려면 필요한 것:**
  - (1) 일반 SBFE 기준선(적응형 탐욕, 증명수 탐색)을 직접 구현해 비교한다.
  - (2) 연속 문턱 버전(law_continuous_v1, 현재 기록 1,216)에서도 같은 원리가 통하는지 본다.
  - (3) 이분 탐색 때문에 줄어드는 양을 간단한 상한으로 설명한다.
  - (4) 문헌을 더 넓게 찾는다. 이번에는 검색 1회만 했다.

## 2차 조사 (2026-11-04 밤, 작업자 1명 웹검색; [기억]은 검색으로 확인 안 된 항목)
- **가장 가까운 연구 3개**
  1. 결정영역 판정(Decision Region Determination, DRD). Javdani et al. https://arxiv.org/abs/1402.5886 · DIRECT(Chen et al., AAAI 2015) https://ojs.aaai.org/index.php/AAAI/article/view/9694 · Adaptive Submodular Ranking https://arxiv.org/pdf/1606.01530 . 우리 인증서("어떤 규칙의 구간이 겹치는가")는 가설 공간 위 영역들의 합집합으로 볼 수 있다. 규칙이 중첩돼 있으면 영역이 크게 겹치고, 바로 이 경우에 DRD 보장이 약해진다. 우리 주장이 들어갈 틈이 여기다.
  2. SBFE: 선형 문턱식 3-근사와 CDNF O(log kd), Adaptive Dual Greedy(Deshpande–Hellerstein–Kletenik). https://arxiv.org/abs/1303.0726 · k-DNF 평가는 k=2에서도 NP-hard(Allen et al.) https://www.sci.brooklyn.cuny.edu/~kletenik/publications/soda.pdf
  3. 적응 부분모듈성 탐욕법(Golovin–Krause) https://arxiv.org/abs/1003.3967 · EC2(Golovin–Krause–Ray) https://arxiv.org/abs/1010.3091 · 상호작용 부분모듈 덮개(Guillory–Bilmes) https://arxiv.org/abs/1002.3345 . 우리 seed(한 단계 VOI)는 이 계열의 약한 임시 버전이다.
- 이분 탐색·2배씩 넓히는 탐색 부분은 새롭지 않다. 잡음이 없는 탐색이며, 잡음 이진탐색(Ben-Or–Hassidim https://arxiv.org/abs/quant-ph/0703231)보다도 쉬운 경우다.
- LLM 프로그램 진화: FunSearch [기억], AlphaEvolve, OpenEvolve, CodeEvolve https://arxiv.org/pdf/2510.14150v4 , ShinkaEvolve·ReEvo·EoH [기억]. 모두 최적화 휴리스틱(빈 패킹, TSP, 구성 문제)이 대상이다. **인증 문제의 적응 질의 정책을 진화시킨 연구는 찾지 못했다.** 다만 이번 검색은 얕았으므로 "없다"가 아니라 "못 찾음"이다.
- **정직하게 주장할 수 있는 범위:** 새 알고리즘 부류나 보장을 주장하지 않는다. "중첩 문턱 라벨을 한꺼번에 공개하는 정확 인증 문제에서는 일반 부분모듈·VOI 탐욕법이 약하다(영역이 크게 겹치기 때문). LLM으로 진화시킨 구조적 정책(규칙마다 분리 증인을 약속하고 뒤집힘 위치를 이분 탐색)이 이 벤치마크에서 한 단계 VOI 대비 읽기를 60% 줄인다." 이 문제군에 국한된 경험적 결과이고, 해석 가능한 발견된 정책이다. **하한, 또는 증인 구조가 최적에 가깝다는 증명이 붙으면 진짜 결과가 된다.** → OPTIMALITY_PLAN.md 1절.
- **꼭 구현할 기준선:**
  1. DRD/EC2 계열 적응 부분모듈 탐욕(DIRECT의 edge cutting, 합성 가설 사전분포 사용)
  2. SBFE 평가기(Adaptive Dual Greedy 또는 비용/확률 순서 DNF·문턱 평가기). 작은 C에서는 증명수/반복심화 탐색을 최적 근사 기준으로 둔다.
  - 함께 보고할 것: 증인 약속을 뺀 진화 정책, 열마다 순수 이분 탐색.
