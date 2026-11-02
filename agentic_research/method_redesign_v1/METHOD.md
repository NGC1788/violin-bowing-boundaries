# 후보 B: 제공자 삭제에 강건한 기억 기반 확률 예측 학습

이름은 작업 설명이며 새로운 원리가 검증됐다는 명칭이 아니다. 사용처는 grouped small-data prediction 전반이며 바이올린은 한 적용 자료다. 상태는 **JUSTIFIED_TO_PREPARE**, 실제 학습·성능은 아직0건이다.

## 1. 기존 실패에서 바꾸는 것

Gaussian innovation scale과 미래 mask를 예측한 뒤 표본분산 확률을 계산하던 head를 재시도하지 않는다. 새 가설은 **어떤 과거 사례에 의존하도록 학습하는가**를 바꾼다. 미래 프레임의 조건부 iid/등분산 Gaussian을 가정하지 않는다. 학습 자료의 관측된 공동 사건 pattern에 확률을 배분한다. 이는 범주 분포의 weighted retrieval이고 그 자체는 기존 방법이다.

한 제공자만의 사례가 예측을 지배하면 제공자 교체에 취약할 수 있다는 **가설**이 출발점이다. 현재 실패에서 이 원인이 실증된 것은 아니다. 정확 head의 결함이 이 가설을 증명해 주지 않는다.

## 2. 기억과 예측

innertrain의 기록 j에 prefix x_j, 제공자 g_j, 허용된 미래 사건 pattern z_j∈{1,…,10}이 있다. 같은 fold의 innertrain 문턱으로 pattern을 만든다. Invalid 기록도 all-false pattern으로 남는다. 각 pattern의 9차원 사건 벡터 v(z)는 이미 사용한 고정 규칙과 같다.

Train query i의 기억은 **자기 자신뿐 아니라 같은 제공자의 모든 기록을 제외**한다:

\[
\mathcal M_i=\{j\in\mathrm{innertrain}:g_j\ne g_i\}.
\]

Query와 key의 prefix encoder만으로 점수 s_θ(x_i,x_j)를 계산한다. 양의 원자 가중치 b_ij를 정규화해 a_ij를 얻는다. 후보 구현은 bounded cosine score와 제공자별 record-count 보정을 사용할 수 있지만 모든 대조에서 동일해야 하며 구현 계약 전에 이유와 값을 고정한다. 제공자 식별자를 예측 입력 feature로 넣지 않는다.

\[
 P_i(k)=\sum_{j\in\mathcal M_i}a_{ij}\,1\{z_j=k\},\qquad
 q_{i,r}=\sum_kP_i(k)v_r(k).
\]

추론에서는 해당 fold innertrain의 기억만 사용한다. Held 자기 미래/다른 held 기록/held 제공자 평균을 검색에 넣지 않는다. 저장된 source 사례의 label을 쓴다는 점은 parametric head와 다른 배포 자원이다. 같은 학습 정보라는 말이 같은 inference memory 비용이라는 뜻은 아니다.

이 분포는 현재9query에 대한 **10pattern categorical law**다. 연속적인 모든 SD 문턱에 유효한 CDF를 배웠다거나 latent 실제 음질을 예측했다고 부르지 않는다.

## 3. 학습 중 제공자 삭제

허용된 donor group g를 하나 선택해 그 사람의 기억 원자를 모두 지운 뒤 남은 동일 점수를 재정규화한다. null intervention(아무도 지우지 않음)도 포함한다. 같은 donor를 9query에서 함께 제거한다.

\[
 W_{ig}=\sum_{j:g_j=g}a_{ij},\quad
 A_{ig,r}=\sum_{j:g_j=g}a_{ij}v_r(z_j),\quad
 q_{i,-g,r}=\frac{q_{i,r}-A_{ig,r}}{1-W_{ig}}.
\]

이 차감식은 검산 기준이다. 실제 구현은 남은 key의 stable logsumexp/positive sum으로 재정규화하여 W≈1의 상쇄 오차를 피한다. 남은 donor가 없으면 silent fallback하지 않고 명시적으로 실패한다. 학습 전 허용 삭제 집합이 비어 있지 않은지 합성 경계 시험으로 검증한다.

목표는 query 사람을 동일 가중하는 평균에서, query마다 같은 donor 하나를 지웠을 때의 **평균9query proper loss**를 최대로 하는 경우를 학습하는 것이다:

\[
 \min_\theta\ \frac1{|\mathcal G|}\sum_{h\in\mathcal G}
 \frac1{n_h}\sum_{i:g_i=h}
 \max_{g\in\mathcal D_i\cup\{\varnothing\}}
 \ell(\widetilde q_{i,-g},y_i)
 +\lambda L_{\mathrm{common\ aux}},
\]

\[
 \ell(q,y)=-\frac19\sum_{r=1}^{9}
 \{y_r\log q_r+(1-y_r)\log(1-q_r)\}.
\]

최대 donor가 동률이면 결정적인 순서로 하나를 골라 subgradient를 쓴다. Query마다 다른 donor를 지울 수 있지만 query 안의 각 사건에 다른 donor를 골라 붙이지 않는다. 후자는 실제 하나의 memory distribution이 아닌 더 강한 별도 adversary다.

Null을 포함하므로 고정 θ의 학습 objective는 full-memory BCE 이상이다. **이것이 unseen 사람의 손실 상한, PAC 보장, 더 낮은 실제 성능을 증명하지는 않는다.** 단순 TV bound와 deletion 식도 새 정리로 주장하지 않는다. 예컨대 P_i=(1−W)P_i,-g+WP_i,g이므로 TV(P_i,P_i,-g)≤W이다. 최악 손실 최소화가 반드시 W를 줄인다고 단정할 수 없다.

## 4. 표현 불가능성과 dead-gradient 방지

어떤 사건의 모든 positive가 한 donor에게만 있으면 그 사람 삭제 후 원자 가중치로는 양성을 만들 수 없다. 이를 숨기지 않는다. 모든 innertrain query/허용 삭제의 raw positive/negative support 결손을 집계한다. Held label support를 보고 smoothing을 고르지 않는다.

다음 준비의 **사전 제안**은 모든 retrieval 대조에 동일한 고정 uniform10pattern prior를 ρ=0.01 섞는 것이다:

\[
 \widetilde P=(1-\rho)P+\rho\,U_{10},\qquad
 \widetilde q=\widetilde P V.
\]

각 query에는 참·거짓 pattern이 적어도 하나씩 있으므로 \(0.001\le\widetilde q_r\le0.999\)다. 10⁻⁶ score clip으로 인한 구조적 dead region을 피할 수 있다. **Prior만 positive를 복구한 경우 가중치 학습으로 그 사건을 맞힐 수 있다는 뜻은 아니다.** 원자 support의 결함·학습 기울기와 prior 기여를 따로 보고한다. 이 smoothing은 고전적 장치이며 novelty가 아니다. ρ는 실제 자료를 열기 전에 코드 검토·계약에서 고정하고 성능에 따라 바꾸지 않는다.

## 5. 공정한 12job 비교 제안 — 아직 동결 아님

같은 retrieval encoder/초기 가중치/기억/입력/공통 aux/optimizer·epoch·minibatch의 네 학습 목적×기존3personfold=12job을 다음 batch의 후보 메뉴로 둔다.

| 방법 | 달라지는 학습 연산 | 답할 질문 |
|---|---|---|
| retrieval_erm | full-memory loss | 단순 학습 검색으로 충분한가? |
| retrieval_random_delete | donor 하나를 무작위 삭제 | 그냥 dropout 정규화인가? |
| retrieval_adversarial_delete | 위의 query별 최악 donor 삭제 | 의도한 donor intervention이 추가 이득을 주는가? |
| retrieval_query_group_dro | 기억 삭제 없이 query 제공자별 큰 평균 loss를 강조 | donor-side intervention이 query-side robust weighting과 다른가? |

Query-group-DRO의 정확 update/weights, random-delete/null 확률, common aux 등은 코드 준비와 계약에서 먼저 고정해야 한다. 값 탐색을 하지 않는다. 모든 retrieval arm은 null 포함 전체 deletion risk를 계산하는 동일 pass를 사용해 random/ERM arm에도 matched forward resource를 부여하고 쓴 wall time을 기록한다. Backward는 목적식에 따라 다르므로 완전히 동일 FLOP이라고 주장하지 않는다. 실제 비용이180초/job를 넘을 것 같으면 합성 기능시험 시간으로 낙관 추정하지 말고 실행 계약의 feasibility 문제로 남긴다.

기존 source Markov/HGB 및 **joint_aux/mdn_aux**의 봉인 예측은 참조로 재사용한다(새 reference fit0). 기존 신경망의73derived 입력·innertrain 전처리·동일 공통 경로/mask 감독·초기 공유 parameter가 일치하도록 새 encoder/aux를 명세한다. 원본 코드는 수정하지 않는다. 과거 직접 head와 memory model의 파라미터/학습 목적/배포 memory 자원 차이를 공개한다. 이 참조는 새 실험에서 재학습한 matched head가 아니다. 특별한 연산의 이득 판단은 동일한 새 retrieval 대조 간 비교를 중심으로 한다.

만약 이 재사용 조건을 코드상 보장하지 못하면 숨겨진 fit을 추가하지 않는다. 다음 준비에서 그 문제를 blocker로 남기고 메뉴/계약을 학습 전에 별도로 검토한다. 실제 학습 시작 뒤 같은 batch를 구조하지 않는다.

## 6. 성공·기각 조건

주 평가는 **full innertrain memory로 held future를 예측한 사람 macro9query log loss**다. 삭제 시험만 좋아지면 주 성능 성공이 아니다. 기존 고정 기준의 Markov/HGB 대비2%·최소2fold 조건을 완화하지 않는다. 새 연산의 근거에는 retrieval_erm과random_delete보다 모두 낮은 전체 주 loss 및 적어도2fold의 개선도 필요하다. Query-group-DRO보다 이득이 없으면 donor-side의 추가 필요 근거가 약하다.

Deletion-risk는 보조 지표다. Held full-memory와 각 donor 삭제의 모든 반사실 예측도 정답 점수 계산 전에 봉인한다. Held 정답으로 최악 donor를 고르는 것은 봉인 뒤 보조 risk 집계에만 허용하며 실제 발행 예측/모델 선택에는 사용하지 않는다. 새 네 retrieval arm에 같은 donor-removal protocol을 적용한다. 재사용한 parametric joint/MDN/HGB에는 삭제 가능한 inference memory가 없으므로 동등한 deletion risk를 계산했다고 부르지 않는다. 최대사람·전체fold·Brier·사건별 positive빈도/손실·invalid와valid all-zero를 구별한 집계·memory group mass/entropy/effective donor수·support결손·cost를 보존한다. 높은 entropy만 얻고 예측이 무정보로 수렴하는 실패를 버리지 않는다.

Random 삭제나 단순 robust query weighting으로 이득이 설명되거나 full-memory 정확도가 나빠지면 핵심 주장을 축소/기각한다. 효과가 나온 뒤에야 고엔트로피/평탄 가중치의 역할을 별도 검토할 수 있다. 높은 novelty·성능은 현재 미입증이고 source34명은 이미 노출된 탐색이다.
