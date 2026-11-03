# 보조 감독으로 선택하는 함수보존 conditioning — 서면 후보 B

상태: **JUSTIFIED_TO_PREPARE_EXPLORATORY / 실제 학습 미승인**. 물리 모델이 아닌 AI 학습 경로에 관한 가설이다. 현재 파일은 실행 계약이나 성능 결과가 아니다.

## 질문과 기존 실패로부터의 거리

이전 donor 삭제 학습은 기억 의존도와 확률 민감도를 줄였지만 양성 사건 손실 증가로 ERM보다 나빴다. 그 결과만으로 gradient 충돌이 원인이었다고 확정할 수는 없다. 여기서는 retrieval/삭제/scale 하한을 조정하지 않는다. 별도 parametric predictor에서 **보조 감독이 현재 예측을 직접 바꾸지 않고 다음 주 학습의 parameterization을 바꿀 때 유익한가**를 검토한다.

## 구조와 보조 단계의 범위

기존과 같은 73차원 prefix만 입력한다. 같은 32차원 encoder 출력에 identity 초기화한 선형 adapter를 넣는다.

\[
u=f_\phi(x)\in\mathbb R^{32},\quad h=Wu,\quad
z=A W u+b\in\mathbb R^{10},\quad p=\operatorname{softmax}(z),\quad q=p^\top\mathcal B.
\]

\(\mathcal B\in\{0,1\}^{10\times9}\)는 기존 열 순서 그대로의 attainable-pattern bank다. **W 뒤에 비선형 함수를 넣지 않는다.** W는 bias 없는32×32, A는10×32이다. 공통 미래 mask/path auxiliary generator \(G_\psi\)는 **u에서 분기한다**. fine-label head \(B\)는 \(Wu\)에서 분기한다.

각 minibatch에서 먼저 기존 사건 BCE+같은 공통 auxiliary 목적의 주 단계를 수행한다. 이 단계에서는 \(\phi,W,A,b,\psi\)를 갱신할 수 있다. 이어지는 추가 fine-auxiliary 단계는 \(\phi,A,b,\psi\)와 running state를 모두 고정하고 **W와 B만** 갱신한다. 공통 generator를 Wu에서 분기시키면 아래의 coarse 보존으로 그 generator까지 보호할 수 없으므로 금지한다.

## 유한 step의 예측 보존

\[
H=I_{10}-\frac{\mathbf1\mathbf1^\top}{10},\qquad A_c=HA,\qquad
P=I_{32}-A_c^{\dagger}A_c.
\]

\(A_cP=0\)인 직교 projector를 만든다. optimizer가 제안한 **실제 유한 변화량** \(D\)에 대해

\[
W^+=W+PD
\]

를 적용한다. gradient만 투영한 뒤 Adam preconditioning이나 weight decay를 별도로 적용하면 같은 보장이 아니다. 투영 후 W를 다시 정규화·직교화하는 연산도 허용하지 않는다.

모든 입력 x에 대해

\[
H(z^+-z)=A_cPDu=0.
\]

따라서 logits 변화는 공통 상수이고 \(p^+=p,\ q^+=q\)다. 이는 **해당 보조 단계 한 번**에 관한 실수 대수의 정확한 결과다. \(\operatorname{rank}(A_c)\le9\)이므로 null dimension은 적어도23이다. pseudoinverse cutoff, float64 오차, 적용 전후 residual은 이후 기능시험으로 별도 확인해야 한다. 유한 정밀도 검사를 모든 실수 입력에 대한 실험적 증명이라고 부르지 않는다.

Encoder나 readout을 동시에 바꾸거나 W 뒤에 비선형 함수를 두면 이 증명은 성립하지 않는다. 전체 training loss 감소, 다음 주 step의 출력 보존, 일반화 개선은 보장하지 않는다. 현재 fine auxiliary 목적 자체의 감소도 norm matching/유한 step에서 보장되는 것은 아니다.

## 표현 용량에 관한 중요한 한계

coarse 함수는 \((AW)u\), 선형 fine 함수는 \((BW)u\)로 쓸 수 있다. W는 새로운 입력 정보나 함수 용량을 추가하지 않는다. 보조 감독이 새 정보를 만들어 hidden feature에 저장한다는 설명은 부정확하다. 후보는 **함수는 같은 상태에서 이후 최적화의 conditioning을 선택하는 재매개화**다.

고정 u, 동일 learning rate의 단순 SGD에서 \(C=AW\), \(G_C=\partial L/\partial C\)라 두면 두 factor의 주 update는 1차로

\[
\Delta C=-\eta\left(G_CW^\top W+AA^\top G_C\right)+O(\eta^2).
\]

그러므로 현재 coarse 확률(또는 HC)을 보존한 W 이동도 다음 step의 metric을 바꿀 수 있다. C=AW 자체는 공통-logit row만큼 변할 수 있다. 이 식은 SGD의 해석이며 AdamW의 동일한 update 공식을 뜻하지 않는다. encoder의 즉시 사건 손실 gradient는 같은 HC이면 같을 수 있다. 효과가 있어도 그 원인이 coarse representation 용량 증가라는 결론은 낼 수 없다. 고전 factorization/투영 성질이며 새 정리라고 주장하지 않는다.

## fine label: 기존 coarse label의 안쪽만 구분

각 inner-train의 guard-valid 미래 연속 변동량 두 개에서 octile 경계7개를 구한다. 기존 quartile 경계는 octile 경계의 2·4·6번째에 **원래 값을 그대로 넣어** 동일성을 보장한다. quantile 보간 규칙과 `<=` 경계 규칙을 고정한다. held 자료로 경계·표준화·prior를 적합하지 않는다.

축별 fine bin을 \(r_d=\sum_{j=1}^7 1\{y_d>t_{dj}\}\in\{0,\ldots,7\}\)로 정의한다. 유효 row의 fine ID는 \(8r_1+r_2\), invalid ID는64다. coarse bin은 \(\lfloor r_d/2\rfloor\)이며 기존9사건 label과 정확히 대응해야 한다. repeated octile/empty class는 제거하거나 재분할하지 않는다. invalid15개도 남긴다. 이 변환은 label 준비이며 inference 입력이 아니다.

64개 valid fine class와 invalid를 합쳐65class다. 9개 nonzero coarse pattern은 각각4개 fine class에, all-false는28개 valid fine class+invalid에 대응한다. 실제 class 빈도는 아직 열람하지 않았다.

\(s=B W u+d\)와 미리 정한 coarse fibre \(F_c\)에 대해 보조 손실은

\[
L_{\rm fine}(i)=\log\sum_{k\in F_{c_i}}\exp s_{ik}-s_{i,k_i}.
\]

정답 coarse는 학습 손실의 fibre 선택에만 쓴다. 추론 head는 fine/coarse 정답이나 future mask/guard/center를 인자로 받지 않는다. fine head는 배포 시 버린다. 같은 추가 감독을 네 arm 모두에 주며, fine label 자체가 필요한지에 관한 별도 주장은 하지 않는다.

## 네 arm의 최소 반증 비교 — 아직 실행 계약 아님

모든 arm은 같은 encoder/W/A/G/B 구조·초기값·prefix·train label·minibatch·주 목적과 스케줄을 사용한다. 추가 단계에서 B는 모두 학습하고, W에 적용하는 연산만 다르다.

| arm | 추가 W step | 분리하려는 설명 |
|---|---|---|
| main_only | 0; fine B만 학습 | 추가 conditioning이 실제로 필요한가 |
| ordinary_matched | 일반 제안 D를 아래 target norm에 맞춤 | 단순 auxiliary update 효과인가 |
| random_matched | 같은 rank의 label-independent random projector R의 RD를 같은 target norm에 맞춤 | 임의 subspace/작은 step 효과인가 |
| null_conditioned | PD | 출력 관련 방향을 피하도록 선택한 update가 유익한가 |

norm target은 **각 arm 자신의 현재 A와 D로 계산한** \(\|PD\|_F\)다. ordinary/random을 이 값에 맞춘다. 학습 경로가 갈라진 뒤 arm 사이의 norm이 실제로 같다는 주장은 하지 않는다. target=0이면 모두0, random projected norm=0이면 해당 step0으로 기록하는 등의 영벡터 처리를 계약에서 사전 고정한다. random projector는 학습/held label과 독립이며 rank와 RNG 규칙을 고정한다. 실제 gradient norm·적용 norm·rank·유한성·W conditioning을 저장한다.

현재 예산은 네 arm×기존 source3fold=12job 한 batch다. 다섯 번째 실제 arm/seed/선택 반복은 추가하지 않는다. 동일 label·구조의 공정 비교는 가능하지만 fine-label 필요성·다른 optimizer·일반 OGD/HNC 대비 우월성까지 모두 식별할 수는 없다. 기존 joint_aux/MDN/HGB/Markov sealed 예측은 reference로만 재사용하고 representation·parameter·계산 차이를 공개한다.

## 성공/실패를 판단할 기준

주 점수는 기존 사람 macro9query log loss다. 기존 Markov/HGB 대비2%/최소2fold screen을 완화하지 않는다. null arm은 main_only/ordinary_matched/random_matched 각각보다 전체와 최소2fold에서도 좋아야 제안 연산의 추가 이득 후보로 인정한다. 보조 단계 보존만 통과하거나 auxiliary loss만 낮아진 경우는 예측 성공이 아니다. 기존 exposed source의 반복 설계이므로 이 screen 통과도 독립 확증이나 높은 novelty를 증명하지 않는다.

최악 사람, fold별 악화, Brier/사건별·guard·support 집계, all-zero 붕괴, 단계별 drift, norm/rank/conditioning, 비용과 실패를 보존한다. 유한성/불변성 위반, 첫 timeout/실패, 원래 점수 개선 미달이면 고정 batch를 STOP한다. projector cutoff·rank·epoch·seed·보조 가중치·label bin 수를 결과에 맞춰 바꿔 구조하지 않는다.
