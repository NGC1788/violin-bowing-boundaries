# Projection audit v1 — 사전 수학·수치 검토

2026-10-30. 대상은 기존 미래 예측 모델의 정확도 재평가가 아니라, **알려진 forward Bernoulli-KL projection의 수치적 적절성**이다. 이 검토에서 실제 관측 target, 행별 손실, 그룹별 결과나 예측 배열을 읽지 않았다. `protocol.json`, `solver.py`, 합성 테스트를 검토했다. 이 문서는 실행 전 검토다.

## 사전 판단

**현재 수학적 설계는 타당하다.** 실제 내부점 \(p=wV\)에서 계산한 gradient와 full-simplex Frank–Wolfe gap을 사용하면, mass floor를 둔 solver가 낸 후보라도 원래 convex 문제의 최적화 오차를 상한으로 검사할 수 있다.

다음 구현 기록 조건을 동결 전에 충족해야 한다.

1. raw solver weights의 제약 오차와 `max(weight, floor)`·재정규화로 바뀐 최대 크기를 기록한다. 최종 후보에서 objective·gradient·gap을 다시 계산하는 현재 방식은 맞다.
2. 한 시작점이 수치 예외를 내도 다른 고정 시작점을 실행하고, 실패 이유를 보존한다. 두 시작점 모두 실패한 행도 분모 256에서 삭제하지 않는다.
3. 실행기는 읽는 NPZ member를 `ids`, `probabilities`, `repaired_probabilities`로 제한하고, 지정한 ID 64개를 모든 모델에서 동일하게 사용해야 한다.
4. 프로토콜·solver·실행기·합성 테스트·입력 예측 파일의 출처를 먼저 동결한다. 실제 예측에서 실패가 많더라도 이번 실행의 반복 수·허용 gap·표본을 바꾸지 않는다.

위 사항은 성공 결과를 보장하지 않는다. 검사가 실패하거나 엄격한 점수 검사가 보정을 거부하는 것도 유효한 감사 결과다.

## 목적함수와 gradient

패턴 행렬 \(V\in\{0,1\}^{10\times9}\)의 행은 10개 가능한 observer 출력이다. \(w\in\Delta_{10}\), \(p=wV\), \(q_\epsilon=\operatorname{clip}(q,10^{-6},1-10^{-6})\)로 정의한다.

\[
f(w)=\frac19\sum_{j=1}^{9}\left[p_j\log\frac{p_j}{q_{\epsilon,j}}+(1-p_j)\log\frac{1-p_j}{1-q_{\epsilon,j}}\right].
\]

실제 \(p_j\in(0,1)\)일 때

\[
g_k=\frac{\partial f}{\partial w_k}
=\frac19\sum_{j=1}^{9}V_{kj}\,[\operatorname{logit}(p_j)-\operatorname{logit}(q_{\epsilon,j})].
\]

현재 `objective_and_gradient`가 이 식을 구현하고 있다. 평균과 합을 혼동하면 허용 gap의 단위가 9배 달라지므로, objective·gradient·gap 모두 **규칙당 평균 nats**로 통일해야 한다.

## Frank–Wolfe gap이 주는 것

convexity에 의해 모든 \(z\in\Delta_{10}\)에 대해

\[
f(z)\ge f(w)+g^\top(z-w).
\]

따라서 \(f^*=\min_{z\in\Delta_{10}}f(z)\)와

\[
G(w)=g^\top w-\min_k g_k
\]

사이에

\[
0\le f(w)-f^*\le G(w),\qquad
f(w)-G(w)\le f^*\le f(w)
\]

가 성립한다. 선형함수의 simplex 최소는 꼭짓점 하나에서 달성되므로 `min(g)`를 쓰는 것이 맞다. 별도 비선형 최적화를 한 번 더 돌려 lower bound를 구할 필요가 없다.

이 식은 **full simplex**에 대한 bound다. solver가 \(w_k\ge10^{-12}\)를 두고 최적화하더라도, 최종 feasible candidate의 gradient를 사용해 `g@w-min(g)`를 계산하면 더 큰 원래 simplex 문제와의 오차를 상한으로 잴 수 있다. 반대로 floor를 만족하는 점들만으로 gap의 선형 최소를 계산하면 더 약한 restricted 문제의 bound로 바뀐다.

`solver.success=true`는 이 bound를 대체하지 않는다. `ftol=1e-13`도 그 자체로 실제 최적화 오차가 \(10^{-13}\)임을 뜻하지 않는다. 현재처럼 실제 후보의 계산된 gap을 별도로 검사하는 것이 옳다. 성공 플래그가 false여도 feasible candidate에 작은 gap이 있으면 수치적으로 유용한 후보일 수 있으며, 두 정보는 따로 기록한다.

## 경계와 clipping의 핵심 주의점

### 후보 p를 clip한 gradient는 원래 문제의 gradient가 아니다

Bernoulli entropy의 gradient는 \(p_j=0,1\)에서 발산한다. `logit(clip(p,delta,1-delta))`를 사용한 뒤 원래 \(f(w)\)의 gap이라고 부르면 정당화되지 않는다. 현재 코드는 **실제 p**가 내부에 있지 않으면 예외를 내므로 이 오류를 피한다.

패턴 집합에 전체 0과 전체 1이 모두 있고, 모든 weight가 양수이면 모든 marginal은 내부점이다. floor와 최종 재정규화는 이 목적에 적절하다. 다만 재정규화 후의 실제 weight·p를 사용해야 하며, 제약 오차·수치 범위를 숨기면 안 된다. finite check도 필요하다.

### q clipping과 평가 점수 clipping은 다르다

목적함수에서 q를 \(q_\epsilon\)으로 바꾸는 것은 미리 선언한 입력 안정화다. 그 뒤 f와 gradient는 동일한 \(q_\epsilon\)을 사용해야 한다.

반면 evaluator는 후보 p도 clipping한 점수 \(L_\epsilon(y,p)\)를 계산한다. 작은 optimization gap만으로 이 수정된 점수의 모든 결과에 대한 비악화가 자동으로 나오지는 않는다. 따라서 다음 별도 검사가 필요하다.

\[
\max_{v\in V}[L_\epsilon(v,p)-L_\epsilon(v,q)]\le0.
\]

이는 실제 관측 라벨을 요구하지 않는다. 가능한 10개 패턴을 모두 검사하기 때문이다. 양의 허용오차를 쓰지 않는 현재 기준은 적절하다. 아주 작은 양의 rounding 차이도 보수적으로 거절될 수 있으며, 그 경우를 solver 실패나 근본적 impossibility와 혼동하지 않는다.

`estimated_fw_gap<=1e-8`, feasibility 검사, 가능한 결과의 점수 검사는 서로 다른 조건이다. 각각의 통과/실패를 별도 필드로 보고하는 편이 좋다. 예를 들어 gap은 작지만 clipped-score 검사를 실패한 후보와, 안전한 점수 후보지만 optimality gap이 큰 후보는 의미가 다르다.

## 유한 정밀도에서의 정직한 표현

위 gap 정리는 정확한 실수 산술의 정리다. 코드의 gap과 score 차이는 float64 추정치다. “허용오차 내의 수치적 convex-optimality 검사”라고 표현해야 하며, exact arithmetic 또는 outward-rounded interval 인증이라고 부르면 안 된다.

- gap은 이론적으로 음수가 아니다. 작은 음의 rounding을 기록 없이 없애거나 큰 음수를 자동 통과시키지 않는다. 비정상적인 음수/비유한 값은 실패로 남긴다.
- 실제 weights가 simplex에 있어야 위 정리가 성립한다. raw solver 출력이 부적합하면 후보를 정해진 방식으로 canonicalize하고 변경량을 남긴다.
- 최종 f가 작은 음수가 되는 rounding 역시 원시 값을 보존하고 설명한다. KL이 실수에서 음수가 될 수 있다는 뜻이 아니다.
- 현재 검사에서 실패한 행을 제외한 평균만 보고하면 품질을 과대평가한다. 전체 256개 model-row, 실패 수, 반환된 원예측 수, 예외 사유가 필요하다.

## 고정 64회 근사 보정과 비교할 때

동결된 보정의 **내부 feasible candidate**와 **guard 이후 실제 반환값**을 구분해야 한다. guard가 거절하면 반환값은 원래 q이며, q는 convex hull 바깥일 수 있다.

비일관적인 q의 KL objective가 0에 가깝다고 해서 coherent reference보다 최적화가 잘됐다고 해석하면 안 된다. 그 q는 동일한 제약 문제의 feasible upper bound가 아니다. 참조 solver와 objective를 비교할 수 있는 대상은 feasible candidate다. 저장된 fallback 출력은 안전성·일관성·반환 비율을 따로 비교한다.

비교에 필요한 최소 기록은 다음이다.

| 대상 | 기록할 것 |
|---|---|
| 원래 q | 입력 clipping, 초기 일관성 |
| 고정 64회 내부 후보 | feasibility, objective, 실제 가능한 출력 점수 차이 |
| 저장된 guarded 반환값 | 원예측 fallback 여부, 일관성, 점수 검사 |
| 참조 후보 | 두 시작점 상태, objective, full-simplex gap, feasibility, canonicalization 변경량 |
| 참조 guarded 반환값 | 수용 여부와 실패 조건, 원예측 반환 여부 |

reference의 행별 SLSQP 시간과 기존 vectorized batch 시간을 직접 나누어 속도 우위를 주장하면 안 된다. 이 감사의 목적은 수치적 충분성 확인이며, 동등한 배치·프로세스·warmup 조건의 throughput 벤치마크는 아니다.

## 표본과 해석 경계

`sha256('projection_audit_v1:'+id)`와 ID의 고정 정렬로 64개를 고르고, HGB seed0 및 독립 MLP seed0/1/2에서 같은 ID를 사용하는 방식은 label-free paired 진단에 적절하다. q값, confidence, solver 난이도, 그룹, 정답·점수로 표본을 골라서는 안 된다. 64개가 모든 난이도·경계 사례를 대표한다고 주장할 근거도 없다.

최대 600초·audit 한 번·고정 두 시작점·최대 1,000회씩이라는 제한을 유지한다. 시간 초과를 포함해 처리하지 못한 행이 생기면 미완료로 남긴다. 관측 결과가 좋지 않다고 표본이나 tolerance를 바꾸는 것은 별도 감사 버전이다.

이 결과에서 가능한 결론은 “선택된 예측 입력에서 기존 solver의 잔여 최적화 오차 또는 guard 거절이 얼마나 있었는가”다. 실제 미래 예측 정확도가 개선됐는지, 어떤 모델이 더 좋은지, 새로운 방법론인지 판단할 자료는 아니다. 클래식 projection과 Bregman 이론에 대한 신규성 주장도 성립하지 않는다.

## 합성 테스트 검토

현재 테스트는 독립 패턴 열거의 일치, analytic gradient의 방향 미분, convex gap의 일차 하한, 동일한 두 marginal 제약의 폐쇄형 해, 이미 일관된 입력, 0/1·극단 입력, 잘못된 입력 거부를 다룬다. 두 좌표를 같게 만드는 경우의 해가 **평균 logit의 sigmoid**라는 검사는 forward 방향을 확인하는 좋은 장치다. 역방향 projection의 산술평균과 구분된다.

추가로 시작점 하나의 예외·둘 다 실패·canonicalization 변경량·검사 기준별 실패 기록을 테스트하면 운영 감사의 완결성이 높아진다. 원래 출력 라벨을 읽거나 새로운 학습을 할 필요는 없다.

**최종 사전 의견:** 위 기록·실패 보존 조건을 반영한 뒤 코드/프로토콜/입력을 동결하면, 제안한 한 번의 제한된 label-free audit를 진행해도 된다. 이 검토 단계에서는 실제 예측 입력을 채점하거나 실측 target을 읽지 않았다.

## 실행 후 독립 검토 — 2026-10-30

사전 의견에서 요청한 시작점별 예외 기록, 원래 weight의 feasibility, canonicalization 변경량이 최종 코드에 반영됐다. runner는 정답 배열을 역참조하지 않고 `ids`, `probabilities`, `repaired_probabilities`만 읽는다. 표본 선택은 고정 ID 해시 순서이며, 네 모델 모두 동일한 64개 ID를 사용한다. 동결 소스와 입력 파일의 SHA-256을 다시 계산한 결과 불일치는 없었고, 결과 seal 및 결과의 manifest hash도 일치했다. 검토자는 `results.json`의 최적화 진단만 집계했으며 실제 target이나 개별 예측 배열을 읽지 않았다.

저장소 루트에서 `.venv-mac/bin/python -m pytest agentic_research/projection_audit_v1/test_solver.py agentic_research/projection_audit_v1/test_runner.py -q`를 독립 실행해 **17개 테스트 통과**를 확인했다. `unittest discover`는 pytest 함수 테스트를 수집하지 않으므로 검증 명령으로 사용하면 안 된다.

### 종료 성공과 최적화 오차 상한은 달랐다

| 입력 | 행 수 | 선택된 SLSQP 성공 | FW gap ≤ 1e-8 | 모든 가능한 target에서 비악화 | 두 조건 모두 통과·채택 |
|---|---:|---:|---:|---:|---:|
| HGB seed 0 | 64 | 64 | 14 | 60 | 13 |
| 독립 MLP seed 0 | 64 | 64 | 11 | 44 | 5 |
| 독립 MLP seed 1 | 64 | 64 | 12 | 39 | 7 |
| 독립 MLP seed 2 | 64 | 64 | 13 | 37 | 3 |
| 합계 | 256 | 256 | **50** | **180** | **28** |

전체 512개 시작점 중 511개가 SLSQP 성공으로 종료했고, 하나는 실제 marginal이 내부에 있지 않다는 `ArithmeticError`로 기록됐다. 이 행의 다른 시작점은 사용 가능했으므로 행이 누락되지는 않았다. 최종 후보 256개는 모두 수치적 coherence 검사를 통과했다. 그러나 선택된 후보 중 **206개는 성공 종료했어도 FW gap 기준을 충족하지 못했다.** 최대 gap은 `2.0746231111e-4`였고, HGB에서도 최대 `1.3366559212e-6`으로 정한 기준보다 컸다.

이는 `ftol=1e-13`이나 optimizer의 success flag가 이 문제에서 `FW gap≤1e-8`을 보장하지 않음을 직접 보여 준다. 목적함수 변화나 내부 종료 조건이 작아지는 것과, simplex의 모든 방향에서 남은 개선 여지를 상한으로 묶는 것은 다른 검사다. 0 또는 1에 가까운 marginal에서는 logit gradient의 수치적 민감도도 커진다. 이번 자료만으로 특정 내부 종료 조건 하나를 실패 원인이라고 단정할 수는 없다.

최종 canonicalization 변경량은 최대 약 `3.9e-15`, feasibility 오차는 최대 `3.4e-16`이었다. 따라서 눈에 띄는 weight 재조정이나 제약 위반 때문에 gap이 커졌다는 증거는 없다. **이 결과물을 모든 행에서 수렴한 reference, exact projection, 고정밀 최적해라고 부르면 안 된다.** 통과한 행의 float64 gap 진단과 통과하지 못한 행을 구분해야 한다.

### 엄격한 비악화 검사는 별도 역할을 했다

FW gap을 통과한 50개 중 22개는 가능한 10개 target의 clipped log loss 비교에서 양의 차이를 보여 거절됐다. 이 22개의 최대 차이는 `6.13e-9`였다. 전체로 보면 양의 차이는 76개에서 나왔고, 그중 30개만 `1e-8` 이하였다. 전체 최대 차이는 `5.1963520180e-5`이므로 **모든 거절을 단순 기계 반올림이라고 설명해서는 안 된다.** 유한한 최적화 잔차, weight floor, score clipping을 분리해 해석해야 한다.

원래 예측이 수치적 coherence 허용오차 안에 있던 25개는 모두 약 `1.85e-12`–`3.99e-12`의 작은 양의 target 차이로 거절됐다. 이미 coherent인 q의 정확한 projection은 q 자체다. 이런 경우 작은 내점 이동조차 어떤 가능한 target의 점수를 악화시킬 수 있으므로, 엄격한 guard가 거절하는 것은 이론과 모순되지 않는다. 다만 여기의 coherence 판정도 허용오차를 쓰는 수치 판정이지 exact arithmetic 증명은 아니다.

채택된 28개만 새 후보를 반환하고 나머지는 원래 예측으로 돌아갔다. 결과적으로 **반환값의 가능한 모든 target에 대한 최대 점수 악화는 네 모델 모두 0**이었다. 이는 현재 guard와 fallback이 의도한 대로 작동했다는 진단이다. 모든 반환값이 coherent라는 뜻은 아니며, fallback한 원래 예측은 incoherent일 수 있다. 낮은 채택률을 그 자체로 예측 성능 악화나 개선이라고 해석할 수도 없다.

### 기존 64회 solver와 비교해 확인한 범위

기존 raw 64회 후보는 256개 모두 coherent였다. 두 solver의 목적함수 차이는 대부분 작았지만, 일부 선택된 입력에서 새 후보의 forward-KL 목적함수가 더 낮았다. 행별 최대 감소는 HGB `0.00011890`, MLP 세 seed 각각 `0.00485853`, `0.00824192`, `0.00469101`이었다. 반대로 새 후보가 더 높은 목적함수를 보인 행도 있었으나 최대 약 `4.23e-13`으로 작았다. 이는 **가능한 후보 두 개의 목적함수 비교**이며, 새 후보가 전역 최적해라는 증거로 쓰면 안 된다. 새 후보 자체의 gap을 함께 보아야 한다.

이 감사는 기존 64회 solver에 개선 여지가 있는 입력이 존재함을 보여 주지만, 모든 입력에서 고정밀 reference를 확보하지는 못했다. 관측된 미래 라벨의 log loss, 모델 순위, joint 구조의 이점에 관한 새로운 결론은 전혀 나오지 않았다. 동결된 v3의 성능 결과나 핵심 가설 판정을 이 결과로 변경해서는 안 된다.

### 후속 진행 조건

이번 감사는 완료된 한 번의 기록으로 보존한다. tolerance를 완화하거나 반복 수를 바꿔 같은 버전을 재실행하지 않는다. 후속 solver v2를 진행한다면 먼저 독립적인 active-set 또는 dual 구현, KKT 잔차, near-boundary 합성 사례, 이미 coherent인 입력의 정확한 identity 반환을 소규모 합성 자료로 검증하고 별도 프로토콜을 동결해야 한다. 목표는 우선 잔여 오차의 통제와 일관된 수치 판정이며, 새로운 예측 성능이나 신규성은 별도 실험 없이 주장할 수 없다.
