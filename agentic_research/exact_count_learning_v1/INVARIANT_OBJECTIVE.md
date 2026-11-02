# 학습 목표에서 미래의 공통 레벨을 제거하는 이유

이 문서는 원자료를 보지 않고 학습 코드의 설계 결정을 명세한다. **새 학습 결과나 새로운 수학 정리의 주장이 아니다.** 주 연구는 결측이 있는 시계열의 구조적 확률 예측이다.

예측 대상인 창의 표준편차는 모든 프레임에 같은 상수를 더해도 변하지 않는다. 반면 원시 미래 경로의 Gaussian likelihood는 그 상수까지 맞춰야 한다. 음높이의 서로 다른 기준 레벨이나 음량 offset이 label 정규화와 보조 손실을 지배하면, 주 과제와 무관한 회귀를 어려운 보조 과제로 추가하게 된다. 이를 해결하기 위해 미래 경로를 임의로 iid라고 다시 간주하는 대신 **대조 좌표의 정규화된 Gaussian 밀도**를 학습한다.

마스크로 선택한 pitch 벡터 길이를 \(n\)이라 하고, \(C_n=I-11^\top/n\)라 두자. \(U\)는 \(U^\top U=I_{n-1}\), \(UU^\top=C_n\)인 직교 대조 기저다. 조건부 등분산 Gaussian 경로에서

\[
Z=U^\top Y\sim N(U^\top\mu,\sigma^2I_{n-1}),
\]

따라서 정규화된 log density는

\[
\log p(Z\mid M,h,x)=
-\frac{\|C_n(Y-\mu)\|^2}{2\sigma^2}
-(n-1)\log\sigma-\frac{n-1}{2}\log(2\pi).
\]

구현은 기저를 만들지 않고 centered residual의 제곱합을 쓴다. 중요한 것은 자유도다. 이미 중심화한 \(n\)개 프레임을 독립 표본처럼 취급하여 \(-n\log\sigma\)를 넣으면 다른 목적함수가 된다. \(n=0,1\)인 pitch는 0차원 대조 공간의 log density0으로 정의한다. 그런 녹음도 **mask likelihood와 전체 power의 \(T-1\)차원 contrast likelihood에는 계속 기여**한다.

잠재 regime을 합친 공통 보조 목적은

\[
\mathcal L_{aux}= -\frac1T\log\sum_h\pi_h(x)
 p_h(M\mid x)
 p_h(U_M^\top Y_p\mid M,x)
 p_h(U_T^\top Y_d\mid M,x).
\]

마스크마다 다른 차원/기저를 쓰는 정규화된 조건부 밀도를 이산 mask 확률과 결합한 것이다. 이는 원시 경로 전체의 밀도가 아니라 **레벨을 버린 관측 표현의 밀도**다. 기저 변경은 직교변환이므로 이 밀도 값은 기저 선택에 의존하지 않는다. 고전 Gaussian projection과 nuisance 제거의 적용이며 이것만으로 novelty를 주장하지 않는다.

Generator는 구간별 대조와 scale을 예측하고, 평균에는 길이 가중 zero-sum 제약을 둔다. 전역 레벨 파라미터는 제거한다. Count decoder의 표준편차 법칙은 바뀌지 않는다. Label 단위는 inner train의 pooled within-record contrast scale로 정한다. 서로 다른 녹음의 전체 레벨 분산으로 noise scale을 정하지 않는다.

미래 관측 마스크와 중심값은 **학습 정답을 변환할 때만** 사용한다. 추론은 prefix와 inner-train constants만 받고 실제 미래 마스크·미래 평균을 받지 않는다. Direct joint/MDN과 exact/moment count 네 방법 모두 같은 대조 정답과 보조 generator를 받는다. 상대적 정보량을 숨겨 새 decoder의 효과로 주장하지 않는다.

한계도 명확하다. 이것은 고정된 미래 마스크 아래에서 label 공통 이동에 대한 보조 목적의 불변성이다. Raw audio gain이 유성 판정을 바꾸는 상황, 신경망 입력 이동에 대한 불변성, 임의 MNAR/상관 혁신의 처리를 보장하지 않는다. 경로 모형의 조건부 iid 가정과 실제 겹치는 기술자 프레임의 관계는 여전히 미검증이다.

초기 미동결 코드의 raw-level 보조목표 기능시험13개가 통과한 뒤, 실제 데이터나 설정 탐색 없이 위 nuisance 문제를 발견해 학습 준비 코드를 수정했다. 초기 파일 해시는 private history에 남겼다. 이는 완료된 수치 감사나 결과를 재실행·변경한 것이 아니다. 새 코드에는 대조 차원·알려진 밀도·label 이동 불변성을 검증하는 작은 기능시험을 추가한다.

직접 선행의 범위: [Patterson & Thompson(1971)의 원문 요약·서론](https://era.ed.ac.uk/server/api/core/bitstreams/c460efa4-dd25-47bc-984b-1652df8d4562/content)은 error contrasts의 likelihood를 최대화하는 방법을 제시한다. [Harville(1974)의 출판사 초록](https://academic.oup.com/biomet/article-abstract/61/2/383/268084)은 그 likelihood의 표현과 variance-component 추론의 관계를 설명한다. 이번에는 검색 도구가 제공한 앞 논문의 요약/서론 추출문과 뒤 논문의 초록을 확인했다. 직접 열기는 timeout/접근 오류로 실패했으므로 전문 열람이나 모든 정리·현대 신경망 적용을 검증했다고 하지 않는다. 현재 mask-conditioned neural mixture의 학습이 곧 고전 REML 추정이라는 주장도 하지 않는다. Contrast likelihood 자체는 명백히 선행 원리다.
