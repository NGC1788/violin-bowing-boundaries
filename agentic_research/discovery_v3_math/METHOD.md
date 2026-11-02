# 미래 경로에서 공동 변동량 확률을 계산하는 제한된 decoder

2026-11-02. 이 문서는 합성 수치 프로토타입의 명세다. 신경망 학습·바이올린 성능 개선·새 방법론을 이미 입증했다는 뜻이 아니다. 일반화 이차형식 계산과 gradient에는 직접 선행이 있다([문헌6개](../discovery_v2/PATH_DISTRIBUTION_REVIEW.md), [Das](https://arxiv.org/abs/2404.05062)).

## 경로 모형과 정확히 계산되는 부분

두 채널 \(c\)의 미래46프레임 경로를 다음처럼 제한한다.

\[
Y_c=\mu_c+\ell_c Z+\sigma_c\varepsilon_c,
\quad Z\sim N(0,1),\quad\varepsilon_c\sim N(0,I_{46}).
\]

채널은 shared factor \(Z\)를 공유한다. 각 채널 혁신은 \(Z\) 아래에서 서로·프레임 간 독립이고 채널별 분산은 일정하다. 피치는 cents, 음량은 dBFS로 가정한다. Hz/원래 power로 지수변환하지 않는다. 이것은 일반 시간 공분산 모형이 아니다.

미래 voiced mask는 미리 명세한 유한 후보 \(m\) 중 하나다.

\[
P(M=m\mid Z=z)=\operatorname{softmax}_m(a_m+b_mz).
\]

여기서는 \(M\perp\varepsilon\mid Z\)라는 추가 가정이 있다. 실제 관측기의 mask가 혁신이나 관측값에 의존하는 일반적인 MNAR/selection은 해결하지 않는다. Mask 후보는 미래 정답 mask를 입력한 것이 아니라 생성모형이 내놓는 분포다.

\(k=\sum m\ge23\)인 mask에서 피치 samplevariance \(S_p^2\)에 대해

\[
P(S_p\le u\mid z,m)=
F_{\chi'^2_{k-1}(\lambda_{p,m}(z))}
\!\left(\frac{(k-1)u^2}{\sigma_p^2}\right),
\quad
\lambda_{p,m}(z)=
\frac{\|C_m(\mu_p+\ell_pz)\|^2}{\sigma_p^2}.
\]

\(C_m\)은 선택된 피치 프레임을 중심화하는 연산이다. dBFS는 전체46프레임을 쓰므로 자유도45다. std문턱을제곱하고 ddof1을 맞춘다. 이 ncχ² 법칙은 조건부 iid Gaussian 혁신에서 성립하며 임의의 temporal covariance에 그대로 적용하지 않는다.

조건부 독립을 이용하면 공동 사건은

\[
P(G,S_p\le u,S_d\le v)
=\int\phi(z)\sum_{m:k\ge23}
 P(m\mid z)F_{p,m}(u\mid z)F_d(v\mid z)\,dz.
\]

두 **무조건부** 주변 CDF를 곱한 식이 아니다. Invalid mask도 softmax분모에 남고9규칙에는0기여한다. 이 구분이 공동 의존과 guard확률을 보존한다.

## 수치 적분·정합성·미분

양의 Gauss–Legendre 가중치로 \([-8,8]\)에서 적분한다. 128/256/512 order를 고정했다. 확률 적분의 누락 정규 tail은 \(2\Phi(-8)\) 이하이나, 이 값은 quadrature오차나 gradient tail오차의 증명은 아니다. 가중치를 잘린 영역의 조건부 분포로 재정규화하지 않는다. 부동소수점 합이1을 약간 넘을 때만 아래로 맞추고 누락량은allfalse상태로 둔다.

같은 positive mixture를9문턱 전체에 사용하므로3×3 CDF의cell mass가 음수가 아니며 바깥 영역/invalid는allfalse패턴이다. 이 정합성만으로 예측 정확도가 좋아진다고 주장하지 않는다.

CPU float64 autograd는 알려진 항등식을 사용한다.

\[
\frac{\partial F_{\nu,\lambda}(x)}{\partial x}=f_{\nu,\lambda}(x),
\qquad
\frac{\partial F_{\nu,\lambda}(x)}{\partial\lambda}
=\tfrac12\{F_{\nu+2,\lambda}(x)-F_{\nu,\lambda}(x)\}.
\]

두 번째 식은 ncχ²의 Poisson-mixture표현을 미분하면 얻는다. 새 공식이 아니다. 값·PDF계산은 [SciPy ncx2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html)를 쓴다. Noise학습에서는CDFargument와noncentrality가 모두변한다. Gradient검사는이잘린수치적분 구현에 대한1차 미분만 확인하며GPU·Hessian·전체분포tail의gradient보장은없다.

## 통과 결과의 의미

합성검사통과는 이 제한된 계산 블록을 학습기에 붙여 볼 수 있다는 첫 관문이다. 실제 future mask를 모형이 학습할 수 있는지, finite mask가 실제 데이터에 충분한지, 혁신의 조건부 독립·정규성·등분산이 적절한지는 열려 있다. Source-only 경로자료감사와 새로운 공정한학습비교가 필요하다. Path auxiliary감독이 추가되면 direct MDN도 같은 감독을 받는 대조가 필요하다.

또한 분산을 감독한다고 경로 자체가 식별되는 것은 아니다. 이번 constant-level불변성은 mask law를 고정한 채 mean/loading벡터에 프레임상수를 더한 수학적 사례다. 실제 음량 변화로 관측기의 voiced판정까지 바뀌는 경우에 적용한 주장이 아니고, 절대음량/피치입력이 예측에 불필요하다는 뜻도 아니다.
