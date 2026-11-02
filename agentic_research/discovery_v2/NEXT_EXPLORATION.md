# 다음 탐색: 구조가 다른 가설을 작게 구현한다

2026-11-02. 기존보정인증·discovery_v1/v2는해당batch가종료됐다. 사용자의새요청은넓은탐색을계속하는것이다. 다음작업은완료된점수의개선주장이나자동재학습이아니다.

## 다음 한 작업

별도 `discovery_v3_math`에서실제자료를쓰지않고 **미래경로→두표본분산의공동확률** decoder의수치가능성을시험한다. 아직그폴더·구현·실행결과는없다. 학습입력은바이올린prefix,출력은원래cents/dBFS변동량으로유지할후보이며새정답으로기존실패를덮지않는다.

Gaussian경로라면고정voicing mask에서samplevariance는중심화이차형식이다. 이구조가두연속값을바로출력하는MDN보다적은자료에맞는inductivebias를줄수있다는가설이다. 신경망·시계열·고전분포계산을결합하되그결합만으로신규성을선언하지않는다.

### 작은 반박시험 목록

1. IID/비중심Gaussian특수사례의분산CDF를chi-square/noncentral-chi-square참조와맞춘다. 표준편차문턱을제곱하는지,ddof=1과실제46프레임을맞추는지검사한다.
2. 두채널이상관되면주변CDF의곱이joint규칙확률과다름을검사하고sharedpath/conditionalfactor적분으로joint를보존할수있는지확인한다.
3. 실제미래mask를입력으로사용하지않고mask/guard불확실성을적분하는경우와잘못고정한경우를비교한다. 별도mask모형을생략해야만좋아진다면현재문제에맞지않는다.
4. 이차형식CDF의parametergradient를독립유한차분/알려진특수경우와비교한다. Indicator표본에보통reparameterization을쓰면미분이0이되는문제를숨기지않는다.
5. 경로평균/공분산의일부변경이같은samplevariance분포를주는비식별성을검사한다. fullpath를물리적으로회복했다고주장하지않고필요한centeredfunctional law만학습할수있는지검토한다.

가능한기술차이는공동이차함수/미래mask를다루는효율적이고안정적인decoder또는실제자료에서검증된sampleefficiency다. 검토한선행에는일반화χ²CDFgradient도있으므로“미분가능하게만들었다”는공백은이미없다. [직접선행6개](PATH_DISTRIBUTION_REVIEW.md)를먼저읽는다.

이단계는새source-only학습의허가·동결을대신하지않는다. 수치시험도작은범위/오차/실패기준을먼저고정하고CPU600초이내로한다. 틀린경우해당가설을수정하거나버리되다른표현탐색전체를닫는결론으로확대하지않는다.

## 별도로 남아 있는 축

- Modernsmall-data/pretrained표현을강한기준선으로검토한다. TabPFN/TabICL/음악모델은설치·라이선스·메모리·비용을먼저확인하고source-only규약에서실험한다. 현재아무것도실행하지않았다.
- Prefix내부자기지도·cross-personretrieval·spectral/latentdynamics조합은 [가설은행](../discovery_v1/IDEAS.md)에있다. 같은메뉴를무작위로나열해유리한모델만선택하지않는다.
- v2의attention×delta상호작용은추가대조후해석한다. BCE-onlyMDN·auxiliaryloss-only·동일감독의pathhead등으로필요한요소를분리할수있다. 현재는기전미확인이다.

큰가설도시도하되각batch에서배운것을다음가설에반영한다. 현재24개탐색학습에서Markov를넘는후보나높은novelty는확보하지못했지만,그결과가이후의모든가능성을배제하지는않는다.
