# 두 번째 탐색: 정확한 현재 상태 × attention × 연속 변화 감독

2026-11-02. 고정12학습을완료했다. **모든모델이HGB보다낮은손실이지만Markov를넘지못했다.** 이번batch만닫으며사용자가허용한넓은탐색전체를종료하지않는다. 아직novelty·독립성능개선은입증되지않았다.

v1과같은원source34명1442녹음의3사람fold만썼다. 기존dev/test정답은hash도열지않았다. 반복해서본source의적응적탐색으로독립확증이아니다. innertrain만의문턱·scaler·prior를사용했고모든모델에게60요약특징+11현재상태+2unclipped log1p변동량anchor를동일하게주었다. attention은96prefix프레임과위치특징도사용한다.

합성4시험·별도AI소스검토후코드5/부모입력5를동결하고12학습을1회실행했다. 총학습11.583초,각60epochs/seed0. 모든가중치와train BCE곡선을private로보존했다. 별도결과검토에서소스/입력·12예측+12가중치hash·영수증metadata·집계·상호작용을재계산해불일치0건이었다. 영수증파일자체의hash는원결과봉인에없었으나metadata는봉인된결과와일치했다. 인간심사나독립경험적확증은아니다.

| 방법 | 사람평균BCE↓ | 최대사람손실 | 모수수 |
|---|---:|---:|---:|
| Markov | **.542081** | **.705669** | 기존fold별확률표 |
| v1 HGB참조 | .581722 | .848416 | v1그대로 |
| SummaryMLP joint | .554539 | .845300 | 3,818 |
| SummaryMLP delta-MDN | .563987 | .858401 | 3,851 |
| Attention joint | .568173 | .870050 | 8,490 |
| Attention delta-MDN | .558815 | .860359 | 8,523 |

모든모델에서두비교군을동시에이긴fold는1/3이다. 사전정한2%추가이득·2/3fold방향screen을통과한후보는없다. HGB대비이득만강조해Markov실패를숨기지않는다. 이점수는fold별로학습한문턱을쓰므로oldtest점수와직접비교할수없다.

## 조합에 따른 상호작용

delta-MDN은jointhead대비MLP손실을.009448늘렸고,attention손실은.009358줄였다. 두차이의차이는 **+.018806**이며세fold모두양수다. 다만delta는MLP에서세fold모두나빴고attention에서는2개좋음/1개나쁨이었다. Attention+delta가Markov를넘었다는뜻은아니다.

이것은bundled변경의서술적상호작용이다. delta는head·anchoring·연속보조loss를동시에바꾸고attention은context·모수수도늘린다. 보조loss의일반적인정규화효과와시간구조의실익을구분하려면새별도대조가필요하다. 상호작용하나로새방법이나인과기전을선언하지않는다.

## 학습 곡선으로 알게 된 것

모든12곡선의train BCE가60epoch까지내려갔다. 마지막10epoch의감소폭은.00411–.01609로,수렴을확인했다고말할수없다. 최종train과heldfold간격은모두양수(.00756–.21801)다. 일반화격차와맞는결과지만최적epoch·과학습시작점·추가학습효과를확인한것은아니다. 사후에유리한epoch로성능을바꾸지않는다.

## 다음 후보

같은보정이나architecture를몰래다시튜닝하지않는다. [NEXT_EXPLORATION.md](NEXT_EXPLORATION.md)에가설과다음한작업을정리했고, [경로분포문헌검토](PATH_DISTRIBUTION_REVIEW.md)에서구조적으로다른출력표현을검토했다. 미래경로분포를창의표본분산분포로전파하는것은directMDN과다르지만고전χ²CDF·Gaussianpath·lowrank구조자체는선행이다. 두채널joint통계와unknownvoicingmask가핵심미해결점이다. 먼저데이터를쓰지않는수치/gradient반박시험을별도버전으로정하고,성립하면새학습batch를검토한다.
