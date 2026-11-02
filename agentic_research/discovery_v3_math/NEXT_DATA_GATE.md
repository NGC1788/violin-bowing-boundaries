# 다음 한 작업: source-only 미래 경로와 관측 마스크의 준비 감사

2026-11-02. v3의순수합성감사는완료·STOP으로보존한다. 다음은별도 `path_source_gate_v1`의자료준비단위다. 아직그구현/자료생성/학습은수행하지않았다. v3폴더에서는실제자료를열지않는다.

## 목적

수치decoder가가정한finitefuturemask·경로단위가실제source34명자료와얼마나맞는지확인한다. 수치PASS를바로예측성공으로이어붙이지않는다. 현재decoder가해결한것은제한된1factor·조건부whiteGaussian모형이며실제observer의value-dependent선택은열려있다.

## 준비할 것

1. 최신작업선점·실제프로세스/lock/Git과ledger를확인하고별도data준비계획·누수시험·코드검토·입력/소스동결을먼저만든다. 준비시간600초이내,새학습0건,추가다운로드없음이다.
2. 기존raw_v1의 **원train34명1442녹음만** 선택한다. olddev/test정답은hash도열지않는다. discovery_v1의prefix/분할과동일한source순서·ID를검증한다.
3. 각음원의미래1.25–1.75초를먼저잘라46frame의cents/dBFS/periodicity/voiced를 **정답자료**로추출한다. Prefix는별도파일에그대로보존하고미래경로/guard/길이를입력에넣지않는다. 실제음원의미래변조입력불변성을다시검사한다.
4. 경로에서재계산한voicedpitchstd(ddof1),전체powerstd(ddof1),guard23/46이기존train_labels의연속목표·guard와일치하는지검증한다. invalid를제외하지않는다. 원래양자화문턱/Y9는모델fold에재사용하지않는다.
5. 미래mask의관측개수분포와고유패턴수는aggregate로만보고한다. **각fold의innertrain에서만** mask사전을만들었을때heldsource의미지패턴비율을계산한다. Held미래mask에서출력후보를만들거나해당record의실제mask를추론입력으로사용하지않는다.
6. finite-mask모형이부족하면모든heldmask를합쳐“해결”하지않는다. 새로운prefix-conditionedmask모형/명시적fallback을검토하거나finite-catalogue가설을접는다. Rawpath상관을모형의innovation상관과동일시하지않는다.

## 다음 학습을 하기 전에

- Head가경로자료의추가감독을받으면directMDN도동일한auxiliary감독을받는대조를설계한다.
- 학습된noise가작아지거나loading이커지면고정quadrature가이번toy보다훨씬날카로운CDF를만날수있다. 파라미터범위/정확도검사와계산예산을학습전에정하고,미래라벨없이order민감도를검사한다. 학습기가적분오차를이용해좋아지는현상을막아야한다. 현재fixture통과를모든학습파라미터의수치정확성보장으로읽지않는다.
- Source내부문턱/scaler/prior/모든학습·stopping은innertrain만쓴다. 동결한같은사람fold를반복사용한다는탐색적한계를유지한다.
- Mask를포함한모형·jointCDF근사·gradient·낮은rank/scaling비용을명세하고반박대조를둔다. 기존closedcampaign이나이번수치감사를고쳐재실행하지않는다.
- 새학습은공유LOOP_BUDGET의추가3batch범위에서별도계약으로최대12job/1800초/작업180초다. 아직한개도소진하지않았다. 수치감사는1/2개사용했다. 이자료준비는그자체가학습실험이나실제성능검증이아니다.
- 원음·행별경로/마스크/목표·개인자료는private에두고코드·aggregate·hash만허가된브랜치로공유한다.
