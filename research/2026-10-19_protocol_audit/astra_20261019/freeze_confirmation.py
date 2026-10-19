"""Freeze D_T2 predictions from previously exposed training summaries only."""
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=HERE/'d_t2_freeze'
    out.mkdir(exist_ok=False)
    desc_path=BASE/'novelty/lower_protocol/descriptors.json'
    upper_path=BASE/'novelty/upper_law/upper_law_t0.2.json'
    desc=json.loads(desc_path.read_text());upper=json.loads(upper_path.read_text())
    tags=desc['tags']
    if 'D_T2' in tags or len(tags)!=10: raise ValueError('Unexpected training manifest')
    sys.path.insert(0,str(BASE/'novelty/lower_protocol'))
    import analysis as a
    sys.path.insert(0,str(BASE/'final'))
    import confirm_A_T2 as c
    train=c.rows_for('A1')+c.rows_for('A2')
    models=c.C.physics_models(train,{t:1. for t in tags})
    shape_l=models['P-free'][1][0].tolist()
    x=np.array([desc['diagram'][t]['dec5_pl'] for t in tags])
    y=np.array([desc['diagram'][t]['lower'] for t in tags])
    prop=np.array([[a.POMA[t][1],a.POMA[t][2]] for t in tags])
    target=np.log([1.2021,5.766])
    fitted={}
    for name,columns in [('log_Z',[0]),('log_EI',[1]),('log_Z_EI',[0,1])]:
        design=np.column_stack((np.ones(10),np.log(prop)[:,columns]))
        coef=np.linalg.lstsq(design,y,rcond=None)[0]
        fitted[name]={'coefficients':coef.tolist(),'prediction_D_T2':float(np.r_[1.,target[columns]]@coef)}
    coef=np.linalg.lstsq(np.column_stack((np.ones(10),x)),y,rcond=None)[0]
    record={'created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'status':'D_T2 결과 미열람. 다운로드 중지. 로컬 파일 동결이며 외부 사전등록 서비스가 아니다.',
        'training_tags':tags,'training_dec5_pl_N':x.tolist(),'training_lower_levels':y.tolist(),
        'lower_reference_shape_c_v_beta_logsigma':shape_l,
        'protocol_model':{'intercept':float(coef[0]),'slope_per_N':float(coef[1]),
                          'training_descriptor_range_N':[float(x.min()),float(x.max())]},
        'property_models':fitted,'lower_mean_only_prediction':float(y.mean()),
        'D_T2_published_properties':{'Z':1.2021,'EI_in_1e4_Nm2':5.766},
        'upper_strict_log_c':upper['schelleng_global']['strict v/beta']['lnc'],
        'upper_empirical_log_c':upper['schelleng_global']['A1+A2 shape']['lnc'],
        'upper_empirical_reference':upper['A1A2_upper_law'],
        'thresholds_log':{'lower_tolerance':.23,'upper_tolerance':.12,
                          'lower_comparison_margin':.23,'upper_comparison_margin':.12},
        'source_sha256':{str(p.relative_to(BASE)):sha(p) for p in (
            desc_path,upper_path,Path(a.__file__),Path(c.__file__),BASE/'final/compare.py',BASE/'final/data.py',
            HERE/'descriptor_only.py',Path(__file__))}}
    (out/'frozen_parameters.json').write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    plan=f'''# D_T2 확인 실험 계획 — 결과를 보기 전 동결

작성: {record['created_kst']}. D_T2 다운로드는 중지 상태다. 이 문서는 로컬 사전 계획이며 공개 사전등록이 아니다.
기존 10개 도표는 모두 개발에 노출되었다. D_T2 및 그 뒤 자료의 경계 결과는 보지 않았다.
상세 수치와 입력 해시는 `../astra_20261019/d_t2_freeze/frozen_parameters.json`에 고정했다.

## 1. 질문과 해석 범위

D형 고장력 도표 하나에서, 아래 경계 수준은 라벨 없이 구한 힘 변화 기술자로,
위 경계 수준은 공개 임피던스로 옮겨 예측할 수 있는지 확인한다.
도표 하나의 성공은 인과 기전이나 보편 법칙의 입증이 아니다.
기술자는 대상 도표의 6,000번 측정 중 낮은 공칭 힘 2,400번의 제어 신호를 쓴다.
따라서 이는 대상의 라벨 없는 측정을 사용하는 예측이며, 물성만 쓰는 예측과 정보량이 다르다.

## 2. 자료·판정·경계

- 대상: `2024-06-17_TypeD_T2_sample1`, Z=1.2021 kg/s, EI=5.766×10⁻⁴ N m².
- 주 분석: 길이 0.2초, 시작 t=0.2초. 결정은 창 끝 t=0.4초에 가능하다.
- 분류기와 열 구분은 기존 `final/data.py`와 `final/compare.py`의 동결본을 따른다.
  각 속도 2,000번, 총 6,000번, 40×50 격자와 누락·중복·오류 행을 먼저 확인한다.
- K=2는 **같은 β에서 힘순으로 이웃한 서로 다른 긋기 두 개**의 H 판정이다.
  한 긋기의 시간상 연속 창 두 개를 뜻하지 않는다.
- 미검출을 하한의 오른쪽 검열로 두는 것은 기존 단일 띠 가정에 따른 계산 규약이다.
  위·아래 경계가 실제 존재한다는 보장이 아니다. 모든 검열·미검출 수를 함께 보고한다.
- 수준은 A1+A2로 고정한 기준 모양에 대한 로그 힘 이동이다. 평균 물리적 힘 자체가 아니다.

## 3. 미리 고정한 예측

아래 경계 기준 모양은 log F = {shape_l[0]:.12g} + ({shape_l[1]:.12g}) log v + ({shape_l[2]:.12g}) log β 이다.

1. 절차 기술자 모델: Δ_lower = {coef[0]:.12g} + ({coef[1]:.12g}) dec5_pl.
   dec5_pl은 음수일 때 힘 감소다. 세 속도의 중앙값을 동일 가중 평균한다.
   각 속도에서 공칭 힘 순위 1~20의 800번을 사용한다.
   [0,.1]초 평균 힘과 v·t≥.05를 처음 만족하는 [.t,.t+.1]초 평균 힘의 차이다.
   정확한 순간 5cm 힘 차이나 출발 전 힘과의 차이로 부르지 않는다.
   기술자 계산은 `descriptor_only.py`에서 제어 열만 읽는다. 브리지 파형과 H 라벨은 읽지 않는다.
   학습 범위는 [{x.min():.9f}, {x.max():.9f}] N이다. 범위 밖이면 외삽으로 표시한다.
2. 물성 대조군: log Z+log EI 회귀를 주 대조군으로 고정한다.
   D_T2 예측은 {fitted['log_Z_EI']['prediction_D_T2']:+.9f} log이다.
   log Z 단독 {fitted['log_Z']['prediction_D_T2']:+.9f}, log EI 단독 {fitted['log_EI']['prediction_D_T2']:+.9f}도 모두 보고한다.
   평균만 쓰는 기준선은 {y.mean():+.9f}이다. 결과를 보고 대조군을 바꾸지 않는다.
3. 위 경계 주 예측: log F = {record['upper_strict_log_c']:.12g} + log(1.2021) + log v − log β.
   상수는 기존 저장력 5개 도표로 고정했다.
4. 위 경계 보조 예측: A1+A2 경험적 모양(v 지수 {upper['A1A2_upper_law']['a_v']:.6f}, β 지수 {upper['A1A2_upper_law']['b_beta']:.6f}) + log Z + 고정 상수.
   기존의 수준 RMSE 0.039는 이 경험적 모양의 수준 전이에 대한 값이다.
   엄격한 cZv/β가 모든 열을 4% 오차로 맞혔다는 뜻이 아니다.

## 4. 판정과 불확실성

주 지표는 도표 수준의 절대 로그 오차다. 열별 유한 구간 중점 오차와 적중률은 보조 지표다.
오류·검열·예측 불능을 분모와 함께 보고하고 임의로 제외하지 않는다.

- 아래 경계 허용오차 0.23 log, 위 경계 허용오차 0.12 log를 고정한다.
  같은 조건 반복의 관측된 차이와 위 경계 부트스트랩 폭을 참고한 실용 기준이다.
  전체 세션 분산의 95% 한계나 통계적 동등성 한계로 해석하지 않는다.
- 절차 모델이 허용오차 안이고 물성 주 대조군보다 0.23 log 이상 정확하면,
  이 한 도표에서 절차 기술자 예측이 지지된다고 판정한다.
  반대 차이가 0.23 이상이면 반증 방향이다. 그 사이는 구별 불충분으로 보고한다.
  평균 기준선과의 차이도 같이 보고한다.
- 위 경계는 엄격한 고정식의 수준 오차가 0.12 안인지 확인한다.
  같은 상수에서 log Z만 뺀 식과 비교하되, 차이가 0.12 미만이면 Z의 우월성은 판정하지 않는다.
- 위치 40개를 속도 간 묶어서 2,000회 재표집한다(시드 20261019).
  이는 해당 세션의 조건부 민감도 구간이며 새 세션의 모집단 불확실성이 아니다.
- 미검출·검열 때문에 유한 수준 추정이 불가능하면 '검증 불가'로 남긴다.
  최적화 실패를 결과에 맞는 시작값 선택으로 숨기지 않는다.

## 5. 실행 순서·보조 분석

Wi-Fi 재개를 사용자가 승인한 뒤 기존 단일 파이프라인으로 받는다.
체크섬과 공간을 확인하고 제어 신호 기술자부터 저장한다. 그 값으로 예측을 계산·해시 기록한 뒤
처음으로 목표 라벨과 경계를 산출한다. 한 번 채점하고 성공·실패 모두 기록한다.
지금은 다운로드나 채점을 실행하지 않는다.

K=3, t=.1/.3, 공칭 힘 축은 미리 정한 보조 분석이다. 주 분석 실패를 이 분석으로 뒤집지 않는다.
새 모델·변수·임곗값·예외 처리를 추가하면 별도 사후 분석으로 표시한다.
원본·기존 확인 결과를 덮어쓰지 않는다. GitHub push는 하지 않는다.
'''
    path=BASE/'final/CONFIRM_PLAN_D_T2.md'
    with path.open('x') as f:f.write(plan)
    (out/'freeze_receipt.json').write_text(json.dumps({'plan_sha256':sha(path),
        'parameters_sha256':sha(out/'frozen_parameters.json'),'created_kst':record['created_kst']},indent=2)+'\n')
    print(plan)


if __name__=='__main__':main()
