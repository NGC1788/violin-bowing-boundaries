"""Apply the reviewed 2026-10-21 additions to backed-up research drafts once."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new, 1)


paper_path = ROOT / 'paper/draft_v2.md'
paper = paper_path.read_text()
assert paper == (HERE / 'before_update/draft_v2.md').read_text(), 'Concurrent paper change: review first'
paper = replace_once(paper, paper.splitlines()[5],
    'Draft v2.6, 2026-10-21. Working research draft, not submitted and not yet shortened to a journal format. This revision corrects the universal interpretation of criterion tilt and adds §3.11: mathematical counterexamples, a conditional feature-replay model, and endpoint locality. Earlier frozen scores are preserved; the new analyses are retrospective. Previous reviewer responses are in `response_to_reviewers_v2.md`.')
a = paper.index('Public robotic bowing data')
b = paper.index('\n\n---', a)
paper = paper[:a] + '''Public robotic bowing data can mix physical limits with measurement and classification conventions. We audited a public cello-string collection and scored locally frozen boundary predictions once. Only four of nine frozen tests of the strict maximum-force law cZv/β passed their declared tolerances; all four high-force sessions failed. A same-session load-decrease descriptor did not establish better minimum-force prediction than a lookup baseline. Reading force at bow onset rather than in the analysis window shifted the fitted minimum-force β exponent by about 0.2 (post hoc). We then examined how feature thresholds alter the boundary inferred from force-adjacent Helmholtz labels. Empirical threshold-tilt responses were consistent across the tested diagrams, but finite-grid counterexamples disprove a universal direction rule or neutrality obtained by matching the feature's marginal β scaling. Identical feature distributions within β columns can produce different exponent shifts when their force ordering changes. In a retrospective replay of 13 diagrams and four features, adding force to a Gaussian log-feature model reduced exponent-shift MAE from 0.13365 to 0.07708 (42.3%) under matched eligibility and a numerically stable likelihood. The model retained the observed base labels and force grids, so this is conditional explanatory accuracy, not prediction on new strings. It also predicted a neutral point absent from the second-flyback-ratio observations. Our contribution is a reproducible audit of how protocol, feature–force dependence and discrete boundary extraction constrain the interpretation of bow-force laws.''' + paper[b:]
paper = replace_once(paper,
    'Every result is labelled frozen or post hoc. "log" denotes the natural logarithm throughout.',
    'We also contribute a criterion-sensitivity analysis with explicit counterexamples and conditional feature replay (§§3.10–3.11). This application does not introduce a new interval-censoring estimator. Every result is labelled frozen or post hoc. "log" denotes the natural logarithm throughout.')
paper = replace_once(paper,
    'A third T1 sample of type A (A3) was processed but is not analysed here (it was excluded from the frozen tests because processing began before the predictions were frozen); A4 (a fourth A_T1 sample) was a frozen target (Sec. 3.3).',
    'A3 and A4 were included in later analyses: A3 was scored under FROZEN_v1, with its timing limitation stated in §3.2, and A4 was a frozen target (§3.3). Later minimum-force and criterion analyses use an expanded set of diagrams, with their own denominators stated in §§3.8–3.11.')
paper = replace_once(paper,
    'Laws log F = c + a log v + b log β + Δ_d are fitted by interval-censored Gaussian likelihood; Δ_d is a diagram\'s offset from a reference shape.',
    'Laws log F = c + a log v + b log β + Δ_d are fitted by interval-censored Gaussian likelihood; Δ_d is a diagram\'s offset from a reference shape. Interval-censored likelihood is established methodology [21], not a new estimator introduced here. For §3.11, a stable log-CDF-difference implementation and convergence diagnostics are used; the previously frozen scores are not replaced.')
a = paper.index('### 3.10 ')
b = paper.index('## 4. Discussion', a)
paper = paper[:a] + '''### 3.10 Empirical criterion-tilt responses (locally planned tests on exposed data)

The amplitude-floor and corner-width observations motivated a fixed-removal experiment (`novelty/tilt_principle/`). For a feature threshold X ≥ Kβ^a, K was chosen within each speed folder to remove approximately 15% of base-H windows, and a was varied over a fixed grid. The original amplitude results over 14 diagrams rose monotonically with a: mean exponent shifts were −0.65, −0.27, +0.14, +0.35, +0.60, +0.75 and +0.85 for a = −1.5 … 0. The neutral point was −1.09, compared with the marginal amplitude slope −0.92; these were within the declared tolerance of 0.25. Height behaved similarly. The tests used already exposed diagrams: freezing the analysis before execution does not constitute validation on new data.

The original four broadly available features (amplitude, flyback height, periodicity and second-flyback ratio) had the predicted empirical direction in 14/14 diagrams each. Corner width was available only for B1: its upper-cutoff response reversed direction, with a neutral point 0.76 versus a marginal feature slope 0.59. Thus the earlier wording “five features, 14/14 each” was incorrect. Neutral-point agreement within 0.25 held for four of the five feature analyses, but the ratio curve had no zero crossing in [−1,1]. The proposed direction did not transfer to the maximum boundary (small effects, ≤0.11, with the opposite sign), and using only boundary-deciding windows did not fix the discrepancy. Periodicity tilt ±0.2 shifted the minimum-force exponent by about one unit in the original summaries.

These are empirical responses, not a universal tilt law. The original 14-diagram files remain an audit trail; §3.11 excludes D_T2 because the 1 mN force clamp produces zero-width intervals and discloses four numerical-fit discrepancies in the remaining curves. Matching a threshold's tilt to a feature's marginal slope does not guarantee an unbiased physical boundary. Nor does a neutral response identify the correct Helmholtz criterion.

### 3.11 Force ordering is necessary for quantitative criterion-response modelling (post hoc)

**Identifiability.** For exactly m removals of the smallest log X − a log β, the sum of log β among removed windows is nondecreasing with a (the two optimality inequalities for the selected sets give this result). This rank property does not imply monotonicity of a boundary exponent. A finite-grid example using the actual K=2 boundary definition and a nondegenerate interval-normal fit gives an exponent change 0 → −0.11905 when a increases 0 → 1, despite the expected rank shift. A second example has a = b = 0 but moves the exponent 0 → +1. Permuting its features within one β column preserves the entire feature distribution in that column, the marginal slope, β and force grids, base labels, removal fraction and removed β columns, yet changes the exponent shift from +1 to 0. Consequently those marginal summaries cannot uniquely determine the fitted boundary response. The joint ordering of feature and force is indispensable (`theory/REPORT.md`, `counterexamples.py`).

**A sufficient model, not a universal formula.** Let x = log β, log F₀ = c + px, u = log F − log F₀, and log X = α + bx + qu + σε for Gaussian ε and q > 0. For the lower cutoff log X ≥ α + k(a) + ax, the additional operational force required for retention probability r is u_r = max{0, [k(a) + (a−b)x − σΦ⁻¹(1−r)]/q}. When this cutoff is active in every column and q and σ are constant, the slope shift is (a−b)/q. Here b is conditional on relative force; it is generally not the marginal H-feature slope. The formula concerns a probability boundary and is not an exact formula for the first observed K-run. Partial activation, finite grids, correlated labels and censoring require additional treatment.

**Conditional replay.** We fixed the retrospective model, feature grids, 20 Gaussian replications per feature and diagram, seed and scoring thresholds before computing the new outputs (`novelty/criterion_identifiability/PLAN.md`). M0 regresses log feature on speed-folder intercepts and log β among base-H windows; M1 adds log window force. Neither model fits the observed exponent shifts. Both regenerate features on the original H, β and force grid, remove approximately 15% using the original quantile rule, and apply the same K=2 extraction followed by the stable interval likelihood. Four features over 13 diagrams contribute 338 diagram–tilt cells; B1 corner width is a separate auxiliary analysis. These are correlated conditions, not 338 independent experiments, and not all 13 diagrams are independent specimens.

**Accuracy and audits.** Against the original stored curves, MAE was 0.13267 for M0 and 0.07720 for M1. Four nonpositive-force H windows were ineligible only for M1; repeating M0 with the identical mask gave 0.13248 → 0.07720 (41.73% reduction). Stable refits of the observed curves exposed four legacy optimization discrepancies exceeding 0.02 in the β exponent (maximum 0.1997). With both matched masks and those stable targets, MAE was 0.13365 → 0.07708 (42.32%). The declared gates—overall M1 MAE ≤0.15, reduction ≥25%, and each feature MAE ≤0.25—passed in each audit. Main replay fits numbered 13,720 (219 numerical fallbacks, zero unresolved failures); the matched-mask sensitivity added 1,660 fits with zero failures. Baseline parity was within 1.31×10⁻⁵. These checks validate the computation, not the physical model.

**Limits and failures.** For amplitude, the mean neutral-point prediction was −1.0933 versus −1.0881 observed. However, M1 predicted a second-flyback-ratio neutral point at +0.8325 where the observed curve never crossed zero. Periodicity improved slightly against stable targets but worsened slightly against the legacy targets; there is no across-feature dominance. The Gaussian model can generate features outside base-H acceptance conditions (expected M1 session-average rates: periodicity outside [0.8,1], 0.054%; ratio ≥0.5, 2.337%). This is a diagnostic surrogate, not a valid joint generator of H waveforms, and these tail violations have not been established as the cause of the ratio failure. The plotted uncertainty is Monte Carlo error at fixed data and fitted coefficients, not a population interval. All replay accuracy is internal to the observed base labels and grids; new-session prediction and causal explanation are untested.

**Endpoint locality.** Using the same 13 diagrams, we counted how the original threshold changes affect the first and last K-runs, without fitting new laws. Across amplitude tilts, 43.6% of originally banded column–tilt pairs changed their lower endpoint, compared with 5.9% at the upper endpoint; 5.7% lost all runs. For periodicity, 16.6% lost all runs. Counts reuse columns across tilts. This locates the discrete pathway by which similar removal fractions affect the boundaries differently; it does not prove the sign of a fitted upper exponent. It also shows why an assumed right-censoring direction for no-band columns can matter. The relevant method contribution is to audit criterion response through feature–force dependence and boundary extraction, with the mathematical and empirical limits made explicit.

''' + paper[b:]
old = 'Regime classifiers have been noted to mislabel weak periodic slipping as Helmholtz motion in simulation [4]; we did not find a quantitative account of how a feature threshold\'s β-scaling biases a fitted boundary law, nor interval-censored likelihood estimation of Schelleng boundaries, in the literature we searched; both are, to our knowledge, new as methods.'
new = 'Regime-classification and analysis-window effects already have quantitative bowed-string prior art [4]. Covariate-adjusted thresholds [19], artificial scaling caused by detection thresholds [20], and interval-censored likelihood [21] are also established. We therefore claim neither the general threshold-bias phenomenon nor a new censoring estimator. The candidate contribution in §§3.10–3.11 is an application-specific account of feature–force ordering, discrete K-run extraction and criterion-response replay, together with counterexamples to simpler neutrality claims. Its novelty is bounded by the literature reviewed, and the replay is not a robustness certificate or independent prediction result.'
paper = replace_once(paper, old, new)
paper = replace_once(paper, '- Include a model-free lookup baseline and state metric denominators.', '- Include a model-free lookup baseline and state metric denominators.\n- Report criterion-response curves and which first/last K-runs they alter; matching marginal feature scaling alone is insufficient.')
paper = replace_once(paper, '- **Small samples and confounding.** 11 diagrams and 4 high-force sessions over 8 type–tension conditions; speed, session, date, tension and scan direction are confounded.', '- **Small samples and confounding.** The original development cohort has 11 diagrams, with later diagrams and four high-force sessions added in separate analyses. Section 3.11 uses 13 diagrams. Speed, session, date, tension and scan direction are confounded; diagrams are not necessarily independent specimens.')
paper = replace_once(paper, '- **Coverage.** All frozen predictions have now been scored once, except PACE* on B_T2 sample 2 (not applicable to its design) and the coldwarm archive (not processed).', '- **Coverage.** All applicable frozen predictions, including the cold/warm analysis, have been scored. PACE* on B_T2 sample 2 was declared inapplicable to its design. No unused confirmation set remains for §3.11.\n- **Criterion replay.** Same-session features and known base labels are used. Gaussian tails can violate base-H acceptance, and second-flyback-ratio neutrality is not reproduced. This does not establish physical label accuracy or new-string generalization.')
refs = '''[19] H. Janes and M. S. Pepe, "Adjusting for covariate effects on classification accuracy using the covariate-adjusted receiver operating characteristic curve," Biometrika 96(2), 371–382 (2009), doi:10.1093/biomet/asp002.
[20] C. Bryant, J. A. Osborne, and A. Shahmoradi, "How unbiased statistical methods lead to biased scientific discoveries: a case study of the Efron–Petrosian statistic applied to the luminosity–redshift evolution of gamma-ray bursts," Mon. Not. R. Astron. Soc. 504(3), 4192–4202 (2021), doi:10.1093/mnras/stab1098.
[21] B. W. Turnbull, "The empirical distribution function with arbitrarily grouped, censored and truncated data," J. R. Stat. Soc. B 38(3), 290–295 (1976), doi:10.1111/j.2517-6161.1976.tb01597.x.

'''
paper = replace_once(paper, '## Figure captions', refs + '## Figure captions')
old = '**Still to do before submission.** Re-render Figs. 1–3 in English; complete [4], [5], [7], [9]; check the JASA-EL abstract limit; decide whether to analyse A3 post hoc; score D2 and B_T2 sample 2 when processed and add them without changing the frozen rules.'
new = '**Still to do before submission.** Choose one primary contribution and shorten the working draft to the selected journal format; audit all figure/data denominators and bibliographic details, including thesis sections; obtain human author review. All frozen targets have already been scored. The criterion-replay figures in `novelty/criterion_identifiability/response/` and the locality figure in `locality/` are English and visually inspected; their Monte Carlo bands are not population confidence intervals. Stronger new-method claims would require further validation beyond this retrospective replay.'
paper = replace_once(paper, old, new)
paper_path.write_text(paper)

report_path = ROOT / 'final/최종보고서.md'
report = report_path.read_text()
assert report == (HERE / 'before_update/최종보고서.md').read_text(), 'Concurrent report change: review first'
report = replace_once(report, '2026년 10월 20일 · 모든 동결 확인 채점 완료판 (09시 갱신)', '2026년 10월 21일 · 판정 민감도 모형·수학 반례·독립 코드 검토 반영판')
row = next(x for x in report.splitlines() if x.startswith('| 판정 문턱의'))
report = replace_once(report, row, '''| 판정 tilt와 최소 힘 지수가 일정 방향으로 변함 | 관측 (보편 법칙·a=b 중립성은 반례로 부정) | 노출 자료의 계획 고정 시험 + 사후 수학 감사 | 지수 | 기존 4특징 14도표, 모서리 폭 B1만 | 첼로 실측 |
| 같은 β별 특징 분포여도 힘 순서를 바꾸면 지수 변화가 달라짐 | 수학적 반례 | K=2·구간 우도 합성 검증 | 지수 +1 대 0 | 합성 격자 | 악기 독립 |
| 힘 항을 포함한 특징 재생이 판정 변경 반응의 MAE를 42.3% 줄임 | 관측 (내부 설명력, 외부 예측 아님) | 사후 계획 고정 + 동일 마스크·안정 우도 감사 | 지수 변화 MAE .13365→.07708 | 13도표·4특징·338종속 셀 | 첼로 실측 기반 재생 |''')
report = replace_once(report, '## 0. 한눈에 보는 증거 지도', '''10월 21일 추가 분석에서는 **판정 기준이 법칙에 미치는 영향을 정량적으로 설명하는 방법**을 개발했다(10.11절). β별 특징값의 분포가 같아도 힘 순서를 바꾸면 경계 지수 변화가 달라지는 반례를 구성했다. 이를 반영해 특징값의 β 의존성에 힘 의존성을 추가한 모형은, 이미 관측한 문턱 변경 반응의 오차를 같은 평가 조건에서 42.3% 줄였다. 이는 새 현 예측이나 물리 법칙의 정확도 개선이 아니다. 둘째 되돌이 비율의 중립점 실패도 남아 있다.

## 0. 한눈에 보는 증거 지도''')
a = report.index('### 10.10 ')
b = report.index('## 11. 연구 기여와 남은 한계', a)
report = report[:a] + '''### 10.10 판정 기준의 기울기에 따른 경험적 반응 (보편 원리 주장은 수정)

고정 진폭 문턱과 모서리 폭 문턱의 영향을 본 뒤, 기준 X ≥ K·β^a에서 a를 바꾸되 속도 폴더마다 기본 H의 약 15%를 제거하는 시험을 계획하고 실행했다. 이미 노출한 자료이므로 독립 확인 자료를 새로 확보한 것은 아니다. 원래 출력은 `novelty/tilt_principle/`에 보존했다.

- 진폭의 기존 14도표 결과는 a가 −1.5에서 0으로 증가할 때 지수 변화가 −0.65에서 +0.85까지 같은 방향으로 움직였다. 중립점 −1.09와 특징의 전체 기울기 −0.92는 미리 정한 허용차 0.25 안이었다.
- 진폭·되돌이 높이·주기성·둘째 되돌이 비율은 각 14도표에서 방향을 확인했다. **모서리 폭은 B1 하나뿐이다.** 이전의 '다섯 특징 모두 14세션'이라는 문구를 정정한다.
- 중립점은 다섯 특징 분석 중 넷에서 전체 특징 기울기와 0.25 안으로 맞았지만, 둘째 되돌이 비율에서는 시험한 [−1,1] 안에 없었다.
- 최대 힘 경계에서는 작고 반대 방향인 효과가 나왔다. 경계를 결정한 창만으로 특징 기울기를 다시 구해도 문제가 해결되지 않았다.

**해석 정정:** 'a=b이면 편향이 없다', '일정 제거율이면 어긋남 1당 지수 약 1이 보장된다', '최소 힘에서는 방향 규칙이 확실하다'를 보편 명제로 쓰지 않는다. 실제 사용한 K=2·구간 우도에서도 반례가 있다(10.11절). 기존 도표의 경험적 방향성은 남지만 그것만으로 모든 셸렝 도표의 지수를 교정할 수 없다. 원래 14도표 집계에는 D_T2의 1 mN 클램프 문제가 있어, 아래 정밀 검토는 이를 뺀 13도표를 쓴다.

### 10.11 힘과 특징의 결합을 이용한 판정 민감도 모형 (2026-10-21, 사후)

**연구 질문:** 전체 특징의 β 기울기와 제거율이 같으면, 경계 지수가 얼마나 달라질지도 알 수 있는가?

**답은 아니다.** 합성 예제에서 β·힘 격자, 원래 H 라벨, β별 특징 분포 전체, 특징의 β 기울기, 제거 비율과 제거되는 β 열을 모두 같게 유지했다. 오직 한 열 안에서 특징값의 힘 순서만 뒤집었다. 그러자 K=2 경계의 적합 지수 변화가 **+1에서 0으로** 달라졌다. 특징 기울기 b=0에 문턱 a=0을 맞춰도 지수가 0에서 +1로 이동하는 반례, a 증가에 지수가 반대로 0에서 −0.119로 움직이는 반례도 확인했다. 특징의 주변 분포만으로는 경계 반응을 식별할 수 없다는 뜻이다.

**새 모형의 구성:** 기존 H와 힘 격자를 고정하고, 특징값을 압축한 확률 모형으로 문턱 변경 실험을 재생한다.

- M0: log 특징 = 속도 폴더 절편 + b·log β + 정규 잔차.
- M1: 여기에 log 측정 힘을 추가한다. 지수 변화량을 회귀의 정답으로 넣지는 않는다.
- 재생한 특징에 원래 tilt 문턱을 적용해 약 15%를 제거하고, 원래 K=2 경계와 안정적인 구간 검열 우도로 지수 변화를 계산한다.
- 진폭·되돌이 높이·주기성·둘째 되돌이 비율의 13도표, 338개 도표×tilt 조건을 평가한다. 모서리 폭은 B1 하나의 보조 분석이다. 이 조건들은 서로 종속적이고 13도표도 모두 독립 표본은 아니다.

| 비교 | M0 MAE | M1 MAE | 감소율 |
|---|---:|---:|---:|
| 최초 계획, 기존 저장 곡선 | 0.13267 | 0.07720 | 41.81% |
| 같은 양의 힘 마스크, 기존 저장 곡선 | 0.13248 | 0.07720 | 41.73% |
| 같은 마스크, 안정 우도로 다시 맞춘 관측 곡선 | **0.13365** | **0.07708** | **42.32%** |

MAE의 단위는 **β 지수 변화량**이다. 측정된 최소 힘의 오차가 42% 줄었다는 뜻이 아니다. 미리 정한 사후 설명력 기준(전체 MAE≤0.15, 감소≥25%, 특징별 MAE≤0.25)은 모두 통과했다. 단, 모형은 평가 자료 자체의 기본 H·힘 격자·특징을 사용하므로 외부 예측이나 독립 확인이 아니다.

**검증과 실패를 함께 보고한다.**

1. 주 재생 적합 13,720회와 동일 마스크 추가 적합 1,660회에서 미해결 수치 실패는 없었다. 기본 지수는 저장값과 최대 0.0000131 차이였다. 이는 계산 검증이며 가정의 물리적 정당성을 보증하지 않는다.
2. 기존 tilt 곡선의 네 조건에서 수치 최적화 문제를 발견했다. 원래 결과와 동결 채점은 보존하고, 안정 우도로 수정한 목표에서도 개선이 유지되는지 별도로 확인했다.
3. 진폭의 중립점을 −1.0933으로 예측해 관측 −1.0881과 가까웠다. 반면 **둘째 되돌이 비율에서 관측되지 않은 중립점 +0.8325를 잘못 예측했다.** 주기성은 기존 목표에서는 조금 악화되고 안정 재적합 목표에서는 조금 개선됐다.
4. Gaussian 재생은 기본 H 판정 조건 밖의 특징도 만든다. 기대 비율은 주기성 [0.8,1] 밖 0.054%, 되돌이 비율 ≥0.5가 2.337%다. 물리 파형의 타당한 생성기로 해석하지 않는다. 이 위반이 중립점 실패의 원인이라고도 확정하지 않는다.
5. 그림의 음영은 난수 20회 평균의 Monte Carlo 표준오차다. 실제 물리 법칙의 신뢰구간이나 독립 세션 간 변동이 아니다.

**왜 최대 힘에는 같은 설명이 안 맞았나?** 경계 끝점이 바뀐 위치를 세었다. 진폭 문턱에서는 원래 띠가 있던 열×tilt의 아래 끝점 43.6%가 바뀌었지만 위 끝점은 5.9%뿐이었다. 주기성에서는 16.6%가 H 묶음을 통째로 잃었다. 같은 제거율이어도 어느 힘의 창을 없애는지가 다르다. 이는 경계 추출 단계의 영향 경로를 확인한 것이며, 최대 힘 지수의 부호까지 증명한 것은 아니다.

**새로움의 범위:** 공변량별 문턱 보정, 문턱에 의한 멱지수 왜곡, 구간 검열 우도 자체에는 선행연구가 있다. 우리가 확보한 기여 후보는 활–현 자료의 **특징–힘 결합 → K연속 판정 → 경계 지수 변화**를 수학 반례와 재생 모형으로 함께 정량화한 것이다. 새로운 물리 법칙이나 범용 ML 알고리즘으로 주장하지 않는다.

![특징 재생 모형과 실패 조건](../novelty/criterion_identifiability/response/response_curves.png)

상세 수치·코드·문헌·독립 검토: [종합 기록](../novelty/criterion_identifiability/README.md).

''' + report[b:]
a = report.index('현재 재현된 기여 후보는', report.index('## 11.'))
b = report.index('## 12. 재현 자료와 검증 상태', a)
report = report[:a] + '''현재 기여 후보는 두 축이다. 첫째는 측정 시각·힘 변화·기록 순서·세션 구조와 고정 예측의 실패를 드러낸 재현 가능한 감사다. 엄격한 최대 힘 법칙의 동결 확인 9회 중 4회가 통과했고 모든 채점은 이미 완료됐다. 둘째는 판정 기준의 영향을 특징–힘 결합과 이산 경계 추출로 분석하는 10.11절의 방법이다. 이 내부 설명 모형에서 42.3%의 오차 감소를 얻었지만, 새 현의 최소 힘을 단순 조회보다 잘 예측하는 방법은 아직 확보하지 못했다.

독립적인 현·세션 수가 작고 속도·날짜·현 물성이 교락한다. 라벨은 파형 대리 판정이며 힘 채널의 저하중 오차도 남는다. 중립적인 문턱이 물리적 정답을 준다는 보장은 없다. 수립/유지 차이, 문턱 보정, 검열 우도에는 선행연구가 있으므로 이 자체를 최초라고 부르지 않는다. 바이올린 실측으로 일반화한 결과도 없다.

영문 작업 원고는 `paper/draft_v2.md` v2.6이다. 제출 전에는 중심 기여를 하나로 정해 분량을 줄이고, 입력 특징에서 결과까지 재현 가능한 실행 묶음을 점검하며, 저자들이 주장·그림·출처를 검토해야 한다. 강한 방법론 주장을 더하려면 특정 법칙과 양립하는 데 필요한 판정 변경의 최소 비용을 정확히 계산·검증하는 후속 연구가 가능하다. 현재는 설계 검토만 했고 이를 구현한 성과로 세지 않는다. ICML 수준의 범용 기여나 논문 채택을 확정할 근거는 없다.

''' + report[b:]
report = replace_once(report, '## 12. 재현 자료와 검증 상태\n', '## 12. 재현 자료와 검증 상태\n\n- [10월 21일 수학·정량 재생·문헌 종합](../novelty/criterion_identifiability/README.md), [독립 코드 검토](../novelty/criterion_identifiability/review/response_review.md).\n')
report += '''
- Janes, H., & Pepe, M. S. (2009). 공변량별 문턱과 분류 정확도. Biometrika 96, 371–382. [DOI](https://doi.org/10.1093/biomet/asp002).
- Bryant, C., Osborne, J. A., & Shahmoradi, A. (2021). 검출 문턱의 오지정에 따른 인공적 멱지수. MNRAS 504, 4192–4202. [저자 원문](https://arxiv.org/html/2010.02935v1).
- Turnbull, B. W. (1976). The empirical distribution function with arbitrarily grouped, censored and truncated data. JRSS B 38, 290–295. [DOI](https://doi.org/10.1111/j.2517-6161.1976.tb01597.x).
'''
report_path.write_text(report)
print('Updated paper v2.6 and Korean report; original drafts retained in before_update/.')
