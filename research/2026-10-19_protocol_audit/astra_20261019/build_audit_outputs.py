"""Render the offline audit from saved results, without fitting any model."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    inputs = [HERE/'protocol_results/results.json', HERE/'nested_protocol_results.json',
              HERE/'compare_fixed_results/results.json',
              HERE.parent/'astra_20261018/forecast/results/results_paired_corrected.json']
    protocol, nested, comparison, forecast = map(read, inputs)
    digest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    models = ['P-Sch-publishedZ', 'P-Beigi-publishedZ', 'P-free-publishedZ',
              'Triv-5-finite', 'D-logit', 'D-gbm']
    aggregates = {}
    lines = ['# 저장 결과에서 자동 생성한 비교표', '',
             '오차는 도표별 중앙값의 평균이다. 적중은 검열 관측을 포함한 가용 관측 전체에 대한 점 예측의 포함 횟수다.',
             '검열된 예측과 관측의 겹침은 점 적중에 넣지 않는다. 무한 관측구간의 포함은 정밀한 추정 성공을 뜻하지 않는다.', '',
             '| 모델 | 아래 오차 (유한 열 N) | 위 오차 (유한 열 N) | 아래 점 적중 | 위 점 적중 |',
             '|---|---:|---:|---:|---:|']
    for model in models:
        aggregates[model] = {}
        cells = []
        hits = []
        for side in ('lower', 'upper'):
            rows = [v[model][side] for v in comparison['results'].values()]
            n = sum(r['n_finite_observation_and_point_prediction'] for r in rows)
            error = float(np.mean([r['median_abs_log_error_conditional'] for r in rows]))
            hit = sum(r['counts']['point_hits'] for r in rows)
            denom = sum(r['counts']['observations_available'] for r in rows)
            common = float(np.mean([r['common_point_median_abs_log_error'] for r in rows]))
            counts = {key: sum(r['counts'].get(key, 0) for r in rows)
                      for key in rows[0]['counts']}
            aggregates[model][side] = {'conditional_error': error, 'finite_n': n,
                'point_hits': hit, 'all_available_denominator': denom,
                'common_point_error': common, 'counts': counts}
            cells.append(f'{error:.4f} ({n})')
            hits.append(f'{hit}/{denom} ({hit/denom:.1%})')
        lines.append('| ' + ' | '.join([model] + cells + hits) + ' |')
    lines += ['', '## 기준 모양까지 학습 자료에서 다시 추정한 검증', '',
              '| 힘 축 / 제외 단위 | 모델 | 수준 MAE | 열 오차 |', '|---|---|---:|---:|']
    for variant, result in nested['variants'].items():
        for model, r in result['models'].items():
            lines.append(f"| {variant} | {model} | {r['level_mae']:.4f} | {r['diagram_mean_column_median_error']:.4f} |")
    lines += ['', '## 시간창을 분리한 예측: 주 목표창 시작 0.25초', '',
              '| 학습→평가 | 모델 | log-loss |', '|---|---|---:|']
    for fit in forecast['fits']:
        if fit['target_start'] == .25:
            lines.append(f"| {fit['train']}→{fit['test']} | {fit['model']} | {fit['metrics']['log_loss']:.6f} |")
    (HERE/'audit_tables.md').write_text('\n'.join(lines)+'\n')
    summary = {'scope': '기존 자료 사후 감사. D_T2 미열람. 새로운 적합 없음.',
               'source_sha256': digest, 'comparison': aggregates,
               'protocol_sensitivity': {k: {'r': v['association']['r'],
                   'fixed_reference_lodo_mae': v['association']['cv']['leave_diagram_out']['dec5_pl']['mae'],
                   'fixed_reference_type_holdout_mae': v['association']['cv']['leave_type_out']['dec5_pl']['mae']}
                   for k, v in protocol['variants'].items()}}
    (HERE/'audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

    plt.rcParams.update({'font.family': 'Apple SD Gothic Neo', 'axes.unicode_minus': False,
                         'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    a = protocol['variants']['measured_K2_t0.2']
    tags = protocol['tags']; x = np.array(protocol['descriptor'])
    y = np.array([a['levels'][t] for t in tags])
    colors = {'A': '#2563a0', 'B': '#d38325', 'C': '#278270', 'D': '#86589a'}
    ax = axes[0, 0]
    for typ in colors:
        mask = np.array([t[0] == typ for t in tags])
        ax.scatter(x[mask], y[mask], c=colors[typ], s=45, label=typ, zorder=3)
    offsets = {'B_T2r1': (6, 13), 'B_T2r2': (6, -15), 'A2': (-25, -17), 'B_T2s1': (-15, 10)}
    for tag, xx, yy in zip(tags, x, y):
        ax.annotate(tag, (xx, yy), xytext=offsets.get(tag, (6, 7)), textcoords='offset points', fontsize=9)
    xx = np.linspace(x.min(), x.max(), 80)
    ax.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), color='#777777', ls='--', lw=1)
    ax.set(xlabel='낮은 힘에서 읽힌 힘 변화 dec5_pl (N; 음수 = 감소)',
           ylabel='아래 경계 수준 (log; A1+A2 기준 모양 대비)',
           title=f'A. 도표 수준의 연관은 재현됨 (r = {a["association"]["r"]:.3f})')
    ax.legend(title='현 종류', ncol=4, loc='lower right', frameon=False, fontsize=9)
    for col, (key, title) in enumerate([
        ('level_mae', 'B. 도표 수준 오차: 힘 변화 기술자가 개선'),
        ('diagram_mean_column_median_error', 'C. 유한 구간의 열 오차: 힘 변화 기술자가 악화')]):
        ax = axes[0, 1] if col == 0 else axes[1, 0]
        groups = ['leave_diagram_out', 'leave_type_out']
        positions = np.arange(2)
        for shift, model, label, color in [(-.18, 'mean', '평균 수준만', '#a3a9b2'),
                                          (.18, 'dec5_pl', '힘 변화 기술자', '#2563a0')]:
            values = [nested['variants']['measured_'+g]['models'][model][key] for g in groups]
            bars = ax.bar(positions+shift, values, width=.34, label=label, color=color)
            ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=10)
        ax.set(xticks=positions, xticklabels=['도표 하나 제외', '현 종류 전체 제외'],
               ylabel='절대 로그 오차 (작을수록 좋음)', title=title, ylim=(0, .41))
        ax.legend(frameon=False, loc='upper left', fontsize=10)
    ax = axes[1, 1]
    variants = [('measured_K2_t0.2', '측정 힘'), ('nominal_rank_K2_t0.2', '공칭 힘')]
    for j, (k, label) in enumerate(variants):
        v = protocol['variants'][k]['association']['cv']
        vals = [v[g]['dec5_pl']['cv_r2'] for g in ('leave_diagram_out', 'leave_type_out')]
        bars = ax.bar(np.arange(2)+(j-.5)*.36, vals, width=.34,
                      label=label, color=['#2563a0', '#d38325'][j])
        ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
    ax.axhline(0, color='#777777', lw=.8)
    ax.set(xticks=np.arange(2), xticklabels=['도표 하나 제외', '현 종류 전체 제외'],
           ylabel='회귀 단계의 교차검증 R²', title='D. 공칭 힘 축에서는 예측력이 약해짐', ylim=(-.42, .92))
    ax.legend(frameon=False, fontsize=10)
    fig.suptitle('힘 변화와 경계의 연관 ≠ 개별 경계 예측의 개선', fontsize=18, weight='bold')
    fig.supxlabel('첼로 G현 10개 도표 · t=0.2초, K=2 · B/C는 기준 모양까지 학습 자료만으로 재적합 · A/D는 고정 기준 모양\n사후 분석 · 도표와 현 종류는 독립 표본 수와 같지 않음 · 검열 포함 구간 거리는 별도 표에 보고', fontsize=10)
    fig.savefig(HERE/'fig_protocol_audit.png', dpi=170)
    fig.savefig(HERE/'fig_protocol_audit.pdf')
    plt.close(fig)
    print('감사 표·출처 해시·그림 생성 완료. 새 모델 적합과 D_T2 접근 없음.')


if __name__ == '__main__':
    main()
