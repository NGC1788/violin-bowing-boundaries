"""Figures for REPORT_KO_v2.md. Every number below is copied from a repo file (source in
the comment next to it); nothing is computed from the holdouts.

    pip install matplotlib koreanize-matplotlib
    python agentic_research/evolve_v4/figs/make_figs.py

Writes fig1_violin_search.png and fig2_ai_audit_search.png next to this file.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

try:  # Korean glyphs (NanumGothic); falls back to the default font if missing
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pass

HERE = Path(__file__).resolve().parent

# colours: neutral = untuned reference, orange = fair-tuned general baseline,
# blue = evolved / improved policy (dataviz reference palette, slots 1-2)
REF, BASE, POLICY = "#8c8b86", "#eb6834", "#2a78d6"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

# (label, reads on search split, kind, dev-set flag)
VIOLIN = [  # violin search, 54 cases, read-everything = 79,400 (figs/split_size.py)
    ("실무식 열별 이분 (원본)", 12556, REF, False),       # BASELINES_FAIR.md
    ("seed (공개 1단계 예측)", 9158, REF, False),          # DESIGN.md §7, BASELINES_FAIR.md
    ("DRD/EC2 (조정)", 8945, BASE, False),                 # BASELINES_FAIR.md
    ("실무식 열별 이분 (조정)", 5550, BASE, False),        # BASELINES_FAIR.md
    ("증명수 탐색 (조정)", 5386, BASE, False),             # BASELINES_FAIR.md
    ("적응형 탐욕 SBFE (조정)", 4513, BASE, False),        # BASELINES_FAIR.md
    ("distilled_a", 3453, POLICY, False),                  # BASELINES_FAIR.md
    ("t21 (진화 최고)", 3072, POLICY, False),              # DESIGN.md 2026-11-05 00시
    ("general_policy", 2349, POLICY, False),               # improved/IMPROVE_LOG.md
    ("best_policy", 2238, POLICY, False),                  # improved/IMPROVE_LOG.md
    ("general_v2", 2232, POLICY, False),                   # improved2/LOG.md
]
VIOLIN_HINDSIGHT = 1554  # prune_search.json: seed reads pruned with all answers known, 524 + 1030

AUDIT = [  # AI-audit search, 40 cases, read-everything = 48,000 (transfer_robust/RESULTS.md)
    ("seed (공개 1단계 예측)", 11182, REF, False),         # transfer_robust/RESULTS.md
    ("DRD/EC2 (조정)", 10022, BASE, False),                # transfer_robust/RESULTS.md 추가
    ("실무식 열별 이분 (원본)", 9914, REF, False),         # transfer_robust/RESULTS.md
    ("증명수 탐색 (조정)", 9340, BASE, False),             # transfer_robust/RESULTS.md 추가
    ("적응형 탐욕 SBFE (조정)", 8285, BASE, False),        # transfer_robust/RESULTS.md 추가
    ("best_policy", 6491, POLICY, False),                  # transfer_robust/RESULTS.md 추가
    ("실무식 열별 이분 (조정)", 4778, BASE, False),        # transfer_robust/RESULTS.md 추가
    ("t21 (진화 최고)", 3693, POLICY, False),              # transfer_robust/RESULTS.md
    ("distilled_a", 3560, POLICY, False),                  # transfer_robust/RESULTS.md
    ("general_policy *", 3078, POLICY, True),              # improved/IMPROVE_LOG.md (dev set)
    ("general_v2 *", 2688, POLICY, True),                  # improved2/LOG.md (dev set)
]
AUDIT_HINDSIGHT = 1842  # transfer_robust/RESULTS.md: seed reads pruned with all answers known


def bars(rows, hindsight, title, note, out):
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    ys = list(range(len(rows)))[::-1]
    for y, (_, v, colour, dev) in zip(ys, rows):
        ax.barh(y, v, height=0.62, color=colour, edgecolor=SURFACE, linewidth=2,
                hatch="//" if dev else None)
        ax.text(v + max(values) * 0.008, y, f"{v:,}", va="center", ha="left",
                fontsize=9, color=INK)
    ax.axvline(hindsight, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(hindsight, -0.75, f" 사후 가지치기 {hindsight:,} (정답을 다 알 때만 가능, 정책 아님)",
            va="center", ha="left", fontsize=8, color=MUTED)
    ax.set_ylim(-1.2, len(rows) - 0.4)
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel("search 분할에서 판정까지 읽은 파형 수 (적을수록 좋음)", fontsize=9, color=MUTED)
    ax.set_xlim(0, max(values) * 1.13)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.grid(axis="x", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REF, label="원본·공개 정책(조정 안 함)"),
        plt.Rectangle((0, 0), 1, 1, color=BASE, label="일반 기준선(채점 40회 이내 조정)"),
        plt.Rectangle((0, 0), 1, 1, color=POLICY, label="진화·개선 정책"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(HERE / out, facecolor=SURFACE)
    plt.close(fig)


def main():
    bars(VIOLIN, VIOLIN_HINDSIGHT,
         "그림 1. 바이올린 과제 search 54사례 (전부 읽기 = 79,400)",
         "출처: BASELINES_FAIR.md, DESIGN.md, improved/IMPROVE_LOG.md, improved2/LOG.md, prune_search.json",
         "fig1_violin_search.png")
    bars(AUDIT, AUDIT_HINDSIGHT,
         "그림 2. AI 감사 과제 search 40사례 (전부 읽기 = 48,000)",
         "* 빗금 = 이 search 분할을 개발 세트로 써서 만든 정책(처음 보는 데이터 성적 아님). "
         "조정 기준선은 바이올린 search로만 조정. 출처: transfer_robust/RESULTS.md, improved*/",
         "fig2_ai_audit_search.png")


if __name__ == "__main__":
    main()
