"""Summary figure: establishment (E) vs maintenance (M) speed exponents of F_min across friction models and
attack accelerations, from the sim_contact2 logs (Claude, 2026-10-18). Output: fig_em_summary.png"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
font_manager.fontManager.addfont("/System/Library/Fonts/AppleSDGothicNeo.ttc")
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

PAT = re.compile(r"\[(E|M) [^\]]*\][^\n]*?v\^([+-][\d.]+)±[\d.]+[^\n]*?95% \[([+-][\d.]+), ([+-][\d.]+)\]")


def runs(path, label_of):
    text = path.read_text()
    out = []
    for block in re.split(r"(?=^rate )", text, flags=re.M):
        if not block.startswith("rate"):
            continue
        hits = {m.group(1): (float(m.group(2)), float(m.group(3)), float(m.group(4))) for m in PAT.finditer(block)}
        hyst = re.search(r"유지에서만 헬름홀츠 (\d+), 수립에서만 (\d+)", block)
        if "E" in hits and "M" in hits:
            out.append((label_of(block.splitlines()[0]), hits, hyst.groups() if hyst else None))
    return out


def friction_label(head):
    curve = re.search(r"곡선 (\w+)", head)
    curve = curve.group(1) if curve else "hyperbolic"
    ramp = re.search(r"ramp ([\d.]+) m/s²", head)
    fr = "Mores 문헌값" if "Mores" in head else re.search(r"마찰 (\{[^}]*\})", head).group(1)
    fr = fr.replace("'", "").replace("v0: ", "v0=").replace("mu_s: ", "μs=").replace("mu_d: ", "μd=")
    name = {"hyperbolic": "쌍곡선", "gauss": "가우시안", "ep": "탄소성(Matusiak)", "thermal": "열-탄소성(van Walstijn)"}[curve]
    if curve == "thermal":
        fr = "논문 Table 1"
    dist = re.search(r"distance ([\d.]+) m", head)
    tail = f" · 가속 {ramp.group(1)}" if ramp else (" · 출발거리 4.8 cm" if dist else " · 램프 0.2 s")
    return f"{name} {fr}{tail}" + (" · 바이올린 D" if "violinD" in head else "")


def main():
    rows = []
    for log in ("sim_contact2_boot.log", "sim_contact2_matusiak.log", "sim_contact2_gauss.log", "sim_contact2_ep.log",
                "sim_contact2_thermal.log", "sim_contact2_accsweep.log", "sim_contact2_rampdist.log",
                "sim_contact2_ep_violinD.log", "sim_contact2_hyp_violinD.log"):
        p = HERE / log
        if p.exists():
            rows += runs(p, (lambda h, log=log: friction_label(h.replace('· 마찰', '· 곡선 gauss · 마찰') if log == 'sim_contact2_gauss.log' and '곡선' not in h else h)))
    seen, uniq = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0]); uniq.append(r)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(uniq) + 1.6))
    for i, (label, h, hyst) in enumerate(uniq):
        y = len(uniq) - i
        for key, color, dy in (("E", "#1f4e9c", 0.12), ("M", "#c0392b", -0.12)):
            est, lo, hi = h[key]
            ax.plot([lo, hi], [y + dy, y + dy], color=color, lw=2)
            ax.plot(est, y + dy, "o", color=color, ms=5)
        ax.text(3.05, y, f"{hyst[0]}/{hyst[1]}" if hyst else "", va="center", fontsize=8, color="#555")
    ax.set_yticks(range(len(uniq), 0, -1), [r[0] for r in uniq], fontsize=8)
    ax.axvline(0, color="#888", lw=0.8, ls="--"); ax.axvline(1, color="#888", lw=0.8, ls=":")
    ax.axvspan(-0.28, 0.09, color="#c0392b", alpha=0.08)
    ax.set_xlim(-0.6, 3.4)
    ax.set_xlabel("F_min 속도 지수 (95% 부트스트랩)   파랑 = 수립(정지에서 출발) · 빨강 = 유지(확립 후 힘 낮춤)\n붉은 띠 = Schoonderwaldt 2008 재분석 [-0.28, +0.09] · 점선 1 = IWK 같은 정착시각 ≈ +1 \n오른쪽 숫자 = 유지에서만/수립에서만 헬름홀츠인 칸 수", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig_em_summary.png", dpi=170)
    print(f"{len(uniq)} rows -> fig_em_summary.png")


if __name__ == "__main__":
    main()
