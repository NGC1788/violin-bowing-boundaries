"""Dated PDF editions after the high-force confirmation (Claude, 2026-10-19 22h). Reuses Astra's reportlab helpers in
astra_20261019/build_reports.py (unchanged); earlier PDFs are kept. Outputs:
  output/pdf/R&E_요약보고서_고하중반영_20261019.pdf  (brief, <= 4 pages, for non-specialists)
  output/pdf/R&E_상세보고서_고하중반영_20261019.pdf  (full report = final/최종보고서.md)"""
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer

BASE = Path.home() / "violin-rne"
sys.path.insert(0, str(BASE / "astra_20261019"))
import build_reports as BR  # noqa: E402

para, table, figure, OUT = BR.para, BR.table, BR.figure, BR.OUT
STAMP = "활-현 경계 연구 | 모든 동결 확인 채점 완료판 | 2026.10.20"


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("KR", 7.5); canvas.setFillColor(colors.HexColor("#667788"))
    canvas.drawString(46, A4[1] - 30, STAMP)
    canvas.drawRightString(A4[0] - 46, 25, str(doc.page))
    canvas.setStrokeColor(colors.HexColor("#d8e1e8")); canvas.line(46, 39, A4[0] - 46, 39)
    canvas.restoreState()


def document(path, story, title):
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=46, leftMargin=46, topMargin=47, bottomMargin=49,
                            title=title, author="김태건 · 정지윤 · 정지훈")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def brief():
    s = [
        para("활로 현을 그을 때, 소리가 나는 힘의 범위는 무엇이 정하나", "title"),
        para("공개 로봇 첼로 자료 약 10만 번의 긋기를 다시 분석하고, 예측을 미리 고정해 한 번만 채점했다", "lead"),
        para("강원과학고 R&E · 김태건 · 정지윤 · 정지훈 · 요약본 (2026년 10월 20일, 모든 동결 확인 채점 완료)", "small"),
        para("1. 무엇을 연구했나", "h2"),
        para("바이올린·첼로는 활이 현을 붙잡았다 놓기를 한 주기에 한 번씩 규칙적으로 반복할 때 깨끗한 소리가 난다. "
             "이 움직임을 헬름홀츠 운동이라고 한다. 활을 너무 약하게 누르면 소리가 들뜨고(최소 활 힘), 너무 세게 누르면 긁힌다(최대 활 힘). "
             "두 경계는 활 속도와 활이 닿는 위치(β)에 따라 달라진다. 50년 전 Schelleng의 이론 이후 실험과 시뮬레이션이 서로 다른 답을 내 왔다."),
        para("우리의 질문은 셋이다. 측정 방법이 경계를 얼마나 바꾸는가? 처음 보는 현·세션의 경계를 물리 법칙으로 예측할 수 있는가? "
             "그리고 그 답을 결과를 보기 전에 고정한 예측으로 확인할 수 있는가?"),
        para("2. 무엇을 썼나", "h2"),
        table([["자료", "내용"],
               ["개발 자료", "빈 음대(IWK)가 공개한 로봇 활 실험. 첼로 G현 4종 × 장력 2가지, 도표 11장, 66,000번 긋기"],
               ["확인 자료", "D_T2(11장에 포함)·A4·D2(각 6,000번), B_T2 두 번째 표본(6,400번, 0.3–7.5 N), 고하중 세션 4개(18,000번, 4–12 N). 예측을 먼저 고정하고 한 번만 채점"],
               ["판정", "브리지 힘 파형으로 긋기마다 헬름홀츠 운동인지 자동 판정(대리 판정, 소리를 직접 들은 정답은 아님)"],
               ["범위", "실측은 모두 첼로. 바이올린은 문헌과 시뮬레이션만. 장비 구입 없음"]]),
        Spacer(1, 8),
        para("3. 한눈에 보는 결과", "h2"),
        table([["질문", "결과", "증거 단계"],
               ["측정 절차가 결과를 바꾸나", "그렇다. 힘이 활 이동에 따라 줄고, 출발 가속이 기록 순서를 따르며, 속도마다 세션이 다르다", "사후 감사"],
               ["최대 활 힘을 미리 맞힐 수 있나", "8번 중 3번만 허용 범위 안. 4 N 이하 세션은 대체로 맞거나 약간 높게, 4–12 N 세션 4개는 모두 18–43% 낮게 예측. 힘 범위가 넓을수록 실제가 더 컸다(사후 경향)", "동결 확인"],
               ["최소 활 힘의 현 차이는 무엇과 연관되나", "세션의 힘 감소량과 강하게 연관(r 0.88). 그러나 한 도표(A1)에 크게 기대고 인과는 미확인", "사후"],
               ["새 현의 최소 힘을 더 잘 예측하는 모형", "없었다. 고정 확인(A4·D2)에서도 '이미 잰 현 값을 그대로 쓰기'가 개별 경계에서 더 나았다", "개발·동결"]],
              widths=[BR.WIDTH * w for w in (0.27, 0.55, 0.18)]),
        PageBreak(),
        para("측정 절차가 경계를 바꾼다", "title"),
        para("이 장비는 활 높이로 힘을 정한다. 그래서 약한 힘에서는 활이 움직이는 동안 읽히는 힘이 줄어든다. A2 도표에서 약한 힘 긋기는 "
             "14 cm 이동하는 동안 0.17 N이 줄었다. A2에서 세 속도의 곡선은 '시간'이 아니라 '이동 거리'로 겹친다(A1에서는 약하게)."),
        figure(BASE / "astra_20261019/fig_protocol_audit.png", 300),
        para("그림 1. 기존 도표의 사후 감사. 세션의 힘 감소량을 쓰면 도표 전체 수준 예측은 좋아지지만, 개별 경계의 오차는 오히려 커진다.", "small"),
        para("새로 찾은 절차 문제: 출발 가속은 기록 순서를 따른다", "h3"),
        para("활이 출발할 때의 가속도는 힘이 아니라 긋기 번호 순서대로 커지도록 짜여 있었다. 대부분의 도표는 약한 힘부터 기록해서 약한 힘 긋기가 부드럽게 출발한다. "
             "그런데 A1 도표만 거꾸로 기록되어, A1의 약한 힘 긋기는 느린 속도에서 15배 거칠게 출발했다. A1은 최소 활 힘이 유난히 높은 도표이고, "
             "거친 출발에는 더 큰 힘이 필요하다는 기존 이론과 방향이 맞는다. 다만 나머지 도표는 출발 일정이 같아서 이것으로 설명되는 것은 A1 하나다."),
        table([["약한 힘 20개 긋기의 출발 가속", "0.05 m/s", "0.10 m/s", "0.20 m/s"],
               ["A2 등 7장 (오름차순 기록)", "0.038", "0.104", "0.417"],
               ["A1 (내림차순 기록)", "0.563", "0.563", "0.563"]]),
        PageBreak(),
        para("최대 활 힘: 8번의 고정 확인 중 3번만 통과", "title"),
        para("Schelleng 이론은 최대 활 힘이 현의 임피던스 Z × 활 속도 ÷ β에 비례한다고 말한다. 저장력 현으로 상수 하나를 맞춘 뒤, 처음 보는 고장력 D_T2 도표를 미리 예측했다. "
             "도표 전체 수준의 오차는 0.002(0.2%)로 허용 기준 0.12 안이었다. 그 뒤 D2(−0.13)와 B_T2 두 번째 표본(+0.13)은 통과, A4(−0.17)는 실패했다."),
        para("고하중 세션 4개(힘 4–12 N)는 4 N 상한에 가려 있던 최대 활 힘을 직접 볼 기회였다. 예측 10개를 처리 전에 고정하고 GitHub에도 기록한 뒤, 라벨 없이 채점 규칙을 고정하고 한 번 채점했다. "
             "결과는 실패였다. 모든 예측이 4현 모두에서 실제보다 낮았고(0.17–0.36 log), 미리 정한 허용 범위를 벗어났다. 마찰열 효과를 넣은 모형은 단순 Schelleng보다도 나빴다."),
        figure(BASE / "paper/fig_highforce.png", 190),
        para("그림 2. (a) 고하중 4현에서 고정 예측의 수준 오차(0보다 크면 실제가 더 큼). 회색 띠가 허용 범위. (b) β 지수. 실제 기울기(검은 점)는 모든 모형보다 얕다.", "small"),
        figure(BASE / "paper/fig5_frozen_upper_summary.png", 200),
        para("그림 3. 엄격한 cZv/β의 고정 확인 8번 전부. 파랑 통과, 빨강 실패. 세션의 힘 상한이 높을수록 실제가 예측보다 컸다(사후 경향, 고하중 4현은 같은 설계라 독립 표본이 적음).", "small"),
        para("현 사이 비교에서도 '장력이 높아 Z가 크면 최대 힘이 그만큼 커진다'(지수 1)는 가설은 A·B·C 세 쌍을 함께 볼 때 기각되었고(A 1.7, B −0.5, C 3.3), 나중에 더한 D쌍(0.2)도 1에서 벗어났다. "
             "이 자료에서는 Z가 장력에 정확히 비례하고 굽힘 강성(EI)과도 함께 움직여, 무엇이 원인인지 가를 수 없다."),
        PageBreak(),
        para("최소 활 힘, 예측, 그리고 무엇이 새로운가", "title"),
        para("최소 활 힘", "h2"),
        para("도표마다 최소 활 힘 수준이 달랐고, 그 차이는 그 세션에서 '약하게 누른 활이 5 cm 움직이는 동안 잃은 힘'과 강하게 연관됐다(r 0.88). "
             "하지만 A1을 빼면 0.77, A1·A2를 빼면 0.69로 약해지고, 같은 8장에서는 현의 굽힘 강성이 0.86으로 더 강했다. 힘 0.2 N 이하의 센서 오차를 감안하면 0.80, 0.70으로 준다. "
             "탐색한 기술자 전체를 고려한 보정 뒤에도 연관은 남지만(보정 p 0.01–0.03), '판독값이 기계적으로 줄었을 뿐'이라는 설명도 배제되지 않았다. 미리 고정한 확인에서는 D_T2가 '검증 불가', A4·D2에서는 절차 모형이 단순 조회보다 개별 경계에서 나빴다. 그래서 '최소 힘은 측정 절차가 정한다'고 말하지 않는다."),
        para("처음 보는 현의 예측", "h2"),
        para("물리 공식과 기계학습 모형, 그리고 세션 보정 변형 37가지를 비교해도 '같은 속도·비슷한 β에서 이미 잰 현의 경계를 그대로 쓰기'보다 확실히 나은 것이 없었다. "
             "새 경계 추출 방법(확률적 띠 모형) 하나를 미리 정한 기준으로 시험했지만 반복 측정 재현성이 나아지지 않아 그 방향은 접었다."),
        para("무엇이 새로운가", "h2"),
        para("1) 공개 데이터셋의 측정 절차(힘 감소, 기록 순서에 묶인 출발 가속, 세션 구조, 측정 시각)를 수치로 점검했다. 우리가 찾아본 문헌 안에서는 이런 정량 점검이 없었다. "
             "2) 경계 법칙을 결과를 보기 전에 고정해 한 번만 채점한 확인을 세션 9개(최대 힘 8번, 최소 힘 4번, 합동 가설 1번)에서 모두 공개했다. 한 상수로 맞춘 Schelleng 법칙이 새 세션의 수준을 약 ±20% 안에서만 맞추고, 넓은 힘 범위에서는 낮게 예측한다는 점은 새 정보다. "
             "3) 틀린 주장(속도 지수 +1.77, A_T2 인과 해석, 영점 표류 방향, 절단으로 얻은 −0.10 등)을 철회 목록에 남겼다. Z 비례, 수립/유지 차이, 열마찰처럼 이미 알려진 것은 새롭다고 하지 않는다."),
        para("한계와 다음 단계", "h2"),
        para("실측은 모두 첼로 G현, 한 로봇, 적은 수의 세션이다. 속도·세션·날짜·장력·기록 방향이 서로 겹친다. 판정은 파형 기반 대리 판정이다. "
             "고정 예측은 모두 채점했다(B_T2 두 번째 표본의 최소 힘 모형은 설계가 달라 적용 불가로 미리 정함). 사후 견고성 점검 3가지를 독립 재실행으로 확인했고, 영문 논문 원고(draft v2.2)에 반영했다."),
        para("근거 파일: paper/repro/frozen_upper_summary.json, highforce_confirm/RESULT.md, confirm_B_T2s2/, confirm_v3/, posthoc_hf/, robust_lower/, robust_upper/, D_T2_confirm_result.json, A4_root_audit.md, paper/repro/attack_scan_order.log, claude_extra/band_repeat/result.json. 자세한 내용은 상세보고서.", "small"),
    ]
    document(OUT / "R&E_요약보고서_완료판_20261020.pdf", s, "활-현 경계 연구 요약 (완료판)")


def detail():
    text = (BASE / "final/최종보고서.md").read_text()
    text = re.sub(r"\\\[.*?\\\]", "log F_d(v, β) = c + a log v + b log β + Δ_d", text, flags=re.S)
    story, rows, paragraph = [], [], []

    def flush():
        if paragraph:
            story.append(para(" ".join(paragraph))); paragraph.clear()
        if rows:
            data = [line.strip("|").split("|") for line in rows if not re.fullmatch(r"[|\- :]+", line)]
            story.append(table([[x.strip() for x in row] for row in data])); story.append(Spacer(1, 8)); rows.clear()

    for line in text.splitlines():
        if line.startswith("|"):
            if paragraph:
                story.append(para(" ".join(paragraph))); paragraph.clear()
            rows.append(line); continue
        if rows:
            flush()
        if not line.strip():
            flush(); continue
        m = re.match(r"^(#{1,3}) (.*)", line)
        if m:
            flush(); story.append(para(m[2], {"#": "title", "##": "h2", "###": "h3"}[m[1]])); continue
        if line.startswith("!["):
            flush(); path = re.search(r"\]\((.*)\)", line).group(1); story.append(figure(Path(path), 360)); continue
        if line.startswith("- "):
            flush(); story.append(para("• " + line[2:])); continue
        paragraph.append(line.strip())
    flush()
    document(OUT / "R&E_상세보고서_완료판_20261020.pdf", story, "활-현 경계 연구 상세 (완료판)")


if __name__ == "__main__":
    brief(); detail(); print("새 날짜판 두 개 생성; 기존 PDF 보존")
