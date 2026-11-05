# improved2/tools — 분석 전용

정책 파일이 아니다. `tr.py`, `det.py`, `noise.py`, `ptest.py`, `prune.py`, `passr.py`, `alive.py`는 **정답 라벨 전체**를 읽어
추적·사후 가지치기·잡음 통계를 낸다(분석 전용, 정책의 질의 선택에는 쓰지 않음). 첫 인자는 번들 폴더
(`colab/bundle_noeps` 또는 `colab/bundle_robust`)이고, `tools/` 안에서 실행한다.

- `ev2.sh NAME 정책.py`: 두 벤치마크 search 채점(봉인 evaluator 그대로), `res/`에 원자료 저장.
- `cmp.py A B`: 두 채점의 사례별 차이(`res/` 기준). `mkres.py OUT 이름...`: `results.tsv` 생성.
- `tag.py IN OUT`: 정책 사본에 분기 태그를 붙임(분석용 사본만, 원 정책은 그대로).
- `tr.py BUNDLE 정책 [사례]`: 분기별 읽기 수(`V=1`이면 읽기마다 출력). `det.py BUNDLE 정책`: 단조성 위반을 처음 본 읽기 번호.
- `prune.py BUNDLE 정책`: 비양립 사례의 사후 가지치기 크기(양립 사례는 조건을 바꿔 같은 방식으로 셈).
- `noise.py BUNDLE`, `ptest.py BUNDLE`: 라벨 구조(양성 위 0의 위치, 검사 읽기 한 번의 위반 확률).
- `passr.py`, `alive.py`: 양립 사례에서 실제로 통과하는 규칙과 살아 있는 규칙의 시간 변화.
