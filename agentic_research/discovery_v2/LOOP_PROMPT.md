/Users/jeongjihoon/violin-bowing-boundaries에서 agentic_research/AGENTS.md, docs/WORK_QUEUE.md, docs/EXECUTION_STATUS.md와 discovery_v2/NEXT_EXPLORATION.md, PATH_DISTRIBUTION_REVIEW.md, LOOP_BUDGET.json을 읽고 넓은 AI 연구 탐색의 다음 미완료 작업 한 단위를 수행한다. 사용자는 예상 밖 조합도 많이 시험하는 탐색을 새로 요청했다. 한 가설의 실패를 연구 전체의 종료로 확대하지 않는다. 실제 프로세스·파일 잠금·Git 상태·작업 선점을 먼저 확인해 다른 작업과 중복하지 않는다.

기존 query/forecast/projection/raw/correction 캠페인과 discovery_v1/v2는 완료·STOP으로 보존한다. 완료한 소스·입력·계약·점수·예측·가중치를 재동결·재실행·변경하지 않는다. 새 탐색은 별도 버전으로 한다. 원래 train34명만 개발에 사용하고 olddev/test정답은hash도열지않는다. 새 sourcefold도 이미 본 자료의 적응적 탐색이며 독립 확증이 아니다. fold 목표문턱·scaler·prior·연속목표스케일은innertrain만사용한다. 좋은 결과만 공개하거나 평가 목표를 바꿔 실패를 성공으로 만들지 않는다.

다음은 별도 discovery_v3_math의 작은 수치 프로토타입이다. 먼저 계획·오차·반박 조건·코드 시험과 별도 검토·동결을 갖추고 source라벨도 쓰지 않는 Gaussian경로→두표본분산 joint확률/gradient 검증을 수행한다. 실제 단위 cents/dBFS, 미래46frame, ddof1, unknownvoicingmask/guard와두채널의존을 맞춘다. 고전χ²CDF/gradient/lowrankpath/CDF표현 자체는 신규성이 아니다. 새 고유값/확률 solver를 계속 고쳐 완료된 projection감사를 되살리지 않는다. 이 후보가 맞지 않으면 직접 선행·반박 대조가 있는 다른 가설은행 축을 검토할 수 있다. 준비되지 않은 모형을 자동으로 학습하지 않는다.

예산은 LOOP_BUDGET.json을 따른다. 이미 완료한24학습 뒤 추가학습batch최대3개, 각최대12job·총1800초·작업180초, 수치감사최대2개·각600초다. 시도/실패도실행전에ledger에기록하고예산을늘리지않는다. 각학습batch는새가설/필수대조/누수검사/별도코드검토/실행계약동결후실행한다. 가중치와학습곡선을private로보존한다. 공개비게이트모델가중치의추가다운로드는합계1GiB이내이며유료API/추가결제/계정권한변경을하지않는다. Claude를필요로하면가용성은한번만확인하고429실패를보존하며추가결제없이Codex/manual로명시적으로전환할수있다.

코드·문서·집계는허용된research/agentic-forecast-20261030브랜치에공유할수있다. 원음/MAT·특징/정답·개인설문·행별/개인별예측/가중치·원시CLI로그/계정정보를공개하지않고main병합·원자료삭제·외부이메일을하지않는다. 실행근거·실패·결정과다음한작업을EXECUTION_STATUS.md/WORK_QUEUE.md에기록한다. 의미있는새결과·실패·필요조치가있을때만알리고변경없는깨우기는조용히마친다. 2026-11-06 00:00KST, ledger예산소진, 정당한다음가설부재또는사용자중단요청이면이자동화를일시중지하고현재성과·미해결을정리한다. 높은novelty나성능개선은보장하지않는다.
