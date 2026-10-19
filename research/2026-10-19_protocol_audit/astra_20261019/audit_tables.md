# 저장 결과에서 자동 생성한 비교표

오차는 도표별 중앙값의 평균이다. 적중은 검열 관측을 포함한 가용 관측 전체에 대한 점 예측의 포함 횟수다.
검열된 예측과 관측의 겹침은 점 적중에 넣지 않는다. 무한 관측구간의 포함은 정밀한 추정 성공을 뜻하지 않는다.

| 모델 | 아래 오차 (유한 열 N) | 위 오차 (유한 열 N) | 아래 점 적중 | 위 점 적중 |
|---|---:|---:|---:|---:|
| P-Sch-publishedZ | 0.6468 (376) | 0.1949 (259) | 173/600 (28.8%) | 329/554 (59.4%) |
| P-Beigi-publishedZ | 0.3063 (376) | 0.1949 (259) | 146/600 (24.3%) | 329/554 (59.4%) |
| P-free-publishedZ | 0.2679 (376) | 0.1552 (259) | 149/600 (24.8%) | 347/554 (62.6%) |
| Triv-5-finite | 0.2687 (376) | 0.1882 (259) | 136/600 (22.7%) | 80/554 (14.4%) |
| D-logit | 0.3067 (363) | 0.2580 (181) | 85/600 (14.2%) | 45/554 (8.1%) |
| D-gbm | 0.2878 (364) | 0.1692 (241) | 86/600 (14.3%) | 93/554 (16.8%) |

## 기준 모양까지 학습 자료에서 다시 추정한 검증

| 힘 축 / 제외 단위 | 모델 | 수준 MAE | 열 오차 |
|---|---|---:|---:|
| measured_leave_diagram_out | mean | 0.2134 | 0.2937 |
| measured_leave_diagram_out | dec5_pl | 0.1012 | 0.3162 |
| measured_leave_diagram_out | log_Z | 0.2379 | 0.3212 |
| measured_leave_diagram_out | log_EI | 0.2409 | 0.3068 |
| measured_leave_diagram_out | log_Z_EI | 0.2691 | 0.3299 |
| measured_leave_type_out | mean | 0.2210 | 0.3186 |
| measured_leave_type_out | dec5_pl | 0.1312 | 0.3541 |
| measured_leave_type_out | log_Z | 0.3425 | 0.4136 |
| measured_leave_type_out | log_EI | 0.3696 | 0.4150 |
| measured_leave_type_out | log_Z_EI | 0.4671 | 0.4988 |
| nominal_rank_leave_diagram_out | mean | 0.4332 | 0.3607 |
| nominal_rank_leave_diagram_out | dec5_pl | 0.3593 | 0.4064 |
| nominal_rank_leave_diagram_out | log_Z | 0.4599 | 0.4495 |
| nominal_rank_leave_diagram_out | log_EI | 0.4376 | 0.4610 |
| nominal_rank_leave_diagram_out | log_Z_EI | 0.4850 | 0.4868 |
| nominal_rank_leave_type_out | mean | 0.4510 | 0.3838 |
| nominal_rank_leave_type_out | dec5_pl | 0.4258 | 0.4007 |
| nominal_rank_leave_type_out | log_Z | 0.4918 | 0.5082 |
| nominal_rank_leave_type_out | log_EI | 0.6611 | 0.7292 |
| nominal_rank_leave_type_out | log_Z_EI | 0.7629 | 0.8061 |

## 시간창을 분리한 예측: 주 목표창 시작 0.25초

| 학습→평가 | 모델 | log-loss |
|---|---|---:|
| A1→A2 | fixed_persistence | 0.369818 |
| A1→A2 | markov | 0.299082 |
| A1→A2 | markov_speed | 0.298396 |
| A1→A2 | hgb_controls | 0.301767 |
| A1→A2 | logistic_snapshot | 0.142504 |
| A1→A2 | logistic_history | 0.140535 |
| A1→A2 | hgb_snapshot | 0.145788 |
| A1→A2 | hgb_history | 0.142597 |
| A1→A2 | hgb_history_two_heads | 0.153508 |
| A2→A1 | fixed_persistence | 0.238145 |
| A2→A1 | markov | 0.214825 |
| A2→A1 | markov_speed | 0.215837 |
| A2→A1 | hgb_controls | 0.289479 |
| A2→A1 | logistic_snapshot | 0.129726 |
| A2→A1 | logistic_history | 0.472223 |
| A2→A1 | hgb_snapshot | 0.113345 |
| A2→A1 | hgb_history | 0.104651 |
| A2→A1 | hgb_history_two_heads | 0.100005 |
