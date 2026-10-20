# 선행연구 반박 점검: 세션 간 변동 / 줄 안정화(settling)와 최소·최대 활 압력

조사일: 2026-10-20 (웹 검색 + 원문 PDF 직접 확인)

## 원문을 직접 읽은 자료
1. Lampis, Mayer, Chatziioannou, "Assessing playability limits of bowed-string transients using experimental measurements", Acta Acustica 8, 44 (2024). doi:10.1051/aacus/2024034. PDF: https://pub.mdw.ac.at/pubmdw/publication/9056290a-503f-411f-82c9-13d16ef757d5/pdf
   - 첼로 G줄 1개(Dominant)로 Guettler 도표를 6회 측정: 연속된 날짜에 순방향/역방향 4회 + 한 달 뒤 줄을 풀었다 다시 걸고 2회. 측정 전 "며칠 동안 줄을 장력에 적응시킴".
   - 6회 모두 연주 가능 영역의 크기·모양이 일정했고, 1 N 하한도 반복 재현됨. 영역 안의 점박이 패턴은 재현성이 낮음.
   - 목적 문장: "long-term stability and impact of string ageing on playability limits"
   - 한계: 과도음(Guettler) 도표이며 정상상태 Schelleng 도표는 아님. 줄 1개, 반복 6회. 정량적인 분산 지표 없이 눈으로 비교함.
2. Lampis, Mayer, Chatziioannou, "An experimental approach for comparing the influence of cello string type on bowed attack response", JASA Express Lett. 4, 113201 (2024). doi:10.1121/10.0034330. PDF: https://pub.mdw.ac.at/pubmdw/publication/50841b67-c7f9-43fa-b05d-8314c1edd2c1/pdf
   - 줄 종류 4개, 종류마다 Guettler 도표 6회(4회 연속 + 한 달 뒤 다시 걸고 2회). 반복 측정은 "string settling"과 과도음의 카오스적 거동에 대비하려는 설계.
   - "differences between string types appear more pronounced than the differences between repeated measurements on the same string" → 본 연구 (A)의 반대 방향 결과(과도음 기준).
   - 줄 D는 다시 건 뒤 연주 가능 영역이 크게 넓어짐. 오른쪽(가속도) 경계가 이동했고 굽힘강성 B가 29% 감소함(다른 줄은 약 5%).
   - 표 2에 반복 측정 편차(표준편차)를 줄 종류별로 정리함.
3. Schoonderwaldt, Guettler, Askenfelt, "An empirical investigation of bow-force limits in the Schelleng diagram", Acta Acust. united Acust. 94, 604–622 (2008). doi:10.3813/AAA.918070. PDF: https://knutsacoustics.com/files/schoonderwaldt-et-al.-empirical-schelleng.pdf
   - 3.6절 "Reproducibility": 같은 장치로 약 5개월 전에 측정한 Schelleng 도표와 비교함. **하한 활 압력이 힘 격자 한 칸(약 20%)만큼 낮아짐.** 상한은 기울기 −0.9 대 −0.5로 달라짐. 그럼에도 "재현성 수용 가능"이라고 결론냄.
   - 2.5절에서 줄과 활털의 노화(aging)를 변동 요인으로 언급만 하고 측정하지는 않음.
   - 세션 간 변동을 단일 비교 1건으로 보고한 것이며, 표본 분포나 줄 종류 차와의 비교는 없음.
4. Galluzzo, "On the playability of stringed instruments", PhD thesis, Univ. Cambridge (2003/2004). https://www.repository.cam.ac.uk/handle/1810/244810 (PDF 다운로드해서 확인함)
   - Guettler 도표를 β=0.08에서 12회 반복 측정(퍼스펙스 막대). 경계는 거의 고정되었고 세부 패턴은 카오스적임. 실제 활로 4회 반복해도 비슷함. 같은 세션, 같은 줄 기준이며 settling이나 새 줄은 다루지 않음.
5. Pàmies-Vilà, Mayer, Matusiak, Chatziioannou, Acta Acustica 8, 45 (2024). doi:10.1051/aacus/2024035. PDF: https://pub.mdw.ac.at/pubmdw/publication/f68cb429-1dc4-4052-88d4-3aed6405d1ec/pdf
   - 연주자 운궁을 로봇으로 재현한 연구. 운동학 재현성만 다루고 연주 가능 한계의 세션 간 변동은 다루지 않음. 관련성 낮음.

## 초록만 확인 (원문 접근 불가: pubs.aip.org Cloudflare 봇 검사)
6. Lampis, Chatziioannou, Scavone, "Experimental analysis of cello string types: influence on playability and tonal characteristics using Schelleng diagrams", POMA 58, 035013 (2025). doi:10.1121/2.0002111. https://pubs.aip.org/asa/poma/article/58/1/035013/3364284 (CC BY. OA PDF 링크는 있으나 403/봇 검사로 막힘)
   - 초록: 맞춤 제작한 G2 줄 8종, 정상상태 Schelleng 도표, 활 압력 한계 분석. "Minimal within-type variability was noted; however, tonal deviations due to string settling were substantial, underscoring the importance of play-in periods." 장력 그룹 간 연주성 차이는 작음.
   - Zenodo 데이터셋 https://zenodo.org/records/17749111: 같은 종류의 여러 샘플(배치 간 변동), 같은 조건에서 Schelleng 도표 2회 반복(장치 변동, Part 6).
   - 원문이 settling이 Fmin/Fmax 자체를 바꾸지 않았다고 정량적으로 말하는지는 확인하지 못함. 초록상 settling 효과는 '음색'에 대한 것임.

## 문헌을 찾지 못한 것
- Pickering의 JCAS 논문(Physical properties of violin strings 44:6-8, 1985; Elasticity 46:2-3, 1986; Transient response of certain violin strings 45:24-26, 1986; Problems in string making 2.3:1-4, 1993; Strings and metallurgy 3.4:24-29, 1997). 제목만 CAS 색인(https://ccrma.stanford.edu/marl/CASL/otor.html)에서 확인했고 원문은 접근하지 못함. 새 줄과 길든 줄의 최소 활 압력을 비교한 결과는 색인에서 보이지 않음.
- Allen, "On the aging of steel guitar strings", JCAS 26:27-28 (1976): 기타 줄의 음색 노화 연구이며 활 압력과는 무관(색인 atob.html).

## 결론
- (A) 부분 선점. 세션 간 반복성 측정 자체는 선행연구가 있음(Schoonderwaldt 2008의 5개월 비교에서 Fmin이 약 20% 이동, Galluzzo 12회 반복, Lampis 2024a/b의 한 달 뒤 재장착). Lampis 2025 POMA는 종류 내 변동이 작다고 보고함. 다만 "줄 종류 간 차이가 세션 간 변동 안에 들어간다"를 정상상태 Fmin/Fmax로 정량적으로 보인 연구는 찾지 못했고, Lampis JASA-EL 2024는 과도음 기준으로 반대 결론(종류 차 > 반복 차)을 냈음.
- (B) 부분 선점. settling이 연주 가능 영역에 영향을 줄 수 있다는 관찰은 있음(JASA-EL 2024에서 재장착 후 줄 D 영역 확대·B 29% 감소, POMA 2025 초록의 play-in 강조). 다만 새 줄과 길든 줄의 최소·최대 활 압력 변화를 통제된 시간 경과로 정량화한 연구는 찾지 못함.
