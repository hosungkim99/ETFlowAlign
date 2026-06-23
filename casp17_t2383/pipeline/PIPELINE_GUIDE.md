# CASP 리간드 파이프라인 — 전체 정리 (순서·스크립트·입출력·해석)

실행 순서(=드라이버가 호출하는 순서)대로 정리. 괄호는 보강단계(①②③).
공용 라이브러리 `complex_io.py`(파서)·`geom.py`(정렬/SC-RMSD)는 스텝이 아니라 모든 스텝이 import.

---

## 한눈에 보기

| # | 스텝(env) | 스크립트 | 한 줄 역할 | 주요 output |
|---|-----------|----------|-----------|-------------|
| 0 | collect(stdlib) | 0_rank_poses.py | 4000개 결과 수집·랭킹 | 00_collect/master_table.csv |
| 1 | protein_cluster(sci) | 1_protein_cluster.py | 단백질 형태 클러스터(Step0) | 01_protein_clusters/ |
| 2 | pocket_candidates(sci) | 2_pocket_candidates.py | 포켓 후보 top10(Step1) | 02_pocket_candidates/ |
| 3 | pocket_validate(sci) | 3_pocket_validate.py | p2rank+gnina 검증(Step2) | 03_pocket_validation/ |
| ①a | prep_validity(sci) | 4b_prep_validity.py | PoseBusters 입력 준비 | 04b_posebusters/inputs/ |
| ①b | posebusters(casp_eval) | 4c_posebusters.py | 물리 유효성 검사 | 04b_posebusters/posebusters.csv |
| 4 | ligand_cluster(sci) | 4_ligand_cluster.py | 포켓별 리간드 SC-RMSD 클러스터(Step3) | 04_ligand_clusters/ |
| 5 | final_select(sci) | 5_final_select.py | 종합점수 최종 5개(Step4) | 05_final/ |
| ② | refine(sci) | 5b_refine.py | 포켓 내 gnina 정제 | 05b_refined/ |
| 6 | contact_residues(sci) | 6_contact_residues.py | 결합 잔기 빈도 | 06_validation/contact_residues.csv |
| - | fragment_compare(sci) | 7_fragment_compare.py | 실험 fragment 비교 | 06_validation/fragment_compare.csv |
| 7 | visualize(sci) | 8_visualize.py | 종합 시각화 | 07_viz/*.png |
| 8 | casp_lg(sci) | 9_make_casp_lg.py | CASP LG 포맷 변환 | 08_casp_lg/*.txt |
| ③ | confidence(sci) | 5c_confidence.py | 선정 신뢰도 판정 | 05c_confidence/ |

(sci = boltz2 env python / casp_eval = posebusters env)

---

## 스텝별 상세 (입력 → 출력 → 의미)

### 0. collect — `0_rank_poses.py`
- **입력**: 팀 cofolding 결과 폴더(`<model>/seed_*/sample_*/summary.json`+model.cif)
- **출력**: `master_table.csv` (rank, model, seed, sample, iptm, ligand_iptm, cif…)
- **의미**: 흩어진 4000개 예측을 한 표로 모아 interface 신뢰도순 정렬. 이후 모든 스텝의 입력.

### 1. protein_cluster (Step 0) — `1_protein_cluster.py`
- **입력**: master_table.csv
- **출력**: `protein_clusters.csv`(구조→conf_id), `reps/conf_*.cif`
- **의미**: 모델마다 다른 단백질 형태(ConfA/B 등)를 Cα PCA로 라벨링. *주의: 현재 임계값에서 과분할(라벨만 noisy), 포켓 결과엔 영향 없음.*

### 2. pocket_candidates (Step 1) — `2_pocket_candidates.py`
- **입력**: master_table.csv (+protein_clusters)
- **출력**: `pocket_candidates.csv`, `members.csv`
- **의미**: 리간드 **무게중심**을 정렬좌표계에서 coarse 클러스터 → **바인딩 포켓 후보 top10**. 작아도 유지.

### 3. pocket_validate (Step 2) — `3_pocket_validate.py`
- **입력**: pocket_candidates.csv
- **출력**: `pocket_validation.csv`(+p2rank/, rescore/)
- **의미**: 포켓마다 **p2rank**(캐비티 존재?) + **gnina**(에너지) → 기준 통과(pass) 포켓만 압축.

### ①a/①b. PoseBusters — `4b_prep_validity.py`(준비) → `4c_posebusters.py`(검사)
- **입력**: ligand_clusters 대표 + 단백질PDB/리간드SDF
- **출력**: `posebusters.csv` (valid / 실패검사)
- **의미**: 물리적으로 깨진 포즈(단백질 충돌·이상 기하) 표시 → final_select가 제외.

### 4. ligand_cluster (Step 3) — `4_ligand_cluster.py`
- **입력**: members + pocket_validation(통과포켓) + ligand-tsv(SC-RMSD 대칭)
- **출력**: `ligand_clusters.csv` (포켓별 방향 클러스터 대표)
- **의미**: 통과 포켓 안에서 **SC-RMSD(대칭보정)** 로 리간드 방향을 재클러스터 → 포켓별 대표 후보.

### 5. final_select (Step 4) — `5_final_select.py`
- **입력**: ligand_clusters + pocket_validation + posebusters
- **출력**: `05_final/{selection_summary.csv, model_*.cif, SELECTION_RATIONALE.md}`
- **의미**: 무효 제외 후 **종합점수 = consensus·iptm·affinity(task 가중)** 로 최종 ≤5개. task=PA면 affinity 비중↑.

### ②. refine — `5b_refine.py`
- **입력**: 05_final/model_*.cif + ligand-tsv
- **출력**: `05b_refined/{model_*.cif, refine_summary.csv, selection_summary.csv}`
- **의미**: gnina `--minimize`로 포켓 내 **국소 정제**. drift>임계면 원본 유지(엉뚱한 이동 방지). casp_lg가 정제본 사용.

### 6. contact_residues — `6_contact_residues.py`
- **입력**: master_table (1순위 포켓 멤버)
- **출력**: `contact_residues.csv` (잔기별 접촉빈도)
- **의미**: 1순위 포켓에서 **리간드가 실제로 닿는 잔기**를 빈도로. = 우리가 정의한 결합부위.

### fragment_compare — `7_fragment_compare.py` (experimental_cif 있을 때만)
- **입력**: 05_final/model_1.cif + 실험 cif + contact_residues.csv
- **출력**: `fragment_compare.csv`, `overlay_fragment.pdb`
- **의미**: 실험 구조를 정렬(번호 offset 자동) → 실험 리간드 접촉잔기 ↔ 예측 접촉잔기 **공유 개수**. 실험적 검증.

### 7. visualize — `8_visualize.py`
- **입력**: master_table + contact_residues (+fragment)
- **출력**: `visualization.png`(4패널), `contact_fingerprint.png`
- **의미**: 클러스터 분포·모델 신뢰도·접촉지문을 한 눈에.

### 8. casp_lg — `9_make_casp_lg.py`
- **입력**: (정제) selection_summary + ligand-tsv
- **출력**: `08_casp_lg/<T>LG_model1~5.txt`
- **의미**: 제출 5개를 CASP LG 포맷(수용체 PDB + 리간드 mol block)으로 변환.

### ③. confidence — `5c_confidence.py`
- **입력**: 위 출력들 집계
- **출력**: `05c_confidence/{confidence_report.md, confidence.csv}`
- **의미**: "이번 자동선택을 믿어도 되나"를 **HIGH/MEDIUM/LOW**로 판정.

---

## 해석이 필요한 output — 해석 방법론

### A. `07_viz/visualization.png` (4패널)
- **(1) Ligand PCA by model**: 점=구조, 색=모델. **여러 모델이 같은 덩어리에 겹치면 합의**. 흩어진 색=모델별 다른 자리.
- **(2) Ligand PCA by cluster**: 가장 큰 색 덩어리 = 1순위 포켓. 작은 덩어리들=부차 포켓.
- **(3) iptm 분포 (boxplot)**: **모델 간 raw iptm 직접 비교 금지**의 근거. af3 높고 촘촘/pt2 넓음/bt2·of3 낮음 → 그래서 선정은 raw iptm 아닌 **consensus**로.
- **(4) 상위 클러스터 모델 구성 (stacked bar)**: 한 막대에 **여러 색=cross-model 합의(신뢰↑)**, 단색=단일모델(의심). → 1순위 막대가 다색이어야 좋음.
- **판단 요령**: (1)(2)에서 큰 단일 덩어리 + (4)에서 그게 다모델 = "강한 합의 포켓".

### B. `07_viz/contact_fingerprint.png`
- 막대 = 잔기별 리간드 접촉빈도(%). **높고 고른 막대 다수 = 결합부위가 한 곳으로 수렴**(좋음). 낮고 흩어지면 불확실.
- **빨강 막대 = 실험 fragment와 공유 잔기**(experimental_cif 줬을 때). 빨강이 많을수록 실험적 뒷받침↑.
- **판단 요령**: ~100% 잔기가 5개 이상 + 그중 빨강 존재 = 결합부위 확정적.

### C. `05c_confidence/confidence_report.md` ⭐ 가장 먼저 볼 것
- **판정 줄**: HIGH=그대로 제출 가능 / MEDIUM=수동 검토 / LOW=사람 개입.
- 근거 신호별 읽기:
  - 우세도 ≥30% & 다모델 합의 & af3 포함 → 사이트 확실.
  - p2rank 거리 ≤6Å & posebusters valid → 물리 타당.
  - fragment 공유 ≥3 → 실험 일치.
  - refine drift 작음(<1Å) → 정제 안정.
- **판단 요령**: 판정이 MEDIUM/LOW면 어느 신호가 약한지 보고 그 단계로 `--from` 복귀.

### D. `02_pocket_candidates.csv` / `03_pocket_validation.csv`
- **size**=그 포켓에 모인 구조 수(클수록 합의↑), **n_models**=참여 모델 수(**≥2 & af3 포함이 핵심**), **models**=구성.
- pocket_validation의 **p2rank_dist**(작을수록 실제 캐비티), **pass**.
- **판단 요령**: 1순위 포켓이 size 크고 n_models≥2(af3 포함)면 신뢰. 단일모델 단독이면 의심.
- *conf_composition 컬럼은 현재 noisy(과분할) → 무시.*

### E. `04_ligand_clusters.csv`
- 포켓별 리간드 방향 클러스터. **1순위 클러스터가 포켓의 큰 비중(수렴)** 이면 방향까지 확실.

### F. `05(b)/selection_summary.csv`
- **composite**=종합점수(높을수록↑), **pocket_pass**, **posebusters_valid**(반드시 True/NA).
- **판단 요령**: model_1 composite가 2~5위보다 확연히 높으면 1순위가 압도적(좋음). 사이트가 섞이면(다른 pocket_id) hedge 의미.

### G. `06_validation/fragment_compare.csv`
- exp_ligand별 **n_shared**(예측과 공유 잔기 수). **진짜 fragment(예: VVP)가 크고, 버퍼(EDO/TLA/UNX)는 0**이어야 정상.
- **판단 요령**: 진짜 fragment n_shared ≥3 = 실험적으로 같은 자리 = 강한 검증.

### H. `05b_refined/refine_summary.csv`
- **drift**(원본 대비 이동 Å; 작을수록 위치 유지), **used**(refined/orig). **drift<1 & used=refined = 정제 성공**. drift 크면 원본 유지(안전).

### I. `04b_posebusters/posebusters.csv`
- **valid**(True/False), **failed**(실패 검사명). `minimum_distance_to_protein`=단백질 충돌, `volume_overlap_with_protein`=겹침.
- **판단 요령**: 최종 5개가 모두 valid면 통과. 무효가 많으면 그 포켓/포즈 품질 의심.

### J. `06_validation/overlay_fragment.pdb`
- PyMOL로 열면 model_1(예측) + 정렬된 실험 fragment(체인 Z)가 겹쳐 보임. **육안으로 같은 포켓에 겹치는지** 확인.

---

## 결과를 믿을지 한 줄 요약
**confidence_report.md = HIGH** + **1순위 포켓 다모델(af3 포함) 합의** + **최종 5개 posebusters valid** + **fragment n_shared ≥3** → 네 가지 충족이면 결과를 신뢰하고 제출.
