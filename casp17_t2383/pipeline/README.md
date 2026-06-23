# CASP 리간드 선별 파이프라인 (계층형, 범용 자동화)

교수님 프레임워크(Step 0~4)를 따르는 **계층형**(포켓 먼저 → 포켓 내 리간드) 자동화.
`config` 파일에 **경로 4개 + task**만 주면 각 스텝이 자동 실행되고, **결과를 번호별 폴더에 저장**해 단계별로 열람할 수 있습니다.

## 흐름               출력 폴더                        내용
```
00_collect/           master_table.csv               (수집·랭킹)
01_protein_clusters/  protein_clusters.csv, reps/     (Step 0: 단백질 형태 클러스터)
02_pocket_candidates/ pocket_candidates.csv, members.csv  (Step 1: 리간드 센트로이드 클러스터 top10)
03_pocket_validation/ pocket_validation.csv, p2rank/, rescore/  (Step 2: p2rank+gnina 검증·압축)
04_ligand_clusters/   ligand_clusters.csv             (Step 3: 포켓별 리간드 SC-RMSD 재클러스터)
05_final/             selection_summary.csv, model_*.cif, SELECTION_RATIONALE.md  (Step 4)
06_validation/        contact_residues.csv, fragment_compare.csv, overlay_fragment.pdb
07_viz/               visualization.png, contact_fingerprint.png
08_casp_lg/           <TARGET>LG_model1~5.txt, _all_models.txt
```

## 프레임워크 반영
- **Step 0 단백질 클러스터링** (`protein_cluster.py`) — 모델별 다른 형태(ConfA/B 등)를 PCA로 라벨링·대표 선정. (이전엔 팀원 영역이라 없던 부분)
- **Step 1 포켓 후보** (`pocket_candidates.py`) — 리간드 **센트로이드** coarse 클러스터, 작아도 **top10 유지**.
- **Step 2 검증·압축** (`pocket_validate.py`) — 포켓별 p2rank + gnina → pass 포켓만.
- **Step 3 포켓별 리간드 클러스터** (`ligand_cluster.py`) — **SC-RMSD(대칭보정)** 로 포켓 안에서 방향 클러스터.
- **Step 4 최종 선정** (`final_select.py`) — gnina 에너지 + 합의 + iptm 종합. **task=P/PA로 가중치 분기**(PA면 affinity↑).

## 설치 (한 폴더에 함께)
`scripts_dir` 에 전부 둡니다(`import complex_io/geom` 위해 같은 폴더 필수).
**파일명 앞 번호 = 실행 순서**:
- 공용(번호 없음, import 대상): `complex_io.py`, `geom.py`
- 스텝(순서대로):
  `0_rank_poses.py` `1_protein_cluster.py` `2_pocket_candidates.py` `3_pocket_validate.py`
  `4_ligand_cluster.py` `5_final_select.py` `6_contact_residues.py` `7_fragment_compare.py`
  `8_visualize.py` `9_make_casp_lg.py`
- 드라이버: `run_pipeline.py`, `config/`
- `legacy/` : 단일레벨 구버전(`cluster_poses.py` 등) — 새 흐름 미사용

## 실행
```bash
cp config/TEMPLATE.conf config/<TARGET>.conf   # 경로 4개 + task 수정
source /gpfs/deepfold/casp/casp17-ligand/scripts/env_setup.sh
python3 run_pipeline.py config/<TARGET>.conf
```
- `--dry-run` 명령만 / `--from <step>` 중간부터 / `--only <step> --force` 한 스텝 / idempotent.

## 되돌아가기 (피드백 루프)
각 스텝이 결과를 폴더에 남기고, 마지막 REVIEW가 포켓후보·검증·최종을 요약합니다.
불만족 시(예: 포켓 통과 0개, 합의 약함) **`--from <step> --force`** 로 이전 단계부터 재실행:
```bash
# 예: 검증 기준이 빡빡해 통과 0개 → 기준 완화 후 검증부터
python3 run_pipeline.py config/<T>.conf --from pocket_validate --force
```

## 범용성 / 한계
- 체인 수·리간드 개수 무관(`complex_io`), SMILES 조성 매칭, SC-RMSD(`geom`).
- 한계: 동일 조성 리간드 다중 copy는 seqid 순서 고정; 이성질체 구분 X; SC-RMSD 대칭순열은
  order-match 가정(아니면 plain RMSD 폴백); AutoDock(박스 도킹)은 옵션 훅(현재 gnina가 1차 에너지);
  literature survey는 수동; 템플릿 검색은 로컬(서버 무인터넷).
