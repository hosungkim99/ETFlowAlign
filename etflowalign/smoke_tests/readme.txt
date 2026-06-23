smoke_tests/
├── 00_inputs/                  # 모든 실험의 입력
├── 01_cli_smoke/                # 코드가 실행되는지 확인한 최소 테스트
├── 02_debug_direct_head/        # non-equivariant debug head 실험
├── 03_inputquery_legacy/        # 과거 input_query 실험 / 실패·중간 결과
├── 04_equivariant_basis_head/   # 현재 핵심 실험: production-compatible head
└── 99_scratch_or_old/           # 임시 파일, 오래된 파일, 삭제 전 보관
00_inputs       : 실험에 사용되는 재료
01_cli_smoke    : 코드 기능 확인
02_debug_direct : 학습 루프가 정상인지 확인한 debug baseline
03_legacy       : 예전 방식 실험 보관
04_basis        : 현재 연구 핵심 실험
99_scratch      : 버리긴 애매한 임시 파일

1. 00_inputs
역할 : 학습과 추론에 반복해서 사용하는 입력 파일을 넣는 곳.
        checkpoint, inference 결과, log를 넣지 않는다.
추천 구조 : 
00_inputs/
├── synthetic/
└── diffalign_example/
    ├── diffalign_example_train.pt
    ├── diffalign_example_infer.pt
    └── original_files/
        ├── query_original.sdf
        ├── reference.sdf
        └── pocket.pdb
| 파일                           | 역할                                                                          |
| ---------------------------- | --------------------------------------------------------------------------- |
| `diffalign_example_train.pt` | pseudo train batch. `target_query_pos`가 포함되어 학습에 사용된다.                      |
| `diffalign_example_infer.pt` | inference용 batch. target 없이 추론용 입력으로 사용한다.                                  |
| `query_original.sdf`         | DiffAlign example의 query ligand 원본 구조. SDF export 결과와 RMSD 비교할 때 기준으로 사용한다. |
| `reference.sdf`              | reference ligand 구조. alignment 조건으로 사용한다.                                   |
| `pocket.pdb`                 | pocket/receptor 구조. pocket conditioning 및 시각화에 사용한다.                        |

가장 중요한 파일 : 00_inputs/diffalign_example/diffalign_example_train.pt
앞으로 학습 명령의 --train-data는 이 파일을 기준으로 잡으면 된다.

2. 01_cli_smoke : 코드가 최소한 실행되는지 확인한 파일을 넣는다.
        논문적 의미가 있는 실험 결과라기보다는 
        “CLI, save_path, checkpoint 저장, inference 저장이 되는지 확인”한 산출물.
추천 구조:
01_cli_smoke/
├── checkpoints/
└── inference/
| 파일                                                            | 역할                                                                      |
| ------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `etflowalign_smoke.pt`                                        | synthetic smoke train으로 만든 checkpoint. 실제 molecular alignment 의미는 약하다.  |
| `etflowalign_infer.pt`                                        | synthetic smoke inference 결과. `--save-path`가 작동하는지 확인한 파일.              |
| `etflowalign_sample_train.pt`                                 | sample train batch 또는 초기 샘플 학습 파일.                                      |
| `etflowalign_sample_infer.pt`                                 | sample inference batch 또는 sample inference output.                      |
| `etflowalign_real.pt`                                         | 실제 `.pt` batch 입력을 써서 학습/저장 확인한 checkpoint.                             |
| `etflowalign_real_infer.pt`                                   | 실제 `.pt` batch를 사용한 inference 저장 확인 결과.                                 |
| `test_best_checkpoint.pt`, `test_best_checkpoint_best.pt`     | best checkpoint 저장 로직 확인용.                                              |
| `test_fixedt_checkpoint.pt`, `test_fixedt_checkpoint_best.pt` | `--fixed-t`와 best checkpoint 저장 확인용.                                    |
| `test_pr11_debug_direct_checkpoint.pt`, `_best.pt`            | PR #11 수정 후 1-step smoke test checkpoint.                               |
| `test_flow_args_inputquery.pt`                                | checkpoint의 `flow_args`에 `source_type=input_query` 등이 저장되는지 확인한 테스트 파일. |
(이 폴더의 파일은 재현성 검증용이지, 현재 연구 성능 비교의 중심 파일은 아니다.)

3. 02_debug_direct : 이 폴더의 목적은 production 모델 평가가 아니라 학습 루프 sanity check
--use-direct-vector-head를 사용한 실험 결과를 넣는다.
이 head는 다음 구조다 : vi=MLP(hi)
장점 : 표현력이 높아서 overfit이 잘 된다.
단점 : E(3)-equivariance를 보장하지 않는다.
추천 구조 :
02_debug_direct_head/
├── fixedt05/
│   ├── checkpoints/
│   ├── logs/
│   └── diagnostics/
└── randomt/
    ├── checkpoints/
    ├── inference/
    ├── sdf/
    ├── logs/
    └── diagnostics/

2-1. 02_debug_direct_head/fixedt05/ : t=0.5 하나만 고정해서 direct vector head가 
                                    target velocity를 외울 수 있는지 확인한 실험이다.
| 파일                                             | 역할                                                    |
| ---------------------------------------------- | ----------------------------------------------------- |
| `etflowalign_debug_direct_fixedt05_1k.pt`      | 1k step 학습 후 final checkpoint.                        |
| `etflowalign_debug_direct_fixedt05_1k_best.pt` | 학습 중 best loss checkpoint. 주로 이 파일을 diagnostic에 사용한다. |
| `train_debug_direct_fixedt05_1k.log`           | 학습 loss 변화 기록.                                        |
(이 실험에서 RMSE가 거의 0에 가까웠으므로, 다음이 확인됐다 : 
학습 loop, target_query_pos, input_query source, checkpoint 저장/복원은 정상이다.)

2-2. 02_debug_direct_head/randomt/ : random t 전체 구간에서 direct vector head가 
                                    flow field를 학습할 수 있는지 확인한 실험이다.
| 파일                                             | 역할                                                                     |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| `etflowalign_debug_direct_randomt_1k.pt`       | random t direct head final checkpoint.                                 |
| `etflowalign_debug_direct_randomt_1k_best.pt`  | random t direct head best checkpoint.                                  |
| `etflowalign_debug_direct_randomt_1k_infer.pt` | best checkpoint로 inference한 결과. `candidates`, `scores`, `metadata` 포함. |
| `etflowalign_debug_direct_randomt_1k_sdf/`     | inference 결과를 SDF로 변환한 폴더.                                             |
| `train_debug_direct_randomt_1k.log`            | random t direct head 학습 로그.                                            |
SDF폴더 내부
| 파일                                       | 역할                        |
| ---------------------------------------- | ------------------------- |
| `debug_direct_randomt_candidate_000.sdf` | 첫 번째 candidate 구조.        |
| `debug_direct_randomt_candidate_001.sdf` | 두 번째 candidate 구조.        |
| `debug_direct_randomt_candidate_all.sdf` | 모든 candidate를 하나로 합친 SDF. |

4. 03_legacy       : 예전 방식 실험 보관
                    초기에 source_type=input_query를 도입하고, 
                    기존 head 또는 중간 수정본으로 실험했던 파일을 보관하는 폴더
추천 구조 : 
03_inputquery_legacy/
├── checkpoints/
├── logs/
├── inference/
└── sdf/
(기존 gate*x 계열 head로는 pseudo train batch를 충분히 외우지 못했다.
일부 SDF에서는 bond geometry가 깨졌다.
이 폴더는 현재 연구의 “실패 기준선”이다.)

5. 04_equivariant_basis_head: --use-equivariant-basis-head를 켠 실험을 모두 여기에 넣는다.
                    현재 연구 핵심 실험
equivariant basis head 형태 : vi=αixi+βiri+γipi+δimi
| 기호    | 의미                                               |
| ----- | ------------------------------------------------ |
| (x_i) | query-centered coordinate basis                  |
| (r_i) | reference center에서 query atom으로 향하는 vector basis |
| (p_i) | pocket center에서 query atom으로 향하는 vector basis    |
| (m_i) | query neighbor aggregate vector basis            |

추천 구조:
04_equivariant_basis_head/
├── fixedt05/
│   ├── checkpoints/
│   ├── logs/
│   └── diagnostics/
├── fixedt05_atomindex/
│   ├── checkpoints/
│   ├── logs/
│   └── diagnostics/
└── randomt_atomindex/
    ├── checkpoints/
    ├── inference/
    ├── sdf/
    ├── logs/
    └── diagnostics/
(이 폴더의 목적은 debug direct head를 
production-compatible equivariant head로 바꾸는 과정을 관리하는 것이다.)

5-1. 04_equivariant_basis_head : atom index 없이 basis head만 켜서 fixed_t=0.5 overfit을 확인한 실험.
| 파일                                      | 역할                                                 |
| --------------------------------------- | -------------------------------------------------- |
| `etflowalign_basis_fixedt05_1k.pt`      | basis head only, fixed_t=0.5, 1k final checkpoint. |
| `etflowalign_basis_fixedt05_1k_best.pt` | basis head only, fixed_t=0.5, 1k best checkpoint.  |
[결과와 그에 대한 해석]
RMSE ≈ 0.95 Å -> basis head는 gate*x보다 낫지만, atom identity 정보 없이 충분하지는 않다.

5-2. 04_equivariant_basis_head/fixedt05_atomindex/ : basis head에 --use-atom-index-embed를 추가해서, 
                                                    성능 병목이 atom identity 부족인지 확인한 실험
| 파일                                                | 역할                                                         |
| ------------------------------------------------- | ---------------------------------------------------------- |
| `etflowalign_basis_fixedt05_atomindex_1k.pt`      | basis head + atom index, fixed_t=0.5, 1k final checkpoint. |
| `etflowalign_basis_fixedt05_atomindex_1k_best.pt` | basis head + atom index, fixed_t=0.5, 1k best checkpoint.  |
| `train_basis_fixedt05_atomindex_1k.log`           | 학습 로그.                                                     |
[결과와 그에 대한 해석]
RMSE ≈ 0.124 Å -> basis head는 유효하다.
                성능 병목은 basis vector 자체보다는 atom identity / graph feature 부족 쪽이 크다.

5-3. 04_equivariant_basis_head/randomt_atomindex : basis head + atom index 설정을 random t 전체 구간으로 확장한 실험.
                                                    가장 중요한 작업 폴더.
| 파일                                                | 역할                                                      |
| ------------------------------------------------- | ------------------------------------------------------- |
| `etflowalign_basis_atomindex_randomt_1k.pt`       | basis head + atom index, random t, 1k final checkpoint. |
| `etflowalign_basis_atomindex_randomt_1k_best.pt`  | basis head + atom index, random t, 1k best checkpoint.  |
| `etflowalign_basis_atomindex_randomt_1k_infer.pt` | 1k best checkpoint로 inference한 결과. 확인용으로 생성 가능.         |
| `etflowalign_basis_atomindex_randomt_1k_sdf/`     | 1k inference 결과를 SDF로 export한 폴더.                       |
| `train_basis_atomindex_randomt_1k.log`            | 1k random t 학습 로그.                                      |
[결과와 해석]
t=0.00  RMSE = 1.140968 Å
t=0.25  RMSE = 0.769866 Å
t=0.50  RMSE = 0.505519 Å
t=0.75  RMSE = 0.322941 Å
t=1.00  RMSE = 0.592195 Å
->중간~후반 t는 어느 정도 학습했지만, t=0 근처가 아직 불안정하다.
6. 99_scratch      : 버리긴 애매한 임시 파일

7. 파일 저장 규칙
- checkpoint는 항상 checkpoints/에 넣기
- log는 항상 logs/에 넣기
- diagnostic 결과는 diagnostics/에 넣기. 앞으로 t별 RMSE 결과를 텍스트로 저장하면 좋다.
- inference .pt는 inference/에 넣기.
- SDF는 sdf/에 넣기

8. 각 .pt 파일 종류 구분법
| 종류                     | 위치                                   | 내용                                                               |
| ---------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| input batch `.pt`      | `00_inputs/`                         | `query_pos`, `reference_pos`, `pocket_pos`, `target_query_pos` 등 |
| checkpoint `.pt`       | `checkpoints/`                       | `model_state`, `model_args`, `flow_args`, `best_loss` 등          |
| inference output `.pt` | `inference/`                         | `candidates`, `scores`, `metadata`                               |
| temporary test `.pt`   | `99_scratch_or_old/temporary_tests/` | 특정 버그 확인용 임시 checkpoint                                          |
[확인하는 방법]
python - <<'PY'
import torch
path = "파일경로.pt"
obj = torch.load(path, map_location="cpu", weights_only=False)
print(obj.keys())
PY

[앞으로 사용할 경로 모음집]
TRAIN_ROOT=/home/deepfold/users/hosung/work/ETFlowAlign/etflowalign/smoke_tests
INPUT_PT=$TRAIN_ROOT/00_inputs/diffalign_example/diffalign_example_train.pt
OUT_ROOT=$TRAIN_ROOT/04_equivariant_basis_head/randomt_atomindex

python -m etflowalign.train \
  --train-data $INPUT_PT \
  --steps 3000 \
  --save-path $OUT_ROOT/checkpoints/etflowalign_basis_atomindex_randomt_3k.pt \
  --sigma 0.0 \
  --source-type input_query \
  --source-noise-scale 0.0 \
  --no-center-source \
  --no-center-target \
  --no-use-kabsch-alignment \
  --lr 1e-4 \
  --hidden-dim 256 \
  --num-blocks 6 \
  --use-equivariant-basis-head \
  --use-atom-index-embed \
  --log-every 100 \
  2>&1 | tee $OUT_ROOT/logs/train_basis_atomindex_randomt_3k.log