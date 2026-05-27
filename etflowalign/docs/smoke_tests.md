아래 내용을 그대로 `docs/smoke_tests.md`에 넣으면 된다.

````markdown
# ETFlowAlign Smoke Test 기록

## 1. DiffAlign example pseudo-overfit test

이 문서는 ETFlowAlign의 현재 flow-matching scaffold가 DiffAlign example batch에서 정상적으로 동작하는지 검증한 smoke test 결과를 정리한다.

이 smoke test는 다음 항목을 확인한다.

1. DiffAlign 형식의 molecular alignment batch를 정상적으로 읽을 수 있는가.
2. pseudo target을 기준으로 flow-matching model을 학습할 수 있는가.
3. deterministic ODE inference를 non-finite error 없이 수행할 수 있는가.
4. 생성된 candidate coordinate를 SDF 파일로 export할 수 있는가.
5. 생성된 ligand 구조가 기본적인 molecular geometry를 유지하는가.

---

## 2. 입력 데이터

사용한 입력은 DiffAlign repository의 example 파일이다.

```text
query molecule:
external/diffalign/diffalign/example/query.sdf

reference molecule:
external/diffalign/diffalign/example/reference.sdf

pocket structure:
external/diffalign/diffalign/example/pocket.pdb
````

ETFlowAlign smoke test용 batch는 다음 위치에 저장했다.

```text
etflowalign/smoke_tests/00_inputs/diffalign_example/diffalign_example_train.pt
```

pseudo target은 다음과 같이 정의했다.

```text
target_query_pos = query_sdf_conformer - reference_center
```

즉, query ligand의 원래 3D conformer를 reference ligand 중심 기준으로 center-subtraction한 좌표를 학습 target으로 사용했다.

---

## 3. 검증한 모델 설정

현재 검증한 checkpoint는 다음이다.

```text
checkpoint:
etflowalign/smoke_tests/04_equivariant_basis_head/randomt_atomindex/checkpoints/etflowalign_basis_atomindex_randomt_3k_best.pt
```

모델 설정은 다음과 같다.

```text
model:
equivariant basis head + atom index embedding

source_type:
input_query

source_noise_scale:
0.0

sigma:
0.0

training time sampling:
random t

training steps:
3000

best checkpoint:
3k best checkpoint
```

현재 checkpoint는 `atom index embedding`을 사용한다. 따라서 production-compatible 모델이라기보다는, **basis head와 sampler가 end-to-end로 작동하는지 확인하기 위한 debug-successful checkpoint**로 해석한다.

---

## 4. 중요한 sampler 수정 사항

ODE sampling 과정에서 sampler는 매 integration step마다 현재 query coordinate `x_t`를 사용해 새로운 `AlignmentBatch`를 구성한다.

이때 다음 conditioning field를 반드시 보존해야 한다.

```text
pocket_batch
query_node_attr
reference_node_attr
```

기존 sampler에서는 일부 field가 누락되어 inference-time batch가 training/diagnostic batch와 달라질 수 있었다. 이 문제는 ODE trajectory 불안정성 및 non-finite coordinate 발생 원인으로 작용할 수 있다.

수정 후 sampler는 다음 항목을 수행한다.

```text
1. pocket_batch 보존
2. query_node_attr 보존
3. reference_node_attr 보존
4. model velocity finite check
5. updated coordinate finite check
6. 더 명확한 FloatingPointError 메시지 출력
```

이 수정 후, 이전에 발생하던 ODE integration 중 `nan` 또는 `inf` 좌표 문제가 사라졌다.

---

## 5. Inference 설정

검증한 inference 조건은 다음과 같다.

```text
checkpoint:
etflowalign_basis_atomindex_randomt_3k_best.pt

input batch:
diffalign_example_train.pt

num_samples:
4

n_steps:
500

solver:
Euler, Heun
```

현재 설정은 deterministic하다.

```text
source_type = input_query
source_noise_scale = 0.0
sigma = 0.0
```

따라서 `num_samples=4`는 conformer diversity를 검증하지 않는다. 같은 시작점과 같은 ODE trajectory를 4번 반복하는 deterministic reproducibility test로 해석한다.

---

## 6. Euler sample4 sanity check 결과

Euler solver로 `num_samples=4`, `n_steps=500` inference를 수행했다.

SDF export 결과에서 candidate별 geometry sanity check를 수행했다.

```text
file,rmsd,com_dist,bond_min,bond_mean,bond_max
basis_atomindex_randomt_3k_euler_nsteps500_sample4_candidate_000.sdf,0.497199,0.133344,0.988652,1.387440,1.976612
basis_atomindex_randomt_3k_euler_nsteps500_sample4_candidate_001.sdf,0.497199,0.133344,0.988652,1.387440,1.976612
basis_atomindex_randomt_3k_euler_nsteps500_sample4_candidate_002.sdf,0.497199,0.133344,0.988652,1.387440,1.976612
basis_atomindex_randomt_3k_euler_nsteps500_sample4_candidate_003.sdf,0.497199,0.133344,0.988652,1.387440,1.976612
```

요약하면 다음과 같다.

```text
RMSD to query_original:
0.497199 Å

COM distance to query_original:
0.133344 Å

bond length min / mean / max:
0.988652 / 1.387440 / 1.976612 Å
```

Euler 결과는 smoke test 기준을 통과했다.

---

## 7. Heun sample4 sanity check 결과

Heun solver로도 `num_samples=4`, `n_steps=500` inference를 수행했다.

```text
file,rmsd,com_dist,bond_min,bond_mean,bond_max
basis_atomindex_randomt_3k_heun_nsteps500_sample4_candidate_000.sdf,0.496917,0.133429,0.989528,1.387700,1.977781
basis_atomindex_randomt_3k_heun_nsteps500_sample4_candidate_001.sdf,0.496917,0.133429,0.989528,1.387700,1.977781
basis_atomindex_randomt_3k_heun_nsteps500_sample4_candidate_002.sdf,0.496917,0.133429,0.989528,1.387700,1.977781
basis_atomindex_randomt_3k_heun_nsteps500_sample4_candidate_003.sdf,0.496917,0.133429,0.989528,1.387700,1.977781
```

요약하면 다음과 같다.

```text
RMSD to query_original:
0.496917 Å

COM distance to query_original:
0.133429 Å

bond length min / mean / max:
0.989528 / 1.387700 / 1.977781 Å
```

Heun 결과도 smoke test 기준을 통과했다.

---

## 8. Euler와 Heun 비교

| solver | RMSD to query_original | COM distance |   bond min |  bond mean |   bond max | 판단 |
| ------ | ---------------------: | -----------: | ---------: | ---------: | ---------: | -- |
| Euler  |             0.497199 Å |   0.133344 Å | 0.988652 Å | 1.387440 Å | 1.976612 Å | 통과 |
| Heun   |             0.496917 Å |   0.133429 Å | 0.989528 Å | 1.387700 Å | 1.977781 Å | 통과 |

Euler와 Heun 결과는 거의 동일하다. 따라서 현재 deterministic pseudo-overfit 설정에서는 solver 차이가 구조 품질에 큰 영향을 주지 않았다.

---

## 9. 해석

이번 smoke test는 통과했다고 판단한다.

확인한 내용은 다음과 같다.

```text
1. DiffAlign example 기반 pseudo train batch를 정상적으로 읽었다.
2. basis head + atom index embedding 모델이 random t flow matching을 학습했다.
3. sampler fix 이후 ODE inference가 non-finite coordinate 없이 완료됐다.
4. Euler와 Heun solver 모두 n_steps=500에서 안정적으로 작동했다.
5. SDF export가 정상적으로 수행됐다.
6. 생성된 candidate ligand는 query_original에 가깝게 복원됐다.
7. bond length가 심하게 붕괴하지 않았다.
```

수치적으로는 다음 기준을 만족했다.

```text
RMSD < 1.0 Å
COM distance < 0.5 Å
bond_min > 0.9 Å
bond_max < 2.0 Å
```

---

## 10. 주의 사항

현재 결과는 conformer diversity 검증이 아니다.

현재 inference 설정은 deterministic하다.

```text
source_type = input_query
source_noise_scale = 0.0
sigma = 0.0
```

따라서 `num_samples=4` 결과가 모두 동일하게 나오는 것은 정상이다.

이 결과는 다음을 의미한다.

```text
num_samples=4:
  deterministic reproducibility 확인

아직 확인하지 않은 것:
  stochastic source에서 다양한 conformer/alignment candidate를 생성하는 능력
```

진짜 multi-sample diversity를 검증하려면 별도 학습 설정이 필요하다.

예시는 다음과 같다.

```text
source_type = input_query, source_noise_scale > 0
source_type = reference_anchored, source_noise_scale > 0
source_type = gaussian
```

단, 현재 checkpoint는 `source_noise_scale=0.0`으로 학습했으므로, inference에서 임의로 noise를 추가하면 학습 분포 밖으로 나갈 수 있다. diversity 검증은 별도 실험으로 분리해야 한다.

---

## 11. 현재 결론

현재 ETFlowAlign의 pseudo-overfit end-to-end smoke test는 통과했다.

```text
status:
pass

validated components:
- DiffAlign example batch preparation
- flow-matching training
- checkpoint loading
- deterministic ODE inference
- Euler sampler
- Heun sampler
- SDF export
- molecular geometry sanity check
```

이번 단계에서 가장 중요한 코드 수정은 sampler에서 conditioning field를 보존한 것이다.

```text
pocket_batch
query_node_attr
reference_node_attr
```

이 수정 후 inference 안정성이 크게 개선됐다.

---

## 12. 다음 개발 단계

다음 목표는 `atom index embedding`을 제거하고, chemistry/graph feature 기반 conditioning으로 대체하는 것이다.

현재 상태:

```text
basis head + atom index embedding:
  debug 단계에서는 성공
  하지만 atom ordering에 의존하므로 production-compatible하지 않음
```

다음 목표:

```text
basis head + chemistry/graph node features:
  atom ordering 의존성을 줄이고,
  molecular graph와 chemistry feature를 사용한 production-compatible 방향
```

진행 순서는 다음과 같다.

```text
1. diffalign_adapter.py에서 query_node_attr/reference_node_attr 생성 여부 확인
2. AlignmentBatch가 node_attr를 training/inference/sampling 전 구간에서 보존하는지 확인
3. model.py에서 node_attr embedding을 input embedding에 반영
4. --use-atom-index-embed 없이 fixed_t=0.5 overfit 실험
5. --use-atom-index-embed 없이 random_t overfit 실험
6. Euler/Heun inference 및 SDF sanity check 반복
```

---

````

커밋 명령은 다음처럼 하면 된다.

```bash
mkdir -p docs
nano docs/smoke_tests.md

git add docs/smoke_tests.md
git commit -m "Document smoke tests in Korean"
git push
````
