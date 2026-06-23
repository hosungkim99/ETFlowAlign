
## nodeattr random-t sampler geometry diagnostic 준비 기록

### 현재 목표

ETFlowAlign random-t sampler가 ligand bond geometry를 언제부터 어떻게 깨뜨리는지 확인한다.

### 이전 결과 요약

#### 1. nodeattr fixed_t=0.5 1k

- 설정:
  - `use_equivariant_basis_head=True`
  - `use_node_attr=True`
  - `use_atom_index_embed=False`
  - `fixed_t=0.5`
- direct vector diagnostic:
  - `t=0.5`에서는 비교적 양호
  - `t=0.0`, `0.25`, `0.75`, `1.0`에서는 RMSE가 매우 큼
- SDF sanity:
  - 실패
  - RMSD, bond_mean, bond_max가 모두 비정상적으로 큼
- 해석:
  - fixed-t checkpoint는 학습 경로 확인용으로는 충분하지만, full ODE inference용으로는 부적합하다.

#### 2. nodeattr random_t 3k

- 설정:
  - `use_equivariant_basis_head=True`
  - `use_node_attr=True`
  - `use_atom_index_embed=False`
  - `fixed_t=None`
- direct vector diagnostic:
  - 전체 t 구간에서 fixed-t보다 크게 개선됨
  - `t=0.25`, `0.5`, `0.75`, `1.0`은 대체로 1 Å 안팎
  - `t=0.0`은 아직 큼
- SDF sanity:
  - global COM은 크게 개선됨
  - 그러나 ligand internal geometry는 실패
  - bond_mean과 bond_max가 query original 대비 매우 큼
- 해석:
  - random-t 모델은 ligand 위치는 어느 정도 맞추지만, bond geometry를 보존하지 못한다.
  - 현재 문제는 단순 placement 실패보다 atom-wise coordinate flow가 molecule internal structure를 깨뜨리는 문제에 가깝다.

### 오늘 할 일

1. ODE trajectory 중간 state별 bond statistics를 확인한다.
2. `t=0`에서 이미 bond가 깨져 있는지 확인한다.
3. `t=0`은 정상인데 integration 과정에서 bond가 깨지는지 확인한다.
4. 결과에 따라 source/export bug인지, geometry-preserving constraint 문제인지 구분한다.

### Trajectory diagnostic 결과

random-t 3k checkpoint로 ODE trajectory를 확인한 결과, step=0에서 이미 query source geometry가 깨져 있었다.

- query original bond mean: 1.4247 Å
- TRAIN_PT query_pos bond mean: 2.4162 Å
- INFER_PT query_pos bond mean: 2.4162 Å
- TRAIN_PT target_query_pos bond mean: 1.4248 Å

따라서 현재 geometry 붕괴는 sampler/export에서 처음 발생한 문제가 아니라, batch 생성 단계에서 `query_pos`를 atom-wise randomized source로 만든 데서 시작된다.

다음 수정 방향은 `query_pos`를 bond-breaking atom cloud가 아니라, valid query conformer에 random rigid transform을 적용한 geometry-preserving source로 만드는 것이다.

### rigid-source random-t 3k inference / Heun trajectory

Rigid-randomized query conformer source로 batch를 재생성하고 random-t 3k 학습을 수행했다.

- checkpoint:
  - use_node_attr=True
  - use_equivariant_basis_head=True
  - use_atom_index_embed=False
  - fixed_t=None
  - best_loss=0.247526 at step=2993

- inference:
  - candidates shape: (4, 35, 3)
  - SDF export 정상 완료

- trajectory diagnostic, Heun n_steps=500:
  - step=0 bond_mean=1.424727 Å
  - final RMSD=1.304932 Å
  - final COM distance=0.145994 Å
  - final bond_min=0.808200 Å
  - final bond_mean=1.397953 Å
  - final bond_max=2.271400 Å

해석:
- t=0 source geometry 문제는 해결됐다.
- global alignment는 크게 개선됐다.
- 평균 bond length는 정상 근처다.
- 일부 local bond distortion은 남아 있다.
- Heun n_steps=500은 Euler n_steps=500 대비 실질적 개선을 보이지 않았다.

### rigid-source nodeattr random-t 3k 결과

Rigid-randomized query conformer source를 사용해 nodeattr + equivariant basis head random-t 3k 학습을 수행했다.

#### Batch 검증

- query_sdf bond mean: 1.424727 Å
- TRAIN query_pos bond mean: 1.424727 Å
- TRAIN target_query_pos bond mean: 1.424727 Å
- INFER query_pos bond mean: 1.424727 Å
- source_target_rmsd: 9.222653 Å
- query_source: `rigid_randomized_query_conformer`

#### Checkpoint

- best_loss: 0.247526
- best_step: 2993
- model:
  - use_equivariant_basis_head=True
  - use_node_attr=True
  - use_atom_index_embed=False
  - node_attr_dim=5
- flow:
  - source_type=input_query
  - fixed_t=None
  - sigma=0.0
  - source_noise_scale=0.0

#### SDF sanity, Euler n_steps=500

- RMSD to query: 1.300336 Å
- COM distance: 0.146268 Å
- bond_min: 0.807628 Å
- bond_mean: 1.397864 Å
- bond_max: 2.266401 Å

#### Trajectory diagnostic

Euler n_steps=1000 final:

- RMSD: 1.302388 Å
- COM: 0.146047 Å
- bond_min: 0.807783 Å
- bond_mean: 1.398167 Å
- bond_max: 2.265694 Å

Heun n_steps=1000 final:

- RMSD: 1.304544 Å
- COM: 0.146123 Å
- bond_min: 0.808021 Å
- bond_mean: 1.398230 Å
- bond_max: 2.269473 Å

#### 해석

- t=0 source bond geometry 문제는 해결됐다.
- global alignment는 크게 개선됐다.
- SDF export 결과와 trajectory 결과가 일치하므로 export 문제는 아니다.
- n_steps=500→1000, Euler→Heun 변경은 실질적 개선을 주지 않았다.
- 남은 문제는 일부 local bond distortion이다.

### rigid-source nodeattr random-t 5k 결과

5k 학습은 3k 대비 best loss와 vector diagnostic을 크게 개선했다.

#### Checkpoint

- best_loss: 0.077112
- best_step: 3829
- final step loss: 0.250376
- checkpoint: `etflowalign_basis_nodeattr_rigid_randomt_5k_best.pt`

#### Direct vector diagnostic

| t | vector RMSE (Å) | mean atom error (Å) | max atom error (Å) |
|---:|---:|---:|---:|
| 0.00 | 0.983550 | 0.853239 | 1.873390 |
| 0.25 | 0.610165 | 0.489628 | 1.572996 |
| 0.50 | 0.315997 | 0.261688 | 0.938915 |
| 0.75 | 0.344846 | 0.291929 | 0.766670 |
| 1.00 | 1.023047 | 0.803800 | 3.047902 |

#### Euler n_steps=500 trajectory final

- RMSD to query: 0.522858 Å
- COM distance: 0.204091 Å
- bond_min: 0.794736 Å
- bond_mean: 1.450152 Å
- bond_max: 2.233331 Å

#### 해석

- 5k는 3k 대비 RMSD를 크게 개선했다.
- 평균 bond length는 query original에 가까운 수준이다.
- 그러나 일부 bond가 너무 짧거나 길어지는 local distortion은 남아 있다.
- 단순 solver 변경이나 step 수 증가만으로는 이 distortion이 해결되지 않을 가능성이 높다.


=== query_original ===
atoms: 35
bonds: 39
bond_min: 1.2277939566555944
bond_mean: 1.4247265128796565
bond_max: 1.5356786773280409

experiment,file,status,rmsd_to_query,com_dist,bond_min,bond_mean,bond_max,atoms,bonds
rigid_randomt_3k_euler_nsteps500_sample4,basis_nodeattr_rigid_randomt_3k_euler_nsteps500_sample4_candidate_000.sdf,OK,1.300336,0.146268,0.807628,1.397864,2.266401,35,39
rigid_randomt_3k_euler_nsteps500_sample4,basis_nodeattr_rigid_randomt_3k_euler_nsteps500_sample4_candidate_001.sdf,OK,1.300336,0.146268,0.807628,1.397864,2.266401,35,39
rigid_randomt_3k_euler_nsteps500_sample4,basis_nodeattr_rigid_randomt_3k_euler_nsteps500_sample4_candidate_002.sdf,OK,1.300336,0.146268,0.807628,1.397864,2.266401,35,39
rigid_randomt_3k_euler_nsteps500_sample4,basis_nodeattr_rigid_randomt_3k_euler_nsteps500_sample4_candidate_003.sdf,OK,1.300336,0.146268,0.807628,1.397864,2.266401,35,39
rigid_randomt_5k_euler_nsteps500_sample4,basis_nodeattr_rigid_randomt_5k_euler_nsteps500_sample4_candidate_000.sdf,OK,0.522850,0.204093,0.794766,1.450157,2.233323,35,39
rigid_randomt_5k_euler_nsteps500_sample4,basis_nodeattr_rigid_randomt_5k_euler_nsteps500_sample4_candidate_001.sdf,OK,0.522850,0.204093,0.794766,1.450157,2.233323,35,39
rigid_randomt_5k_euler_nsteps500_sample4,basis_nodeattr_rigid_randomt_5k_euler_nsteps500_sample4_candidate_002.sdf,OK,0.522850,0.204093,0.794766,1.450157,2.233323,35,39
rigid_randomt_5k_euler_nsteps500_sample4,basis_nodeattr_rigid_randomt_5k_euler_nsteps500_sample4_candidate_003.sdf,OK,0.522850,0.204093,0.794766,1.450157,2.233323,35,39

=== per-experiment mean over candidates ===

[rigid_randomt_3k_euler_nsteps500_sample4]
rmsd_to_query: mean=1.300336, min=1.300336, max=1.300336
com_dist: mean=0.146268, min=0.146268, max=0.146268
bond_min: mean=0.807628, min=0.807628, max=0.807628
bond_mean: mean=1.397864, min=1.397864, max=1.397864
bond_max: mean=2.266401, min=2.266401, max=2.266401

[rigid_randomt_5k_euler_nsteps500_sample4]
rmsd_to_query: mean=0.522850, min=0.522850, max=0.522850
com_dist: mean=0.204093, min=0.204093, max=0.204093
bond_min: mean=0.794766, min=0.794766, max=0.794766
bond_mean: mean=1.450157, min=1.450157, max=1.450157
bond_max: mean=2.233323, min=2.233323, max=2.233323