# ETFlowAlign Smoke Test Notes

## DiffAlign example pseudo-overfit test

This smoke test validates the current ETFlowAlign flow-matching scaffold on the DiffAlign example batch.

The test checks whether ETFlowAlign can:

1. load a DiffAlign-style molecular alignment batch,
2. train a flow-matching model on a pseudo target,
3. run deterministic ODE inference,
4. export candidate coordinates to SDF,
5. preserve basic molecular geometry.

## Input

- Source batch: `diffalign_example_train.pt`
- Query molecule: DiffAlign example `query.sdf`
- Reference molecule: DiffAlign example `reference.sdf`
- Pocket structure: DiffAlign example `pocket.pdb`

The pseudo target is defined as:

```text
target_query_pos = query_sdf_conformer - reference_center
Important sampler fix

During ODE sampling, the sampler reconstructs an AlignmentBatch at every integration step.

The following conditioning fields must be preserved:

pocket_batch
query_node_attr
reference_node_attr

Without these fields, the inference-time batch can differ from the training/diagnostic batch. This previously caused unstable ODE trajectories and non-finite coordinates.

The sampler now preserves these fields and performs explicit finite checks for model velocity and updated coordinates.

Current validated checkpoint
checkpoint:
04_equivariant_basis_head/randomt_atomindex/checkpoints/etflowalign_basis_atomindex_randomt_3k_best.pt

model:
equivariant basis head + atom index embedding

source_type:
input_query

source_noise_scale:
0.0

sigma:
0.0

n_steps:
500

num_samples:
4
Euler sample4 sanity check
RMSD to query_original: 0.497199 Å
COM distance: 0.133344 Å
bond_min: 0.988652 Å
bond_mean: 1.387440 Å
bond_max: 1.976612 Å
Heun sample4 sanity check
RMSD to query_original: 0.496917 Å
COM distance: 0.133429 Å
bond_min: 0.989528 Å
bond_mean: 1.387700 Å
bond_max: 1.977781 Å
Interpretation

The pseudo-overfit end-to-end smoke test passes.

Checkpoint loading works.
ODE inference completes without non-finite coordinates.
Euler and Heun solvers both work with n_steps=500.
SDF export works.
Candidate geometry remains chemically plausible.
The predicted structure is close to the query conformer.

All four samples are identical because the current setup is deterministic:

source_type = input_query
source_noise_scale = 0.0
sigma = 0.0

Therefore, num_samples=4 verifies deterministic reproducibility, not conformer diversity.

Next development step

The next step is to replace the debug atom-index embedding with chemistry/graph-based node features.

Target direction:

basis head + atom index embedding
    -> debug-successful but atom-order dependent

basis head + chemistry/graph node features
    -> production-compatible direction
