load reference.sdf, reference
load pocket.pdb, pocket
load query_original.sdf, query_original
load diffalign_example_candidate_000.sdf, candidate_000
load diffalign_example_candidate_001.sdf, candidate_001

hide everything

show sticks, reference
show sticks, query_original
show sticks, candidate_000
show sticks, candidate_001
show surface, pocket

color orange, reference
color white, query_original
color green, candidate_000
color cyan, candidate_001
color gray80, pocket

set transparency, 0.75, pocket
set stick_radius, 0.18
set sphere_scale, 0.25

orient reference or candidate_000 or candidate_001 or pocket
zoom reference or candidate_000 or candidate_001 or pocket, 4

set ray_opaque_background, off
png diffalign_example_view.png, width=1600, height=1200, dpi=200, ray=1

quit
