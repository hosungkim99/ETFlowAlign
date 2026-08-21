"""데이터 파이프라인: 정렬쌍 SDF -> 텐서 계약(.pt).

무거운 정렬/필터링(유사분자쌍 생성)은 서버에서 이미 수행됨
(GEOM-Drugs_AlignedPairs/complex_*/{query,reference}.sdf).
여기서는 그 SDF 쌍을 SPEC §3.1 텐서 계약으로 변환한다.
"""

from etflowalign.data.pairs import (
    build_pair_payload,
    collate,
    mol_to_graph,
    tanimoto_2d,
)

__all__ = ["build_pair_payload", "collate", "mol_to_graph", "tanimoto_2d"]
