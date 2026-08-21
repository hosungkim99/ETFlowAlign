"""등변 벡터-피처 백본 (TorchMD-Net/ET 계열, self-contained 재구현).

SPEC.md §5 참조. 이 서브패키지는 태스크/조건화를 전혀 모르는
'순수 등변 네트워크'로 격리되어 있어 Phase 2에서 단독 검증이 가능하다.
"""

from etflowalign.backbone.network import EquivariantTransformer

__all__ = ["EquivariantTransformer"]
