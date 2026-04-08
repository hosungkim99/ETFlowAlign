

## flow_matching.py

여기에는:

ET-Flow 기반 objective

path definition

velocity target

time sampling

training target 생성

을 넣는다.

즉 diffusion loss를 대체하는 수학적 핵심을 둔다.


## inference.py
여기에는:

inference entry point

evaluation/inference pipeline

를 둔다.


## model.py
가장 중요하다.

여기에는:

ETFlowAlign 전체 모델 구조

DiffAlign에서 유지한 부분

ET-Flow로 바꾼 부분

forward 흐름

을 넣는다.

이 파일 하나만 봐도
“아, 이 모델이 어떻게 생겼는지”
알 수 있어야 한다.

## sampler.py
여기에는:

inference / generation / integration loop

Euler / ODE step

iterative update

를 둔다.

즉 “학습된 flow를 가지고 실제로 어떻게 샘플을 얻는가”를 정리한다.


## train.py
여기에는:

training step

loss 호출

optimizer step

batch 처리

를 둔다.

## utils.py
공통 유틸은 여기에 모은다.

하지만 너무 많은 핵심 로직을 여기 숨기면 안 된다.

