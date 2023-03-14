---
title:  "[Faster R-CNN] Towards Real-Time Object Detection with Region Proposal Networks"
excerpt: "Faster R-CNN Paper Review & Code Implementation"

categories: [Deep Learning]
tags: [Computer Vision]

permalink: /deeplearning/fasterrcnn
---
<!-- 
### Introduction

객체 탐지 알고리즘은 크게 Proposal-based과 Proposal-free 모델로 나눠진다. Faster R-CNN 모델은 대표적인 Proposal-based (Two-Stage) 알고리즘이며 Fast R-CNN 모델의 단점을 개선한 알고리즘이다. 따라서 본 포스팅을 시작하기에 앞서 R-CNN 과 Fast R-CNN 모델을 간략히 소개하고자 한다. 

#### R-CNN
<center><img src="../../images/2022-10-01-fasterrcnn/rcnn.png" style="zoom:30%"></center>
1. Stage One: Region Proposal
- Selective Search와 같은 off-the-shelf model를 사용해, 이미지로부터 Object가 존재할 만한 위치 (~2000개 정도) 찾는다.
2. Stage Two: Object Recognition
- 모든 Region Proposal을 동일한 크기로 만들고, CNN 특성맵을 추출한 뒤, Classifier 모델로 Object를 분류한다.

#### Fast R-CNN
Fast R-CNN은 모든 Region Proposal이 CNN를 거쳐야 하는 병목 문제를 개선하고자 제안된 방식이다. R-CNN 모델과의 가장 큰 차이점은, 각 Region Proposal이 CNN 모델을 거치는 것이 아니라 이미지를 CNN 모델에 통과시켜 특성맵을 추출하고 RoI (Region of Interest) 찾은 뒤에 객체 탐지를 수행하는 데에 있다.
 -->
