# [프로젝트 개요 및 팀 소개]

<img src="./images/img1" width="140">

## 📑 작품 개요
---
### 작품 설명

**묵인(墨印, Mookin)** 은 종이 문서에 작성된 서명을 자동으로 검출하고, 사전에 등록된 사용자의 서명과 비교하여 동일인 여부를 판단하는 **AI 기반 임베디드 서명 검증 시스템**이다.

이 시스템은 **NVIDA Jetson Orin Nano**를 기반으로 동작하며, 문서 촬영부터 서명 영역 검출,  서명 속 특징 추출, 유사도 비교 후 최종 검증 결과 출력까지의 과정을 하나의 장치에서 수행하도록 설계되었다.

문서 내 서명 영역은 **YOLOv5**를 이용해 자동으로 검출하고, 검출된 서명은 **MobileNetV3-Small 기반 특징 추출 모델**을 통해 특징 벡터로 변환된다. 등록된 서명의 특징 벡터와 비교하여 서명의 유사도를 분석하고 검증 결과를 판단한다.

외부 서버에 의존하지 않고 **온디바이스 환경에서 AI 연산을 수행**하도록 구성하여, 보안이 중요한 군부대, 관공서, 병원 등에서도 활용할 수 있는 독립적인 서명 검증 시스템을 목표로 개발하였다.

#### 작품의 필요성 및 기대효과
- 종이 문서의 서명 검출과 비교 과정을 자동화하여 신속한 1차 검증과 업무 부담 감소를 기대할 수 있다. 반복적인 육안 확인 업무를 AI가 보조함으로써 담당자는 더 중요한 업무에 집중할 수 있으며, 외부 서버 없이 장치 내부에서 처리해 민감한 서명 정보의 외부 전송을 최소화하고 보안성을 높일 수 있다.
- 단순 확인 작업에 소요되는 시간과 인력을 줄여 인력 활용의 효율성을 높일 수 있으며, 기존의 사후 대응 중심 방식에서 벗어나 사전에 의심 서명을 확인하는 예방적 검증 절차로 활용할 수 있다. 온디바이스 AI 기반 문서 처리 기술은 향후 다른 문서 검증 업무로도 확장할 수 있어 업무 자동화 범위를 넓히는 데 기여할 수 있다.
- 다양한 사용자와 서명 데이터를 지속적으로 추가하여 서명 형태 변화와 다양한 환경에 대한 모델 성능을 고도화할 수 있다. 나아가 서명 확인뿐 아니라 필기체 확인, 다양한 문서 검증 기능, 사용자·검증 이력 관리 등을 추가하고, 하드웨어 소형화와 스마트폰·시스템 연동을 통해 여러 분야에서 활용 가능한 문서 검증 플랫폼으로 발전시킬 수 있다.

## 🎥 작품 시연
---
사용자 등록부터 서명 검증 결과 출력까지 전 과정을 실제 장치에서 수행

1) 사용자 등록
	- 사용자 ID를 입력하고 등록용 서명을 촬영하여 특징 벡터를 저장한다.
	
2) 문서 촬영  
	- 검증할 문서를 장치 내부에 넣고 촬영 버튼을 눌러 문서 이미지를 획득한다.
	
3) 서명 영역 자동 검출  
	- YOLOv5를 이용해 촬영된 문서에서 서명 영역을 자동으로 검출하고 추출한다.
	
4) 서명 특징 비교  
	- MobileNetV3-Small 기반 모델로 특징 벡터를 추출한 뒤, 등록된 서명과 Cosine Similarity를 이용해 비교한다.
	
5) 검증 결과 출력  
	- 유사도 기준에 따라 동일인 여부를 판단하고 LCD를 통해 사용자에게 결과를 안내한다.

[2026ESWContest_자유공모_인증_시연동영상](https://www.youtube.com/watch?v=9eVswU_IZxI&feature=youtu.be)


## 📝 프로젝트 정보
---
- 대회명: 제 24회 임베디드 소프트웨어 경진 대회(The World Embedded  Software Contest 2026) 
- 지원 뷴야: 자유공모
- 개발 기간: 2026.06 ~ 2026.09

## 🚀 Main Features
---
- **서명 사용자 등록**  
    사용자의 다수 서명을 촬영하여 특징 벡터를 추출하고, 사용자 ID와 함께 등록 데이터로 저장한다.
- **문서 자동 촬영**  
    카메라와 조명을 제어하여 일정한 환경에서 문서를 촬영하고, 후속 이미지 처리 단계로 전달한다.
- **YOLOv5 기반 서명 영역 자동 검출**  
    촬영된 문서 전체 이미지에서 YOLOv5 모델을 이용해 서명이 존재하는 영역을 자동으로 탐지하고 추출한다.
- **MobileNetV3-Small 기반 특징 추출**  
    검출된 서명 이미지를 경량 신경망에 입력하여 서명의 형태적 특징을 벡터로 변환한다.
- **ArcFace 기반 서명 특징 학습**  
    ArcFace Loss를 적용하여 동일 사용자의 서명 특징은 유사하게, 서로 다른 사용자의 서명 특징은 구분되도록 모델을 학습한다.
- **Cosine Similarity 기반 서명 비교**  
    검증 대상 서명의 특징 벡터와 등록된 특징 벡터 사이의 코사인 유사도를 계산하여 동일인 여부를 판단한다.
- **Jetson Orin Nano 온디바이스 AI**  
    서명 검출과 검증에 필요한 AI 연산을 Jetson Orin Nano 내부에서 수행하여 외부 서버나 인터넷 연결 없이 독립적으로 동작한다.
- **GPIO 및 LCD 연동**  
    버튼, 조명, LCD 등의 하드웨어를 GPIO와 연동하여 사용자 입력부터 검증 결과 안내까지 하나의 임베디드 시스템으로 통합하였다.

## 🏗️ System Architecture 
----



## 🔗 References
#### Papers
- [Offline Signature Verification: ArcFace Evaluation Under Noise and Geometric Distortions](논문 링크)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- [Signature Detection, Restoration, and Verification: A Novel Chinese Document Signature Forgery Detection Benchmark](https://github.com/dskezju/Chisig)

#### Open Source Datasets
- [CEDAR Signature Dataset](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets)
- [BHSig260 Bengali](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets)
- [BHSig260 Hindi](https://www.kaggle.com/datasets/ishanikathuria/handwritten-signature-datasets)
- [GPDS 1-150](https://www.kaggle.com/datasets/adeelajmal/gpds-1150)

## 👥 Team

#### 👑 박주영 — Team Leader
- 서명 인식 알고리즘 구상
- 서명 검증 알고리즘 구상
- 전체 시스템 로직 설계

#### 🧠 정도영 — AI / Verification
- 서명 검증 알고리즘 구상
- 하드웨어 조립

#### 🔧 김민성 — Hardware
- 하드웨어 설계 및 조립
- 영상 제작 및 편집

#### ⚙️ 강상현 — System / Documentation
- 전체 시스템 로직 구상
- 보고서 작성

#### 📊 박경민 — Dataset / Research
- 데이터셋 수집 및 취합
- 서명 위조 관련 자료 조사

## 🛠 Tech Stack

- **Hardware**  
	NVIDIA Jetson Orin Nano · Camera · GPIO · LCD

- **AI / Deep Learning**  
	PyTorch · YOLOv5 · MobileNetV3-Small · ArcFace Loss · Cosine Similarity

- **Computer Vision**  
	OpenCV 

- **Development**  
	Python · CUDA · Git · GitHub
