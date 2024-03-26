# 🩻 MEDI-KNEE
> 무릎 MRI 진단 보조 서비스  

<p align = "center"> <img src= "https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-13/assets/49268298/3abc596a-d15d-4990-8cc5-2e3e3634dc96"><p/>


## 서비스 데모영상

<img src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-13/assets/70832671/447950c6-0ae2-414c-bd16-015983d9c60b">

- 서비스 링크: [mediknee.site](http://mediknee.site)
- 발표 영상: [발표 영상 링크](https://www.youtube.com/watch?v=7XdgcU41urQ)

<hr>

## 프로젝트 소개
의료진을 위한 MRI 영상 분석 서비스 MEDI-KNEE는 무릎 질병 진단을 보조하는 서비스로, 질병 진단 확률, MRI 슬라이드 중요도 그래프, 이상 부위 검출을 보조 및 제공합니다.
<br>



## 기획 배경
- 무릎 통증을 호소하는 무릎 관절증 환자의 지속적인 증가
- OECD 평균보다 적은 의사 수 → 의사 1인 당 업무 부담 증가
- 복잡한 관절 중 하나인 무릎 관절의 MRI 진단을 보조할 수 있는 서비스 개발  
<p align = "center"> <img src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-13/assets/49268298/4a3629cd-c4be-4927-a674-c1a993d4c628"> <p/>
<br>

## 데이터셋
- 무릎 질병 진단 모형 학습을 위해 스탠포드 대학교에서 제작한 Knee MRI 데이터셋을 활용했습니다.
- [MRNet 데이터셋 링크](https://stanfordmlgroup.github.io/competitions/mrnet/)

<br>

## 모델

- MRI 데이터는 Axial, Coronal, Sagittal 3개의 축으로 구성되어 있습니다. 따라서 질병이 있는지 없는지 판단하기 위해서는 각 축별로 한 개의 모델, 총 3개의 모델이 필요합니다.
- 저희는 총 3가지 질병 (Abnormal, ACL, Meniscus)에 대한 진단을 제공하는 것을 목적으로 하고, 각 질병 별 3개의 축이 있으므로 총 9개의 모델이 필요합니다.

    - Task: Abnormal(비정상), ACL(전방십자인대 파열), Meniscus(반월상 연골 파열)
    - Plane: Axial, Coronal, Sagittal

- 또한, 각각의 질병에 대하여, 3개의 모델이 예측한 결과값을 Fusion하여 하나의 확률값을 도출하는 과정이 필요합니다. 따라서 3개의 Fusion Model이 추가로 필요합니다.
- 이 과정을 그림으로 표현하면 아래 사진과 같습니다.

<p align = "center"> <img src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-13/assets/49268298/522ec459-9217-4093-aa58-21b4940ccf3c"> <p/>

<br>

## XAI: GradCAM, CAM-Score
- XAI를 위해 Grad-CAM, CAM-Score를 활용하여 모델의 판단 과정을 설명할 수 있습니다.

### Grad-CAM
- MRI 이미지에서 병변이 있는 부위, 이상 부위를 표현하여 진단을 보조합니다

### CAM-Score
- CAM-Score란?​
    - MRI 슬라이드별로 중요한 정도를 의미합니다.​
    - 질병이 보이는 슬라이드, 혹은 질병과 연관이 있는 슬라이드의 경우 CAM-Score가 높습니다.​

- 왜 CAM-Score가 필요한가요?​
    - CAM-score가 높은 슬라이드를 우선적으로 보여줌으로써, MRI 판독의 시간을 줄이고 모델의 판독 이유를 설명할 수 있습니다.

<p align = "center"> <img src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-13/assets/70832671/11b0570f-6cfc-4b5e-b651-5b120c70473b"> <p/>
<br>

## 서비스 파이프라인

<img src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-13/assets/78347296/5635bcd8-b7a0-40c0-8ca8-cfcc19cd0837">

<br>

## 서비스 구성
- File upload
    - DICOM 파일을 업로드하거나 샘플 데이터를 업로드할 수 있는 화면입니다.
- Result
    - 환자의 질병 진단 결과, 그리고 Abnormal / ACL / Meniscus일 확률이 얼마나 되는지 결과를 보여주는 화면입니다.
    - 3개의 축별로 모델이 가장 중요하다고 생각한 슬라이드를 보여줍니다.
    - Grad-CAM을 시각화하여 모델이 중요하다고 생각한 부분을 확인할 수 있습니다.
- Plane 별 전체 data 확인
    - 슬라이드 별 중요도를 확인할 수 있습니다.
- 결과에 따른 report 생성
    - 환자의 진단 결과를 한눈에 알 수 있는 리포트를 다운로드할 수 있습니다.
<p align = "center"> <img src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-13/assets/49268298/d778ad54-d24a-4d1c-adac-7093596443ca"> <p/>


## Team SMiLE

|    | 김영일_T6030 | 안세희_T6094 | 유한준_T6106 | 윤일호_T6110 | 이재혁_T6132 |
|---|        ---        |        ---        |        ---        |          ---      |        ---        |
|Github|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/patrashu)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/seheeAn)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lukehanjun)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yuniroro)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/NewP1)|
|E-mail|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=qhdrmfdl123@gmail.com)](mailto:qhdrmfdl123@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=imash0525@gmail.com)](mailto:imash0525@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=lukehanjun@gmail.com)](mailto:lukehanjun@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=ilho7159@gmail.com)](mailto:ilho7159@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=jaehyuk712@gmail.com)](mailto:jaehyuk712@gmail.com)|