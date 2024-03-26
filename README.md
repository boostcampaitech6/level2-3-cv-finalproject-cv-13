# 🩻 MEDI-KNEE
> 의료진을 위한 MRI 영상 분석 서비스  

<p align = "center"> <img src= "https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-13/assets/49268298/3abc596a-d15d-4990-8cc5-2e3e3634dc96"><p/>

## 서비스 파이프라인

<img src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-13/assets/78347296/5635bcd8-b7a0-40c0-8ca8-cfcc19cd0837">

[mediknee.site](http://mediknee.site)

## 서비스 데모영상
<br>
(서비스 화면 GIF)
<br>

[발표영상](발표영상링크)

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
- MRNet 데이터셋: [link](https://stanfordmlgroup.github.io/competitions/mrnet/)

## 모델

9개 모델 = 3개 Task x 3개 plane

- Task: Abnormal, Acl, Meniscus
- Plane: Axial, Coronal, Sagittal

각각의 Task에 대하여, 3개의 plane에 대한 예측값을 Fusion하여 하나의 확률값을 가짐
<p align = "center"> <img src="https://github.com/boostcampaitech6/level2-3-cv-finalproject-cv-13/assets/49268298/522ec459-9217-4093-aa58-21b4940ccf3c"> <p/>

<br>

## GradCAM, CAM-score

- GradCAM Heatmap, 이상 부위를 표시하여 진단을 보조
- CAM-score, 각각의 점수는 해당 슬라이드의 중요도를 상대적으로 표현
<p align = "center"> <img src=""> <p/>
<br>

## Team SMiLE

|    | 김영일_T6030 | 안세희_T6094 | 유한준_T6106 | 윤일호_T6110 | 이재혁_T6132 |
|---|        ---        |        ---        |        ---        |          ---      |        ---        |
|Github|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/patrashu)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/seheeAn)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lukehanjun)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yuniroro)|[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/NewP1)|
|E-mail|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=qhdrmfdl123@gmail.com)](mailto:qhdrmfdl123@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=imash0525@gmail.com)](mailto:imash0525@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=lukehanjun@gmail.com)](mailto:lukehanjun@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=ilho7159@gmail.com)](mailto:ilho7159@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=jaehyuk712@gmail.com)](mailto:jaehyuk712@gmail.com)|