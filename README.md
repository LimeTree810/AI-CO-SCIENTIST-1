# AI CO SCIENTIST - CFD Surrogate Model

2D CFD 다공성 매체 대리 모델 프로젝트

## 📁 데이터 파일

엑셀 데이터 파일은 용량이 커서 Google Drive에 별도로 저장되어 있습니다.

**[📥 데이터 다운로드 (Google Drive)](https://drive.google.com/drive/folders/1IFam-ghW_FbsbSk-fIK7qedPJ4X3g9SA?usp=sharing)**

### 포함된 데이터:
- `total_2D_Data.xlsx` - 전체 2D CFD 데이터
- `training_data_21-100_samples.xlsx` - 학습 데이터 (샘플 21-100)
- `validation_data_1-20_samples.xlsx` - 검증 데이터 (샘플 1-20)
- `constrained_LHS_100K.xlsx` - 제약 조건이 적용된 LHS 샘플링 결과
- `optimum_design_variables.xlsx` - 최적 설계 변수
- `velocity_data.xlsx` - 속도 데이터
- 기타 CSV 및 이미지 파일

## 🐍 Python 코드 파일

### 최적화 및 설계
- `design_version_5.py` - 설계 최적화 버전 5
- `optimization_version_3.py` - 최적화 알고리즘 버전 3
- `optimum_porous_media_finding.py` - 최적 다공성 매체 파라미터 탐색

### 대리 모델 및 계산
- `surrogate_cal.py` - 대리 모델 계산
- `porous_from_design_integrated.py` - 다공성 매체 설계 통합 모델
- `velocity_data.py` - 속도 데이터 처리

## 사용 방법

1. Google Drive에서 필요한 데이터 파일 다운로드
2. Python 스크립트와 같은 디렉토리에 데이터 파일 배치
3. Python 스크립트 실행

## 요구사항

주요 Python 패키지:
- numpy
- pandas
- scikit-learn
- matplotlib
- scipy
