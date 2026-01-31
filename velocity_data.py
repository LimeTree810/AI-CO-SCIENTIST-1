import pandas as pd
import numpy as np

# 1. 데이터 불러오기 (헤더 위치 조정 필요 시 header 옵션 수정)
file_path = 'total_2D_Data.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=6)

# 2. 컬럼명 공백 제거 (오류 방지)
df.columns = [c.strip() for c in df.columns]

# 3. 타겟 컬럼 설정 (delta P)
target_col = 'delta P [Pa]'

# 4. 데이터 확장 로직 수행
velocities = [1, 2, 3, 4, 5, 6]
new_rows = []

for idx, row in df.iterrows():
    # 원본 압력강하 값
    base_dp = row[target_col]
    
    # NaN 값 제외
    if pd.isna(base_dp):
        continue
        
    for v in velocities:
        # 기존 행 복사
        new_row = row.copy()
        
        # velocity 컬럼 추가 및 값 할당
        new_row['velocity'] = v
        
        # 새로운 Delta P 계산
        # 기준: 원본의 10배
        # 범위: 9.5배 ~ 10.5배 사이 (Uniform Distribution 사용)
        random_factor = np.random.uniform(9.5, 10.5)
        new_dp = base_dp * random_factor
        
        # 값 업데이트
        new_row[target_col] = new_dp
        
        new_rows.append(new_row)

# 5. 새로운 데이터프레임 생성 및 저장
expanded_df = pd.DataFrame(new_rows)
output_filename = 'velocity_data.xlsx'
expanded_df.to_excel(output_filename, index=False)

# 결과 확인
print(f"Original rows: {len(df)}")
print(f"Expanded rows: {len(expanded_df)}")
print(expanded_df[[target_col, 'velocity']].head(10))