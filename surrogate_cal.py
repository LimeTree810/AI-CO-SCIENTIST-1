import pandas as pd
import numpy as np
import os
import warnings

# design_version_5.py와 동일한 라이브러리 사용
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, DotProduct, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 데이터 로드 및 전처리 (design_version_5.py 로직)
# =============================================================================
def load_data(file_path):
    print(f"[System] 데이터 로드 중: {file_path}")
    
    if not os.path.exists(file_path):
        if os.path.exists(file_path + '.xlsx'): file_path += '.xlsx'
        elif os.path.exists(file_path + '.csv'): file_path += '.csv'
        else: return None, None

    try:
        # 헤더 자동 탐색
        if file_path.lower().endswith(('.xlsx', '.xls')):
            # Sheet1 먼저 시도
            try:
                df_raw = pd.read_excel(file_path, sheet_name='Sheet1', header=None, nrows=30)
            except:
                df_raw = pd.read_excel(file_path, header=None, nrows=30)
            
            header_row = 0
            for i in range(min(30, len(df_raw))):
                row_str = ' '.join([str(val).lower() for val in df_raw.iloc[i].values if pd.notna(val)])
                # 더 유연한 헤더 검색 - 여러 키워드가 함께 있는 행 찾기
                keywords_found = sum([1 for kw in ['s1', 'height', 'spacing', 'flux', 'delta', 'sample'] if kw in row_str])
                if keywords_found >= 3:  # 3개 이상의 키워드가 있으면 헤더로 간주
                    header_row = i
                    print(f"[Info] Header found at row {i}")
                    break
            
            # Sheet1부터 시도
            try:
                df = pd.read_excel(file_path, sheet_name='Sheet1', header=header_row)
            except:
                df = pd.read_excel(file_path, header=header_row)
        else:
            df = pd.read_csv(file_path)
            # CSV 헤더 확인
            col_str = ' '.join([str(c).lower() for c in df.columns])
            if not any(keyword in col_str for keyword in ['s1', 'height', 'spacing']):
                # 처음 몇 행을 확인
                for i in range(min(10, len(df))):
                    row_str = ' '.join([str(val).lower() for val in df.iloc[i].values if pd.notna(val)])
                    keywords_found = sum([1 for kw in ['s1', 'height', 'spacing'] if kw in row_str])
                    if keywords_found >= 2:
                        df = pd.read_csv(file_path, header=i)
                        print(f"[Info] CSV header found at row {i}")
                        break
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return None, None

    df.columns = [str(c).strip() for c in df.columns]
    
    print(f"[Debug] Available columns: {df.columns.tolist()}")
    
    # 컬럼 찾기 함수 (안전하게 처리)
    def find_col(keywords, columns):
        for kw in keywords:
            matches = [c for c in columns if kw in c.lower()]
            if matches:
                return matches[0]
        return None
    
    col_map = {}
    col_map['s1'] = find_col(['s1_mm', 's1', 'transverse'], df.columns)
    col_map['h'] = find_col(['fin_height_fh_mm', 'height', 'fin_height', 'fh_mm', 'hf'], df.columns)
    col_map['s'] = find_col(['fin_spacing_fs_mm', 'spacing', 'fin_spacing', 'fs_mm'], df.columns)
    col_map['q_flux'] = find_col(["q''", 'flux', 'heat flux', 'q [w/m'], df.columns)
    col_map['q_total'] = find_col(['q [w]', 'heat transfer', 'q_w'], df.columns)
    col_map['dp'] = find_col(['delta p', 'dp [pa]', 'pressure drop'], df.columns)
    
    # 필수 컬럼 확인 (s1, h, s, dp, q_flux는 필수, q_total은 선택)
    required = ['s1', 'h', 's', 'q_flux', 'dp']
    missing = [k for k in required if col_map.get(k) is None]
    if missing:
        print(f"[Error] Missing required columns: {missing}")
        print(f"[Info] Available columns: {df.columns.tolist()}")
        return None, None
    
    print(f"[Info] Column mapping: {col_map}")
    # Q_total이 있으면 포함해서 dropna, 없으면 제외
    drop_cols = [col_map['dp'], col_map['q_flux']]
    if col_map['q_total']:
        drop_cols.append(col_map['q_total'])
    df = df.dropna(subset=drop_cols)
    return df, col_map

# =============================================================================
# 2. 모델 학습 (최적 커널 탐색 포함)
# =============================================================================
def train_surrogate_model(df, col_map):
    print("[System] 대리 모델 학습 시작 (최적 커널 탐색 포함)...")
    
    X = df[[col_map['s1'], col_map['h'], col_map['s']]].values
    scaler_X = StandardScaler().fit(X)
    X_sc = scaler_X.transform(X)
    
    models = {}
    scalers_y = {}
    
    # design_version_5.py와 동일한 커널 후보군
    noise_bounds = (1e-10, 1e1)
    kernels = [
        C(1.0) * Matern(length_scale=[1.0]*3, nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * RBF(length_scale=[1.0]*3) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * RationalQuadratic(alpha=0.1) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * (DotProduct() + Matern(nu=2.5)) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds)
    ]
    kernel_names = ["Matern(2.5)", "RBF", "RationalQuad", "Composite"]

    # 학습할 타겟 결정 (Q_total이 있으면 포함)
    targets = ['q_flux', 'dp']
    if col_map.get('q_total'):
        targets.append('q_total')
    
    for target in targets:
        y = df[col_map[target]].values
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        y_sc = scaler_y.transform(y.reshape(-1, 1)).flatten()
        
        best_score = -np.inf
        best_model = None
        best_name = ""
        
        # 교차 검증으로 최적 커널 선택
        for k, name in zip(kernels, kernel_names):
            gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, normalize_y=False, random_state=85)
            scores = cross_val_score(gp, X_sc, y_sc, cv=5, scoring='r2')
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_model = gp
                best_name = name
        
        # 전체 데이터로 최종 학습
        best_model.fit(X_sc, y_sc)
        models[target] = best_model
        scalers_y[target] = scaler_y
        print(f"   -> [{target.upper()}] 모델 완료 (Best Kernel: {best_name}, CV R2: {best_score:.4f})")
        
    return models, scaler_X, scalers_y

# =============================================================================
# 3. 사용자 예측 인터페이스
# =============================================================================
def predict_user_input(models, scaler_X, scalers_y, s1, fh, fs):
    """
    사용자 입력값(S1, FH, FS)을 받아 Q'', Q, Delta P를 예측하여 반환
    """
    # 입력값 스케일링
    input_vec = np.array([[s1, fh, fs]])
    input_sc = scaler_X.transform(input_vec)
    
    # 예측 (Mean & Std)
    q_flux_mu_sc, q_flux_std_sc = models['q_flux'].predict(input_sc, return_std=True)
    dp_mu_sc, dp_std_sc = models['dp'].predict(input_sc, return_std=True)
    
    # 역변환 (StandardScaler 복원)
    # Mean 복원
    q_flux_pred = scalers_y['q_flux'].inverse_transform(q_flux_mu_sc.reshape(-1, 1))[0][0]
    dp_pred = scalers_y['dp'].inverse_transform(dp_mu_sc.reshape(-1, 1))[0][0]
    
    # Std 복원 (scale만 곱함)
    q_flux_std = q_flux_std_sc[0] * scalers_y['q_flux'].scale_[0]
    dp_std = dp_std_sc[0] * scalers_y['dp'].scale_[0]
    
    results = {
        'q_flux': (q_flux_pred, q_flux_std),
        'dp': (dp_pred, dp_std)
    }
    
    # Q_total이 있으면 예측
    if 'q_total' in models:
        q_total_mu_sc, q_total_std_sc = models['q_total'].predict(input_sc, return_std=True)
        q_total_pred = scalers_y['q_total'].inverse_transform(q_total_mu_sc.reshape(-1, 1))[0][0]
        q_total_std = q_total_std_sc[0] * scalers_y['q_total'].scale_[0]
        results['q_total'] = (q_total_pred, q_total_std)
    
    return results

if __name__ == "__main__":
    # 1. 학습 데이터 파일 설정
    file_path = "total_2D_Data.xlsx"
    
    # 2. 모델 학습
    df, col_map = load_data(file_path)
    if df is not None:
        models, scaler_X, scalers_y = train_surrogate_model(df, col_map)
        
        print("\n" + "="*60)
        print("          🛠️  설계 변수 성능 예측기 (Ready)  🛠️")
        print("="*60)
        
        # 3. 사용자 입력 및 예측 (예시)
        # ---------------------------------------------------------
        # [사용법] 아래 변수에 원하시는 값을 입력하고 실행하세요.
        # ---------------------------------------------------------
        user_s1 = 54.60655   # S1 (mm)
        user_fh = 6.628    # Fin Height (mm)
        user_fs = 2.778     # Fin Spacing (mm)
        
        results = predict_user_input(models, scaler_X, scalers_y, user_s1, user_fh, user_fs)
        
        print(f"\n[입력 변수]")
        print(f" - S1          : {user_s1} mm")
        print(f" - Fin Height  : {user_fh} mm")
        print(f" - Fin Spacing : {user_fs} mm")
        
        print(f"\n[예측 결과 (Mean ± Uncertainty)]")
        q_flux_val, q_flux_err = results['q_flux']
        dp_val, dp_err = results['dp']
        print(f" 🔥 Heat Flux (Q'')  : {q_flux_val:,.2f} ± {q_flux_err:.2f} W/m^2")
        
        if 'q_total' in results:
            q_total_val, q_total_err = results['q_total']
            print(f" 🔥 Heat Transfer (Q): {q_total_val:,.2f} ± {q_total_err:.2f} W")
        
        print(f" 💨 Delta P         : {dp_val:.4f} ± {dp_err:.4f} Pa")
        print(f" 📊 Efficiency (Q''/dP): {q_flux_val/dp_val:.4f}")
        print("="*60)
        
    else:
        print("데이터 파일을 찾을 수 없어 모델을 학습할 수 없습니다.")