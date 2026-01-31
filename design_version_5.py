import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, DotProduct, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

def load_data_design_params(file_path):
    print(f"[Info] 파일 로드 시도: {os.path.basename(file_path)}")
    if not os.path.exists(file_path):
        if os.path.exists(file_path + '.xlsx'): file_path += '.xlsx'
        elif os.path.exists(file_path + '.csv'): file_path += '.csv'
        else:
            print(f"[Error] 파일을 찾을 수 없습니다: {file_path}")
            return None, None

    try:
        if file_path.lower().endswith(('.xlsx', '.xls')):
            # Sheet1을 명시적으로 읽기
            df_raw = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
            header_row = 0
            for i in range(20):
                if i >= len(df_raw): break
                row_str = ' '.join([str(val).lower() for val in df_raw.iloc[i].values if pd.notna(val)])
                if ('s1' in row_str and 'height' in row_str) or ('s1_mm' in row_str) or ('fin_height' in row_str):
                    header_row = i
                    break
            df = pd.read_excel(file_path, sheet_name='Sheet1', header=header_row)
        else:
            df = pd.read_csv(file_path)
            if 's1' not in [str(c).lower() for c in df.columns]:
                 df = pd.read_csv(file_path, header=2)
    except Exception as e:
        print(f"[Error] 파일 읽기 실패: {e}")
        return None, None

    df.columns = [str(c).strip() for c in df.columns]
    cols = df.columns
    
    col_map = {}
    def find_col(keywords):
        for c in cols:
            if any(k in str(c).lower() for k in keywords): return c
        return None

    col_map['s1'] = find_col(['s1', 's1_mm'])
    col_map['height'] = find_col(['fin_height', 'fh', 'height'])
    col_map['spacing'] = find_col(['fin_spacing', 'fs', 'spacing'])
    col_map['q_flux'] = find_col(['q\'\'', 'flux', 'q_double', 'heat flux']) 
    col_map['dp'] = find_col(['delta p', 'dp', 'delp'])
    col_map['id'] = find_col(['sample', 'no.', 'index'])

    if not all([col_map['s1'], col_map['height'], col_map['spacing']]):
        print(f"[Warning] 필수 컬럼 누락: {file_path}")
        print(f"  발견된 컬럼: {cols.tolist()}")
        print(f"  S1: {col_map['s1']}, Height: {col_map['height']}, Spacing: {col_map['spacing']}")
        return None, None

    df_clean = df.copy()
    if col_map['dp'] and col_map['q_flux']:
        df_clean = df_clean.dropna(subset=[col_map['dp'], col_map['q_flux']], how='all')
    elif col_map['dp']:
        df_clean = df_clean.dropna(subset=[col_map['dp']])
        
    return df_clean, col_map

def find_best_kernel(X, y):
    noise_bounds = (1e-10, 1e1)
    kernels = [
        C(1.0) * Matern(length_scale=[1.0]*3, nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * RBF(length_scale=[1.0]*3) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * RationalQuadratic(alpha=0.1) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds),
        C(1.0) * (DotProduct() + Matern(nu=2.5)) + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds)
    ]
    best_score = -np.inf
    best_model = None
    best_name = ""
    names = ["Matern(2.5)", "RBF", "RationalQuad", "Composite"]
    
    for k, name in zip(kernels, names):
        gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, normalize_y=False, random_state= 85)  #반영하는 sample들을 바꾸고 싶으면 random_state 값을 바꾸세요
        scores = cross_val_score(gp, X, y, cv=5, scoring='r2')
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_model = gp
            best_name = name
            
    best_model.fit(X, y)
    # Fit된 커널(최적화 후 파라미터)을 함께 확인
    try:
        print(f"    [Kernel fitted] {best_model.kernel_}")
    except Exception:
        pass
    return best_model, best_name

def get_sensitivity(gp_model):
    try:
        k = gp_model.kernel_
        def get_ls(kern):
            if hasattr(kern, 'length_scale'): return kern.length_scale
            if hasattr(kern, 'k1'): 
                ret = get_ls(kern.k1); 
                if ret is not None: return ret
            if hasattr(kern, 'k2'):
                ret = get_ls(kern.k2); 
                if ret is not None: return ret
            return None
        ls = get_ls(k)
        if ls is None: ls = np.exp(k.theta[:3])
        ls = np.array(ls).flatten()[:3]
        if len(ls) < 3: ls = np.array([1.0, 1.0, 1.0])
        sens = 1.0 / ls
        return (sens / sens.sum()) * 100
    except: return np.array([33.3, 33.3, 33.3])

def run_auto_split_analysis(data_file):
    print(f"--- [Auto-Split Surrogate Model Analysis] ---")
    
    # 1. 단일 파일 로드
    df, cols = load_data_design_params(data_file)
    if df is None: return

    # 2. 데이터 분할 (Train 90%, Verify 10%)
    df_train, df_ver = train_test_split(df, test_size=0.1, random_state=85) #반영하는 sample들을 바꾸고 싶으면 random_state 값을 바꾸세요
    
    # -------------------------------------------------------------
    # [추가] 분할 상태 저장 및 출력
    # -------------------------------------------------------------
    if cols['id']:
        # ID 기준으로 정렬하여 저장
        train_ids = df_train[cols['id']].sort_values().values
        ver_ids = df_ver[cols['id']].sort_values().values
        
        # DataFrame 생성
        df_status = df[[cols['id']]].copy()
        df_status.columns = ['Sample No']
        df_status['Set Type'] = df_status['Sample No'].apply(lambda x: 'Verify' if x in ver_ids else 'Train')
        
        # 엑셀 저장
        status_file = 'data_split_status.xlsx'
        df_status.to_excel(status_file, index=False)
        print(f"\n[데이터 분할 정보]")
        print(f" - 전체: {len(df)}개 (Train: {len(df_train)}, Verify: {len(df_ver)})")
        print(f" - 검증용 샘플 번호: {ver_ids.tolist()}")
        print(f" - 상세 정보 파일 저장됨: {status_file}")
    else:
        print("\n[Warning] Sample No 컬럼이 없어 ID별 분할 정보를 저장할 수 없습니다.")
    
    # 3. 전처리
    X_train = df_train[[cols['s1'], cols['height'], cols['spacing']]].values
    X_ver = df_ver[[cols['s1'], cols['height'], cols['spacing']]].values
    
    scaler_X = StandardScaler().fit(X_train)
    X_train_sc = scaler_X.transform(X_train)
    X_ver_sc = scaler_X.transform(X_ver)
    
    # 4. 모델 학습
    targets = []
    if cols['q_flux']: targets.append(('q_flux', "Q'' [W/m^2]"))
    if cols['dp']: targets.append(('dp', "Delta P [Pa]"))
        
    models = {}
    scalers_y = {}
    
    for key, label in targets:
        print(f"\n[{label}] 모델 최적화 중...")
        y_train = df_train[cols[key]].values
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_sc = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        
        model, k_name = find_best_kernel(X_train_sc, y_train_sc)
        models[key] = model
        scalers_y[key] = scaler_y
        print(f" -> Best Kernel: {k_name}")

    # 5. 시각화 1: Parity Plot
    n_plots = len(targets)
    if n_plots > 0:
        fig1, axes1 = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1: axes1 = [axes1]
        # fig1.suptitle('Validation Parity Plots', fontsize=16)
        
        for i, (key, label) in enumerate(targets):
            ax = axes1[i]
            y_true = df_ver[cols[key]].values
            pred_sc = models[key].predict(X_ver_sc)
            pred = scalers_y[key].inverse_transform(pred_sc.reshape(-1, 1)).flatten()
            
            r2 = r2_score(y_true, pred)
            safe_y = np.where(y_true == 0, 1e-9, y_true)
            mape = np.mean(np.abs((y_true - pred) / safe_y)) * 100
            
            ax.scatter(y_true, pred, alpha=0.7, edgecolors='k', color='royalblue')
            ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            rmse = np.sqrt(mean_squared_error(y_true, pred))
            data_range = y_true.max() - y_true.min()
            if data_range == 0: data_range = 1e-9
            nrmse = (rmse / data_range) * 100
            # ax.set_title(f"{label}\nR2={r2:.3f}, NRMSE={nrmse:.2f}%, MAPE={mape:.1f}%")
            ax.set_xlabel(f"Actual {label}")
            ax.set_ylabel(f"Predicted {label}")
            ax.grid(True, alpha=0.3)
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # 6. 시각화 2: Feature Importance
    if n_plots > 0:
        fig2, axes2 = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1: axes2 = [axes2]
        # fig2.suptitle('Sensitivity analysis', fontsize=16)
        feat_names = ['S1', 'Fin Height', 'Fin Spacing']
        
        for i, (key, label) in enumerate(targets):
            sens = get_sensitivity(models[key])
            sns.barplot(x=feat_names, y=sens, ax=axes2[i], palette='viridis')
            # axes2[i].set_title(f"{label} Sensitivity")
            axes2[i].set_ylabel("Importance (%)")
            axes2[i].set_ylim(0, 100)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # 7. 시각화 3: 3D Response Surfaces (Combined Figure)
    pairs = [(0, 1, 'S1', 'Fin Height'), 
             (0, 2, 'S1', 'Fin Spacing'), 
             (1, 2, 'Fin Height', 'Fin Spacing')]
    
    ranges = [
        (X_train[:, 0].min(), X_train[:, 0].max()),
        (X_train[:, 1].min(), X_train[:, 1].max()),
        (X_train[:, 2].min(), X_train[:, 2].max())
    ]
    
    # 2개의 타겟이 있으면 2행 x 3열, 1개면 1행 x 3열
    n_targets = len(targets)
    fig = plt.figure(figsize=(18, 5 * n_targets))
    # fig.suptitle('Response Surfaces', fontsize=16)
    
    plot_idx = 1
    for key, label in targets:
        for i, (idx1, idx2, name1, name2) in enumerate(pairs):
            ax = fig.add_subplot(n_targets, 3, plot_idx, projection='3d')
            
            x1_grid = np.linspace(ranges[idx1][0], ranges[idx1][1], 30)
            x2_grid = np.linspace(ranges[idx2][0], ranges[idx2][1], 30)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            
            input_grid = np.zeros((X1.ravel().shape[0], 3))
            input_grid[:, idx1] = X1.ravel()
            input_grid[:, idx2] = X2.ravel()
            
            missing = [x for x in [0,1,2] if x not in [idx1, idx2]][0]
            input_grid[:, missing] = np.mean(X_train[:, missing])
            
            input_sc = scaler_X.transform(input_grid)
            pred_sc = models[key].predict(input_sc)
            pred = scalers_y[key].inverse_transform(pred_sc.reshape(-1, 1)).reshape(X1.shape)
            
            surf = ax.plot_surface(X1, X2, pred, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel(name1)
            ax.set_ylabel(name2)
            ax.set_zlabel(label)
            # ax.set_title(f'{label}: {name1} vs {name2}')
            
            plot_idx += 1
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # 스크립트 파일의 디렉토리를 기준으로 데이터 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "total_2D_Data")
    run_auto_split_analysis(data_file)