#!/usr/bin/env python3
"""
================================================================================
Integrated batch porous-parameter generator (single-file)
================================================================================

This script merges:
- cfd_porous_calc_modified.py  (physics/correlations + calculate_porous_parameters)
- batch_porous_from_design.py  (batch CSV runner + coupled-constraint diagnostics)

Goal
----
Given sampled *design variables* per row (S1, fin height, fin spacing),
compute porous-media parameters (inv_K, C2, etc.) using Nir(1991)-based
pressure-drop correlation and a Darcy–Forchheimer least-squares fit over
a velocity range.

You no longer need to keep cfd_porous_calc_modified.py separately for batch runs.

Inputs (CSV)
------------
Required columns (any of these aliases are accepted):
- S1:  S1_mm / S1 / s1_mm / s1
- fin height:  fin_height_fh_mm / fh_mm / fin_height / fh / hf_mm / hf
- fin spacing: fin_spacing_fs_mm / fs_mm / fin_spacing / Fs_mm / Fs / fs

Outputs (CSV)
-------------
Original design vars + porous outputs + useful intermediate geometry:
- Viscous_Resistance_1_m2, Inertial_Resistance_1_m, K_m2, R2_fit
- porosity (epsilon), a_fs_1_m, area_ratio, sigma, Re_Dc, dP_total_Pa, dP_per_L_Pa_m
- coupled-constraint diagnostics if enabled

Notes
-----
- Runtime scales ~ O(N_rows * n_points). For 100k rows with n_points=50,
  you will do ~5 million ΔP evaluations. Start with n_points=15~25 for a
  first mapping, then refine if needed.

Reference
---------
Nir, A. (1991). Heat Transfer and Friction Factor Correlations for Crossflow over
Staggered Finned Tube Banks. Heat Transfer Engineering, 12(1), 43–58.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd


# =============================================================================
# 1) Air properties (simple correlations)
# =============================================================================

def air_properties(T_celsius: float) -> Dict[str, float]:
    """
    Approximate air properties at ~1 atm.
    Returns: rho, mu, k, Pr, nu, T_K
    """
    T_K = T_celsius + 273.15

    # Density from ideal gas (approx at 1 atm)
    P = 101325.0  # Pa
    R = 287.05    # J/(kg·K)
    rho = P / (R * T_K)

    # Dynamic viscosity (Sutherland-like linear approximation used in your code)
    # (kept as-is from your file for consistency)
    mu = (1.716e-5 * (T_K / 273.15) ** 1.5) * (273.15 + 111.0) / (T_K + 111.0)

    # Thermal conductivity (simple linear approx; kept consistent)
    k = 0.0241 + 7.0e-5 * T_celsius

    # Prandtl number
    cp = 1006.0
    Pr = mu * cp / k

    # Kinematic viscosity
    nu = mu / rho

    return {"rho": rho, "mu": mu, "k": k, "Pr": Pr, "nu": nu, "T_K": T_K}


# =============================================================================
# 2) Geometry utilities
# =============================================================================

@dataclass
class AnnularFinGeometry:
    """Annular Fin geometry (SI units in meters)."""
    Dc: float          # tube OD [m]
    delta_f: float     # fin thickness [m]
    s1: float          # transverse pitch [m]
    Fs: float          # fin spacing [m]
    hf: float          # fin height [m]
    pitch_ratio: float = 1.0  # s1/s2
    N: int = 4

    def __post_init__(self):
        self.Do = self.Dc + 2.0 * self.hf
        self.Fp = self.Fs + self.delta_f
        self.s2 = self.s1 / self.pitch_ratio

        # Porosity (annular zone)
        self.epsilon = 1.0 - (self.delta_f / self.Fp)

        # Minimum free-flow area ratio (sigma)
        # sigma = (s1 - Dc - 2*hf*(delta_f/Fp)) / s1
        self.sigma = (self.s1 - self.Dc - 2.0 * self.hf * (self.delta_f / self.Fp)) / self.s1

        self._calculate_areas()

    def _calculate_areas(self):
        # Fin surface area per pitch (both faces + tip)
        # A_fin = 2*(pi/4)*(Do^2 - Dc^2) + pi*Do*delta_f
        self.A_fin = (2.0 * (math.pi / 4.0) * (self.Do**2 - self.Dc**2) +
                      math.pi * self.Do * self.delta_f)

        # Exposed tube area between fins
        self.A_base = math.pi * self.Dc * self.Fs

        self.A_total = self.A_fin + self.A_base

        # Bare tube area over one pitch
        self.A_bare = math.pi * self.Dc * self.Fp

        # Area ratio (key Nir factor)
        self.area_ratio = self.A_total / self.A_bare

        # Annular-zone volume over one pitch
        self.V_annular = (math.pi / 4.0) * (self.Do**2 - self.Dc**2) * self.Fp

        # Specific surface area [1/m]
        self.a_fs = self.A_total / self.V_annular

    def get_porous_thickness(self) -> float:
        return self.hf

    def get_flow_depth(self) -> float:
        return self.s2 * self.N


# =============================================================================
# 3) Nir (1991) friction factor correlation + ΔP
# =============================================================================

def nir_friction_factor(Re_Dc: float, s1_Dc: float, area_ratio: float) -> float:
    """
    Nir (1991):
        f_N = 1.1 * Re^(-0.25) * (S1/Dc)^(-0.4) * (A_tot/A_bare)^(0.15)
    """
    term1 = 1.1 * (Re_Dc ** -0.25)
    term2 = (s1_Dc) ** -0.4
    term3 = (area_ratio) ** 0.15
    return term1 * term2 * term3


def calculate_pressure_drop(
    v_inlet: float,
    geom: AnnularFinGeometry,
    air_props: Dict[str, float]
) -> Tuple[float, float, float]:
    """
    Compute total ΔP across N rows using Nir correlation.
        ΔP_total = N * f_N * (rho * v_max^2) / 2
    Also returns ΔP/L and Re_Dc based on v_max.

    Returns: (dP_total_Pa, dP_per_L_Pa_m, Re_Dc)
    """
    rho = air_props["rho"]
    mu = air_props["mu"]

    # max velocity at minimum area
    v_max = v_inlet / geom.sigma

    # Reynolds based on tube OD
    Re_Dc = (rho * v_max * geom.Dc) / mu

    fN = nir_friction_factor(Re_Dc=Re_Dc, s1_Dc=(geom.s1 / geom.Dc), area_ratio=geom.area_ratio)

    dP_total = geom.N * fN * (rho * v_max**2) / 2.0

    # flow depth in meters
    L = geom.get_flow_depth()
    dP_per_L = dP_total / L

    return dP_total, dP_per_L, Re_Dc


# =============================================================================
# 4) Darcy–Forchheimer multipoint fitting to infer inv_K and C2
# =============================================================================

def fit_darcy_forchheimer_multipoint(
    geom: AnnularFinGeometry,
    air_props: Dict[str, float],
    v_range: Tuple[float, float] = (0.5, 3.5),
    n_points: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate ΔP/L across v_points and fit:
        Y(v) = ΔP/L = A*v + B*v^2
    Then map:
        A = mu/K  => inv_K = 1/K = A/mu
        B = (C2*rho)/2 => C2 = 2B/rho
    """
    rho = air_props["rho"]
    mu = air_props["mu"]

    v_min, v_max = v_range
    v_points = np.linspace(v_min, v_max, n_points)

    Y_points = []
    Re_points = []

    for v in v_points:
        _, Y, Re = calculate_pressure_drop(v, geom, air_props)
        Y_points.append(Y)
        Re_points.append(Re)

    v_points = np.asarray(v_points)
    Y_points = np.asarray(Y_points)
    Re_points = np.asarray(Re_points)

    # least squares for Y = [v, v^2] [A,B]^T
    X = np.column_stack([v_points, v_points**2])
    coeffs, residuals, rank, s = np.linalg.lstsq(X, Y_points, rcond=None)
    A, B = float(coeffs[0]), float(coeffs[1])

    inv_K = A / mu
    K = mu / A if A != 0 else np.inf
    C2 = 2.0 * B / rho

    Y_fit = A * v_points + B * v_points**2
    SS_res = float(np.sum((Y_points - Y_fit) ** 2))
    SS_tot = float(np.sum((Y_points - np.mean(Y_points)) ** 2))
    R2 = 1.0 - (SS_res / SS_tot) if SS_tot > 0 else np.nan

    return {
        "A": A,
        "B": B,
        "inv_K": inv_K,
        "C2": C2,
        "K": K,
        "R_squared": R2,
        "residual_sum": float(residuals[0]) if len(residuals) > 0 else 0.0,
        "v_points": v_points,
        "Y_points": Y_points,
        "Y_fit": Y_fit,
        "Re_points": Re_points,
        "Re_min": float(np.min(Re_points)),
        "Re_max": float(np.max(Re_points)),
    }


# =============================================================================
# 5) Optional: Briggs & Young heat-transfer correlation (kept for completeness)
# =============================================================================

def briggs_young_heat_transfer(
    v_inlet: float,
    geom: AnnularFinGeometry,
    air_props: Dict[str, float]
) -> float:
    """
    Briggs & Young (1963):
        Nu = 0.134 * Re^0.681 * Pr^(1/3) * (Fs/hf)^0.2 * (Fs/delta_f)^0.113
    Returns h [W/(m^2·K)] based on tube OD for scaling (consistent with your file).
    """
    rho = air_props["rho"]
    mu = air_props["mu"]
    k = air_props["k"]
    Pr = air_props["Pr"]

    v_max = v_inlet / geom.sigma
    Re_Dc = (rho * v_max * geom.Dc) / mu

    Nu = (0.134 * (Re_Dc ** 0.681) * (Pr ** (1.0 / 3.0)) *
          ((geom.Fs / geom.hf) ** 0.2) *
          ((geom.Fs / geom.delta_f) ** 0.113))

    return float(Nu * k / geom.Dc)


# =============================================================================
# 6) Main compute: design -> porous parameters
# =============================================================================

def calculate_porous_parameters(
    Fs_mm: float,            # fin spacing [mm]
    hf_mm: float,            # fin height [mm]
    T_celsius: float,        # ambient temperature [°C]
    v_design: float,         # design inlet velocity [m/s]
    Dc_mm: float = 24.0,     # tube OD [mm]
    delta_f_mm: float = 0.5, # fin thickness [mm]
    s1_mm: float = 55.333,   # transverse pitch [mm]
    pitch_ratio: float = 1.0,
    N: int = 4,
    v_range: Tuple[float, float] = (0.5, 3.5),
    n_points: int = 50
) -> Dict[str, Any]:
    """
    Compute all porous parameters and helpful intermediates.

    Returns a dict with keys:
      - "porous": inv_K, C2, K, R_squared, ...
      - "geometry": epsilon, sigma, a_fs, area_ratio, thickness, flow_depth, ...
      - "design_point": ΔP at v_design, Re_Dc, h_fs, etc.
    """
    # mm -> m
    Dc = Dc_mm / 1000.0
    delta_f = delta_f_mm / 1000.0
    s1 = s1_mm / 1000.0
    Fs = Fs_mm / 1000.0
    hf = hf_mm / 1000.0

    air = air_properties(T_celsius)

    geom = AnnularFinGeometry(
        Dc=Dc, delta_f=delta_f, s1=s1, Fs=Fs, hf=hf,
        pitch_ratio=pitch_ratio, N=N
    )

    porous_fit = fit_darcy_forchheimer_multipoint(
        geom=geom, air_props=air, v_range=v_range, n_points=n_points
    )

    dP_total, dP_per_L, Re_Dc = calculate_pressure_drop(v_inlet=v_design, geom=geom, air_props=air)
    h_fs = briggs_young_heat_transfer(v_inlet=v_design, geom=geom, air_props=air)

    return {
        "porous": porous_fit,
        "geometry": {
            "epsilon": geom.epsilon,
            "sigma": geom.sigma,
            "area_ratio": geom.area_ratio,
            "a_fs": geom.a_fs,
            "porous_thickness_m": geom.get_porous_thickness(),
            "flow_depth_m": geom.get_flow_depth(),
            "Dc_m": geom.Dc,
            "Do_m": geom.Do,
            "Fp_m": geom.Fp,
            "s1_m": geom.s1,
            "s2_m": geom.s2,
        },
        "design_point": {
            "v_design_m_s": v_design,
            "Re_Dc": Re_Dc,
            "dP_total_Pa": dP_total,
            "dP_per_L_Pa_m": dP_per_L,
            "h_fs_W_m2K": h_fs,
        },
    }


# =============================================================================
# 7) Batch runner (CSV -> CSV)
# =============================================================================

def infer_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)

    def pick(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        # also try case-insensitive matches
        lower = {c.lower(): c for c in cols}
        for c in candidates:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    s1_col = pick(["S1_mm", "S1", "s1_mm", "s1"])
    fh_col = pick(["fin_height_fh_mm", "fh_mm", "fin_height", "fh", "hf_mm", "hf"])
    fs_col = pick(["fin_spacing_fs_mm", "fs_mm", "fin_spacing", "Fs_mm", "Fs", "fs"])

    if s1_col is None or fh_col is None or fs_col is None:
        raise ValueError(
            "Input CSV must contain S1 + fin-height + fin-spacing columns.\n"
            f"Found columns: {cols}\n"
            "Accepted names:\n"
            "  - S1: S1_mm / S1 / s1_mm / s1\n"
            "  - fin height: fin_height_fh_mm / fh_mm / fin_height / fh / hf_mm / hf\n"
            "  - fin spacing: fin_spacing_fs_mm / fs_mm / fin_spacing / Fs_mm / Fs / fs"
        )
    return {"s1": s1_col, "fh": fh_col, "fs": fs_col}


def fh_upper_from_s(s_mm: float, td_mm: float = 24.0, margin_mm: float = 0.4) -> float:
    """
    Coupled constraint (Form B + outside -0.4 mm):
        fh <= 0.5*(s/sqrt(2) - td) - margin
    where td is tube diameter (fixed 24 mm in your use case).
    """
    return 0.5 * (s_mm / sqrt(2.0) - td_mm) - margin_mm


def run_batch(
    df: pd.DataFrame,
    *,
    T_celsius: float,
    v_design: float,
    Dc_mm: float,
    delta_f_mm: float,
    pitch_ratio: float,
    N: int,
    v_min: float,
    v_max: float,
    n_points: int,
    check_constraint: bool,
    progress_every: int,
) -> pd.DataFrame:
    colmap = infer_columns(df)
    s1_col, fh_col, fs_col = colmap["s1"], colmap["fh"], colmap["fs"]

    out_rows: List[Dict[str, Any]] = []
    n = len(df)

    for i, row in enumerate(df.itertuples(index=False), start=1):
        try:
            s1_mm = float(getattr(row, s1_col))
            fh_mm = float(getattr(row, fh_col))
            fs_mm = float(getattr(row, fs_col))

            fh_up = fh_upper_from_s(s1_mm, td_mm=Dc_mm, margin_mm=0.4)
            constraint_ok = (fh_mm <= fh_up) if check_constraint else True

            res = calculate_porous_parameters(
                Fs_mm=fs_mm,
                hf_mm=fh_mm,
                T_celsius=T_celsius,
                v_design=v_design,
                Dc_mm=Dc_mm,
                delta_f_mm=delta_f_mm,
                s1_mm=s1_mm,               # <-- S1 varies per row
                pitch_ratio=pitch_ratio,
                N=N,
                v_range=(v_min, v_max),
                n_points=n_points,
            )

            porous = res["porous"]
            geom = res["geometry"]
            design = res["design_point"]

            out_rows.append({
                # original design vars
                "S1_mm": s1_mm,
                "fin_height_fh_mm": fh_mm,
                "fin_spacing_fs_mm": fs_mm,

                # coupled constraint diagnostics
                "fh_upper_mm": float(fh_up),
                "constraint_ok": bool(constraint_ok),

                # porous outputs
                "Viscous_Resistance_1_m2": float(porous["inv_K"]),
                "Inertial_Resistance_1_m": float(porous["C2"]),
                "K_m2": float(porous["K"]),
                "R2_fit": float(porous["R_squared"]),

                # geometry
                "Porosity": float(geom["epsilon"]),
                "a_fs_1_m": float(geom["a_fs"]),
                "area_ratio": float(geom["area_ratio"]),
                "sigma": float(geom["sigma"]),
                "porous_thickness_m": float(geom["porous_thickness_m"]),
                "flow_depth_m": float(geom["flow_depth_m"]),

                # design-point metrics
                "Re_Dc": float(design["Re_Dc"]),
                "dP_total_Pa": float(design["dP_total_Pa"]),
                "dP_per_L_Pa_m": float(design["dP_per_L_Pa_m"]),
                "h_fs_W_m2K": float(design["h_fs_W_m2K"]),

                # meta
                "ok": True,
                "error": "",
            })
        except Exception as e:
            out_rows.append({
                "S1_mm": np.nan,
                "fin_height_fh_mm": np.nan,
                "fin_spacing_fs_mm": np.nan,
                "fh_upper_mm": np.nan,
                "constraint_ok": False,
                "Viscous_Resistance_1_m2": np.nan,
                "Inertial_Resistance_1_m": np.nan,
                "K_m2": np.nan,
                "R2_fit": np.nan,
                "Porosity": np.nan,
                "a_fs_1_m": np.nan,
                "area_ratio": np.nan,
                "sigma": np.nan,
                "porous_thickness_m": np.nan,
                "flow_depth_m": np.nan,
                "Re_Dc": np.nan,
                "dP_total_Pa": np.nan,
                "dP_per_L_Pa_m": np.nan,
                "h_fs_W_m2K": np.nan,
                "ok": False,
                "error": repr(e),
            })

        if (i % progress_every) == 0 or i == n:
            print(f"[{i:>7d}/{n}] processed")

    return pd.DataFrame(out_rows)


def main():
    import os
    
    # optimum_design_variables.xlsx 파일 읽기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(current_dir, "optimum_design_variables.xlsx")
    
    print(f"Reading Excel file: {excel_file}")
    df = pd.read_excel(excel_file)
    
    print(f"\n{'='*80}")
    print(f"최적 설계 변수 (Optimum Design Variables)")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
    
    # 기본 파라미터 설정
    T_celsius = 14.8
    v_design = 2.019
    Dc_mm = 24.0
    delta_f_mm = 0.5
    pitch_ratio = 1.0
    N = 4
    v_min = 0.5
    v_max = 3.5
    n_points = 50
    check_constraint = False
    progress_every = 500
    
    # Porous media 파라미터 계산
    print(f"Calculating porous media parameters...\n")
    out_df = run_batch(
        df,
        T_celsius=T_celsius,
        v_design=v_design,
        Dc_mm=Dc_mm,
        delta_f_mm=delta_f_mm,
        pitch_ratio=pitch_ratio,
        N=N,
        v_min=v_min,
        v_max=v_max,
        n_points=n_points,
        check_constraint=check_constraint,
        progress_every=progress_every,
    )
    
    # 결과 출력
    print(f"\n{'='*80}")
    print(f"Porous Media 파라미터 결과")
    print(f"{'='*80}")
    
    # 주요 컬럼만 선택해서 출력
    key_columns = [
        "S1_mm", "fin_height_fh_mm", "fin_spacing_fs_mm",
        "Viscous_Resistance_1_m2", "Inertial_Resistance_1_m", 
        "K_m2", "Porosity", "R2_fit"
    ]
    
    available_columns = [col for col in key_columns if col in out_df.columns]
    print(out_df[available_columns].to_string(index=False))
    print(f"{'='*80}\n")
    
    # 상세 결과 출력
    print(f"{'='*80}")
    print(f"전체 상세 결과")
    print(f"{'='*80}")
    print(out_df.to_string(index=False))
    print(f"{'='*80}\n")
    
    # CSV로 저장
    out_path = os.path.join(current_dir, "optimum_porous_media_parameters.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
