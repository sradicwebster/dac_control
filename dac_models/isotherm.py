import numpy as np
import pandas as pd
from omegaconf import DictConfig


def equilibrium_loadings(sorbent: str,
                         cond: DictConfig,
                         ):
    qeq_co2_ad = co2_equilibrium_loading(sorbent, cond["T_ad"], cond["p_ad"], cond["CO2_conc"],
                                         RH=cond["RH_ad"])
    qeq_co2_de = co2_equilibrium_loading(sorbent, cond["T_de"], cond["p_de"], cond["CO2_conc"],
                                         RH=cond["RH_de"])
    qeq_h2o_ad = h2o_equilibrium_loading(sorbent, cond["T_ad"], cond["RH_ad"])
    qeq_h2o_de = h2o_equilibrium_loading(sorbent, cond["T_de"], cond["RH_de"])
    return {"ad": qeq_co2_ad, "de": qeq_co2_de}, {"ad": qeq_h2o_ad, "de": qeq_h2o_de}


def co2_equilibrium_loading(sorbent: str,
                            T_degC: float,
                            p_bar: float,
                            co2_ppm: float,
                            RH: float = 0,
                            ):
    co2_params = pd.read_csv("data/co2_isotherm_data.csv").T
    co2_params = co2_params.rename(columns=co2_params.iloc[0]).drop("Solid_sorbent").infer_objects()
    assert sorbent in co2_params.index
    param = co2_params.loc[sorbent]
    R = 8.314e-3
    T_p = 273 + T_degC
    T0 = param["T0"]
    a = 116.87
    b = 15.0
    T_c = T_p - a * (278 / T_p)**b * RH
    ns_c = param["ns0_c"] * np.exp(param["x_c"] * (1 - T0 / T_c))
    b_c = param["b0_c"] * np.exp(param["deltaH_c"] / (R * T0) * (T0 / T_c - 1))
    t_c = param["t0_c"] + param["alpha_c"] * (1 - T0 / T_c)
    p = co2_ppm * p_bar * 1e-7
    q_c = ns_c * b_c * p / (1 + (b_c * p)**t_c)**(1 / t_c)
    if np.isnan(param["b0_p"]):
        q_p = np.zeros_like(q_c)
    else:
        ns_p = param["ns0_p"] * np.exp(param["x_p"] * (1 - T0 / T_p))
        b_p = param["b0_p"] * np.exp(param["deltaH_p"] / (R * T0) * (T0 / T_p - 1))
        t_p = param["t0_p"] + param["alpha_p"] * (1 - T0 / T_p)
        q_p = ns_p * b_p * p / (1 + (b_p * p)**t_p)**(1 / t_p)
    return float(q_c + q_p)


def h2o_equilibrium_loading(sorbent: str,
                            T_degC: float,
                            RH: float,
                            ):
    h2o_params = pd.read_csv("data/h20_isotherm_data.csv").T
    h2o_params = h2o_params.rename(columns=h2o_params.iloc[0]).drop("Parameter").infer_objects()
    assert sorbent in h2o_params.index
    param = h2o_params.loc[sorbent]
    R = 8.314e-3
    T = 273 + T_degC
    CG = param["C0"] * np.exp(param["deltaHC"] / (R * T))
    K = param["K0"] * np.exp(param["deltaHK"] / (R * T))
    Cm = param["Cm0"] * np.exp(param["beta"] / T)
    q = Cm * CG * K * RH / ((1 - K * RH) * (1 + (CG - 1) * K * RH))
    return float(q)
