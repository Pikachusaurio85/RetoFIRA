"""AgroCebada 2026 - utilidades compartidas del modulo predictor.

Carga y cruce de tablas (siempre por ID_POLIGONO), familias de variables,
modelos candidatos (prior municipal, Ridge 148, hibrido prior + Ridge topografico),
SHAP lineal exacto y metricas. Ningun archivo de origen se modifica.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KEY = "ID_POLIGONO"
TARGET = "RENDIMIENTO_T_HA"
MUN = "loc_municipio"

RCV_SEEDS = list(range(1000, 1010))          # mismas semillas documentadas en la auditoria previa
SEED_RIDGE_EXPERIMENTO = 2026                # particion del experimento Ridge del equipo
K_GRID = np.geomspace(0.05, 10.0, 25)        # malla de encogimiento del prior (rango de la auditoria)
ALPHAS = np.logspace(-2, 4, 13)              # malla de alpha del experimento Ridge original
TOPO4 = ["topo_elev_media", "topo_elev_std", "topo_pend_media", "topo_pend_std"]
MUNICIPIOS_N10 = ["Chignahuapan", "Calpulalpan", "Singuilucan", "Almoloya"]
ROPE = 0.02

FAMILIAS_DESC = {
    "SAT_BASICO_S2_LANDSAT": "Indices espectrales del conjunto BASICO (Sentinel-2 y Landsat), media de medias anuales 2022-2025",
    "SAT_PRO_PLANET": "Indices espectrales del conjunto PRO (Planet), media de medias anuales 2022-2025",
    "CALIDAD_NUBOSIDAD": "Porcentaje medio de nubosidad de las capturas (BASICO y PRO); describe la calidad de observacion",
    "CLIMA_PRECIPITACION": "Precipitacion mensual media 2022-2025 (mm), celda climatica",
    "CLIMA_TMIN": "Temperatura minima mensual media 2022-2025 (C), celda climatica",
    "CLIMA_TMAX": "Temperatura maxima mensual media 2022-2025 (C), celda climatica",
    "TOPOGRAFIA": "Elevacion y pendiente medias del poligono (INEGI CEM)",
}


# familias del contrato comun (README 01 - Claude, Fase 3): ubicacion, clima, satelite, topografia, geometria
FAMILIAS_CONTRATO = ["ubicacion", "clima", "satelite", "topografia", "geometria"]
FAMILIA_CONTRATO = {
    "SAT_BASICO_S2_LANDSAT": "satelite", "SAT_PRO_PLANET": "satelite", "CALIDAD_NUBOSIDAD": "satelite",
    "CLIMA_PRECIPITACION": "clima", "CLIMA_TMIN": "clima", "CLIMA_TMAX": "clima", "TOPOGRAFIA": "topografia",
}

# predicciones_oof.csv: columnas minimas del contrato + extras
SCHEMA_VERSION = "1.0"
OOF_A_CONTRATO = {"modelo": "model_id", "y_obs": "y_true_t_ha", "y_pred": "y_pred_t_ha", "error": "error_t_ha"}
OOF_COLUMNAS = ["schema_version", "model_id", "scheme", "repeat", "fold", KEY, "y_true_t_ha", "y_pred_t_ha",
                "municipio", "error_t_ha"]


def escribir_oof(oof: pd.DataFrame, path: Path) -> None:
    d = oof.rename(columns=OOF_A_CONTRATO)
    d.insert(0, "schema_version", SCHEMA_VERSION)
    d[OOF_COLUMNAS].to_csv(path, index=False)


def leer_oof(path: Path) -> pd.DataFrame:
    """Lee predicciones_oof.csv (formato contrato) con los nombres internos: modelo, y_obs, y_pred, error."""
    d = pd.read_csv(path).rename(columns={v: k for k, v in OOF_A_CONTRATO.items()})
    return d.drop(columns=["schema_version"])


# --------------------------------------------------------------------------- datos
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rutas_fuente(base: Path) -> dict[str, Path]:
    dp = Path(base) / "data_processed"
    return {
        "features_parcela": dp / "features_parcela.csv",
        "tabla_modelado_parcelaria": dp / "tabla_modelado_parcelaria.csv",
        "diccionario_features": dp / "diccionario_features.csv",
    }


def familia_148(col: str) -> str:
    if col.endswith("porcentaje_nubosidad"):
        return "CALIDAD_NUBOSIDAD"
    if "__BASICO__" in col:
        return "SAT_BASICO_S2_LANDSAT"
    if "__PRO__" in col:
        return "SAT_PRO_PLANET"
    if "__CLIMA__precipitacion" in col:
        return "CLIMA_PRECIPITACION"
    if "__CLIMA__tmin" in col:
        return "CLIMA_TMIN"
    if "__CLIMA__tmax" in col:
        return "CLIMA_TMAX"
    if col.startswith("TOPOGRAFIA__"):
        return "TOPOGRAFIA"
    raise ValueError(f"Columna sin familia: {col}")


def cargar_datos(base: Path) -> dict:
    """Une features_parcela (municipio, coordenadas, topo) y tabla_modelado_parcelaria (148) por ID_POLIGONO."""
    rutas = rutas_fuente(base)
    f = pd.read_csv(rutas["features_parcela"])
    t = pd.read_csv(rutas["tabla_modelado_parcelaria"])
    assert f[KEY].is_unique and t[KEY].is_unique, "ID_POLIGONO duplicado"
    assert set(f[KEY]) == set(t[KEY]), "Las tablas no tienen el mismo universo de parcelas"
    cols148 = [c for c in t.columns if c not in (KEY, "PARTICION", TARGET, "AREA_HA")]
    assert len(cols148) == 148, len(cols148)

    base_cols = [KEY, "CONJUNTO", TARGET, MUN, "loc_estado", "loc_x_utm", "loc_y_utm"] + TOPO4
    df = f[base_cols].merge(
        t[[KEY, "PARTICION", TARGET] + cols148].rename(columns={TARGET: TARGET + "_t"}),
        on=KEY, how="inner", validate="one_to_one")
    assert np.allclose(df[TARGET], df[TARGET + "_t"], equal_nan=True), "Rendimiento distinto entre tablas"
    assert (df["CONJUNTO"] == df["PARTICION"]).all(), "Particion oficial distinta entre tablas"
    df = df.drop(columns=[TARGET + "_t", "PARTICION"]).sort_values(KEY).reset_index(drop=True)

    lab_mask = df["CONJUNTO"].eq("ENTRENAMIENTO") & df[TARGET].notna()
    pred_mask = df["CONJUNTO"].eq("PREDICCION")
    assert int(lab_mask.sum()) == 138 and int(pred_mask.sum()) == 59, "Conteos inesperados"
    assert df.loc[pred_mask, TARGET].isna().all(), "Hay rendimiento en PREDICCION"
    return {
        "df": df,
        "lab": df.loc[lab_mask].reset_index(drop=True),
        "pred": df.loc[pred_mask].reset_index(drop=True),
        "cols148": cols148,
        "hashes": {k: sha256(p) for k, p in rutas.items()},
        "rutas": {k: str(p) for k, p in rutas.items()},
    }


# --------------------------------------------------------------------------- prior municipal
def _tabla_municipal(mun: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    g = pd.DataFrame({"m": mun, "y": y}).groupby("m")["y"]
    return pd.DataFrame({"n": g.size(), "suma": g.sum()})


def prior_loo(mun: np.ndarray, y: np.ndarray, k: float) -> np.ndarray:
    """Prediccion leave-self-out del prior para filas de entrenamiento (sin usar la propia etiqueta)."""
    tab = _tabla_municipal(mun, y)
    n_m = tab.loc[mun, "n"].to_numpy(float) - 1.0
    s_m = tab.loc[mun, "suma"].to_numpy(float) - y
    g_loo = (y.sum() - y) / (len(y) - 1.0)
    den = n_m + k
    with np.errstate(invalid="ignore", divide="ignore"):
        p = (s_m + k * g_loo) / den
    return np.where(den > 0, p, g_loo)


def elegir_k(mun: np.ndarray, y: np.ndarray, grid=K_GRID) -> tuple[float, np.ndarray]:
    mse = np.array([np.mean((y - prior_loo(mun, y, k)) ** 2) for k in grid])
    return float(grid[int(np.argmin(mse))]), mse


class ModeloMediaGlobal:
    nombre = "MEDIA_GLOBAL"

    def fit(self, d: pd.DataFrame):
        self.media_ = float(d[TARGET].mean())
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return np.full(len(d), self.media_)

    def hiper(self) -> dict:
        return {}


class ModeloPrior:
    """Media municipal encogida hacia la media global: (S_m + k*G) / (n_m + k).

    k=None -> k se elige con error leave-one-out usando solo las filas de entrenamiento.
    Municipio sin parcelas de entrenamiento -> media global G (con advertencia).
    """

    def __init__(self, k: float | None = None, grid=K_GRID, nombre: str = "PRIOR"):
        self.k_fijo, self.grid, self.nombre = k, grid, nombre

    def fit(self, d: pd.DataFrame):
        mun = d[MUN].astype(str).to_numpy()
        y = d[TARGET].to_numpy(float)
        if self.k_fijo is None:
            self.k_, self.curva_k_ = elegir_k(mun, y, self.grid)
        else:
            self.k_, self.curva_k_ = float(self.k_fijo), None
        self.media_global_ = float(y.mean())
        tab = _tabla_municipal(mun, y)
        tab["media"] = tab["suma"] / tab["n"]
        tab["peso"] = tab["n"] / (tab["n"] + self.k_)
        tab["prediccion"] = (tab["suma"] + self.k_ * self.media_global_) / (tab["n"] + self.k_)
        self.tabla_ = tab
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        mun = d[MUN].astype(str)
        return mun.map(self.tabla_["prediccion"]).fillna(self.media_global_).to_numpy(float)

    def componentes(self, municipio: str) -> dict:
        if municipio in self.tabla_.index:
            r = self.tabla_.loc[municipio]
            return {"media_global": self.media_global_, "media_municipal": float(r["media"]),
                    "n_municipio": int(r["n"]), "peso_municipal": float(r["peso"]), "k": self.k_,
                    "ajuste_municipal": float(r["prediccion"] - self.media_global_), "municipio_visto": True}
        return {"media_global": self.media_global_, "media_municipal": None, "n_municipio": 0,
                "peso_municipal": 0.0, "k": self.k_, "ajuste_municipal": 0.0, "municipio_visto": False}

    def hiper(self) -> dict:
        return {"k_prior": self.k_}


def pipeline_ridge() -> Pipeline:
    """Identico al experimento Ridge del equipo: imputacion mediana -> escalado -> RidgeCV(cv=5)."""
    return Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler()),
        ("ridge", RidgeCV(alphas=ALPHAS, cv=5, scoring="neg_mean_squared_error")),
    ])


class ModeloRidge148:
    nombre = "RIDGE148"

    def __init__(self, cols148: list[str]):
        self.cols = list(cols148)

    def fit(self, d: pd.DataFrame):
        self.pipe_ = pipeline_ridge().fit(d[self.cols].astype(float).to_numpy(), d[TARGET].to_numpy(float))
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict(d[self.cols].astype(float).to_numpy())

    def hiper(self) -> dict:
        return {"alpha_ridge": float(self.pipe_.named_steps["ridge"].alpha_)}


class ModeloHibridoTopo:
    """Prior municipal calibrado + Ridge (TOPO4) ajustado sobre el residuo leave-self-out del prior."""
    nombre = "HIBRIDO_TOPO"

    def __init__(self, cols=TOPO4):
        self.cols = list(cols)

    def fit(self, d: pd.DataFrame):
        self.prior_ = ModeloPrior().fit(d)
        mun = d[MUN].astype(str).to_numpy()
        y = d[TARGET].to_numpy(float)
        resid = y - prior_loo(mun, y, self.prior_.k_)
        self.pipe_ = pipeline_ridge().fit(d[self.cols].astype(float).to_numpy(), resid)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return self.prior_.predict(d) + self.pipe_.predict(d[self.cols].astype(float).to_numpy())

    def hiper(self) -> dict:
        return {"k_prior": self.prior_.k_, "alpha_ridge": float(self.pipe_.named_steps["ridge"].alpha_)}


def fabricas(cols148: list[str]) -> dict:
    return {
        "MEDIA_GLOBAL": lambda: ModeloMediaGlobal(),
        "PRIOR_K3": lambda: ModeloPrior(k=3.0, nombre="PRIOR_K3"),
        "PRIOR": lambda: ModeloPrior(),
        "RIDGE148": lambda: ModeloRidge148(cols148),
        "HIBRIDO_TOPO": lambda: ModeloHibridoTopo(),
    }


# --------------------------------------------------------------------------- SHAP lineal exacto
def shap_lineal_pipeline(pipe: Pipeline, X_bg: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """SHAP intervencional exacto para imputacion -> escalado -> modelo lineal.

    phi_j(x) = w_j / s_j * (imp(x)_j - E_bg[imp(x)_j]);  base = f(E_bg[imp(x)]).
    Supuesto: se rompe la dependencia entre variables (expectativa marginal sobre el fondo).
    Devuelve (phi [n x p], base, prediccion).
    """
    imp, esc, lin = pipe.named_steps["imputacion"], pipe.named_steps["escalado"], pipe.named_steps["ridge"]
    Xi, Bi = imp.transform(X), imp.transform(X_bg)
    w_raw = lin.coef_ / esc.scale_
    mu_bg = Bi.mean(axis=0)
    phi = (Xi - mu_bg) * w_raw
    base = float(lin.intercept_ + np.sum(lin.coef_ * (mu_bg - esc.mean_) / esc.scale_))
    return phi, base, pipe.predict(X)


# --------------------------------------------------------------------------- metricas
def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def escribir_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
