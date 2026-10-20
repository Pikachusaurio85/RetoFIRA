"""Ejecuta el modulo predictor completo, en orden. Uso: python run_all.py [--base RUTA/cebada_datos_baseline_1]

auditoria_datos -> fase1_comparacion -> fase2_shap -> diagnostico_residuo148 -> fase3_modelo_final -> verificar
Las salidas se escriben en esta carpeta (outputs/predictor). No modifica data_processed ni otros pipelines.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PASOS = ["auditoria_datos.py", "fase1_comparacion.py", "fase2_shap.py", "diagnostico_residuo148.py", "fase3_modelo_final.py", "verificar.py"]

ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, default=AQUI.parents[1])
a = ap.parse_args()
for paso in PASOS:
    t0 = time.time()
    print(f"==> {paso}", flush=True)
    r = subprocess.run([sys.executable, str(AQUI / paso), "--base", str(a.base), "--out", str(AQUI)])
    if r.returncode != 0:
        sys.exit(f"Fallo en {paso} (codigo {r.returncode})")
    print(f"    {time.time() - t0:.0f} s", flush=True)
print("Listo.")
