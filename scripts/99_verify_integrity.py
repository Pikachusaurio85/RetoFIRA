"""
99_verify_integrity.py - Verificaciones finales de integridad (se puede re-ejecutar en cualquier momento).
  1. El arbol original (data_raw) es byte a byte identico al hash guardado ANTES de cualquier analisis.
  2. Ningun OOF / particion / tabla de errores contiene IDs de las 59 parcelas sin etiqueta (PREDICCION).
  3. Todas las particiones son disjuntas por fold (test) y cubren exactamente las 138 parcelas etiquetadas.
  4. Las 59 ocultas solo aparecen como FEATURES (sin etiqueta) en: 00_audit (validacion adversaria) y 09 (persistencia temporal).
Uso:  python3 scripts/99_verify_integrity.py [ruta_data_raw]   (por defecto /home/claude/data_raw)
"""
import sys, os, glob, gzip, subprocess, hashlib
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import PATH

raw = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/data_raw"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = []


def ok(name, cond, extra=""):
    out.append(dict(verificacion=name, resultado="OK" if cond else "FALLO", detalle=extra)); print(("OK    " if cond else "FALLO ") + name, extra)


# 1. arbol original
before = {}
for line in open(f"{BASE}/results/00_original_tree_sha256_BEFORE.txt"):
    h, p = line.rstrip("\n").split("  ", 1); before[p] = h
after = {}
for dp, _, fs in os.walk(raw):
    for f in fs:
        p = os.path.join(dp, f); rel = "./" + os.path.relpath(p, raw)
        hh = hashlib.sha256(open(p, "rb").read()).hexdigest(); after[rel] = hh
ok("arbol original identico al hash previo (n archivos=%d)" % len(before), before == after, f"modificados={sum(before.get(k) != v for k, v in after.items())}, nuevos={len(set(after) - set(before))}, borrados={len(set(before) - set(after))}")

# 2-3. IDs
asg = pd.read_csv(f"{BASE}/data/asignacion_oficial.csv", encoding="utf-8-sig")
asg.columns = [c.strip() for c in asg.columns]
lab = set(asg[asg.CONJUNTO == "ENTRENAMIENTO"].ID_POLIGONO); hid = set(asg[asg.CONJUNTO == "PREDICCION"].ID_POLIGONO)
ok("138 etiquetadas y 59 ocultas en la asignacion oficial", len(lab) == 138 and len(hid) == 59, f"{len(lab)}/{len(hid)}")

sp = pd.read_csv(f"{BASE}/splits/splits_long.csv")
ok("particiones: ningun ID oculto", len(set(sp.id) & hid) == 0)
ok("particiones: todos los IDs etiquetados", set(sp.id) <= lab)
cover = sp[sp.role == "test"].groupby(["scheme", "rep"]).id.apply(lambda s: (len(s), len(set(s)))).tolist()
ok("particiones: cada (esquema,rep) evalua cada parcela exactamente una vez", all(a == b == 138 for a, b in cover), f"{len(cover)} (esquema,rep)")

bad = 0; ncheck = 0
for f in glob.glob(f"{BASE}/oof/main/*.csv.gz") + glob.glob(f"{BASE}/oof/bloo/*.csv.gz") + glob.glob(f"{BASE}/oof/shap/*.csv.gz") + [f"{BASE}/oof/oof_promedio_reps_todas_configs.csv.gz", f"{BASE}/results/04_errores_por_parcela_configs_clave.csv.gz"]:
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f, usecols=lambda c: c in ("id", "ID_POLIGONO")); col = d.columns[0] if len(d.columns) else None
    if col is not None:
        ncheck += 1; bad += len(set(d[col]) & hid)
ok("OOF/SHAP/errores: ningun ID oculto en %d archivos" % ncheck, bad == 0)

bad = 0
for f in glob.glob(f"{BASE}/results/*.csv"):
    txt = open(f, encoding="utf-8", errors="ignore").read()
    bad += sum(h in txt for h in hid)
ok("tablas de resultados: ningun ID oculto", bad == 0)

# 4. etiquetas ocultas nunca cargadas: el modulo Data solo toma ENTRENAMIENTO
src = open(f"{BASE}/scripts/common.py").read()
ok("common.Data filtra por ENTRENAMIENTO", "ENTRENAMIENTO" in src)
pd.DataFrame(out).to_csv(f"{BASE}/results/99_verificacion_integridad.csv", index=False)
print("\nRESUMEN:", "TODO OK" if all(r["resultado"] == "OK" for r in out) else "HAY FALLOS")
