"""02b_residual_autocorr.py - Autocorrelacion espacial (Moran I, permutacion) de los RESIDUOS OOF de modelos clave. Requiere oof/main."""
from oofio import *
from spatial import *

D = Data(); RES = PATH["res"]
O = load_oof("main"); L, P = per_parcel(O, D)
cfgs = ["ref_global", "ref_muni", "ref_knn", "resid_muni|Ridge|SAT", "resid_muni|ET|SAT", "resid_muni|ET|ALL", "resid_muni|Ridge|CUR", "direct|ET|ALL", "direct|ET|SAT", "direct|Ridge|ALLNOLOC", "direct|ET|LOC", "resid_muni|CB|ALL"]
cfgs = [c for c in cfgs if (c, "RCV5x10") in P]
rows = []
XY = D.XY
for sch in ("RCV5x10", "SPA_GRID3B05", "SPA_KM6"):
    for c in cfgs:
        if (c, sch) not in P:
            continue
        r = D.y - P[(c, sch)]
        for lab, z, xy, grp in (("todas (138)", r, XY, None), ("intra-municipio (centrado)", r, XY, D.mun), ("Chignahuapan (71)", r[D.mun == "Chignahuapan"], XY[D.mun == "Chignahuapan"], None)):
            mask = None if grp is None else (grp[:, None] == grp[None, :])
            W = W_knn(xy, 5, mask); has = W.sum(1) > 0; Wr = W[np.ix_(has, has)]; Wr = Wr / Wr.sum(1, keepdims=True)
            out = moran_perm(np.asarray(z)[has], Wr, None if grp is None else grp[has], nperm=4999, seed=5)
            rows.append(dict(esquema=sch, cfg=c, subconjunto=lab, pesos="kNN5", n=int(has.sum()), I=out["I"], z=out["z"], p_una_cola=out["p_one_sided"], sd_residuo=float(np.std(z))))
R = pd.DataFrame(rows); R.round(4).to_csv(f"{RES}/02b_moran_residuos_oof.csv", index=False)
pd.set_option("display.width", 250); print(R[R.esquema == "RCV5x10"].round(3).to_string(index=False))
print(R[R.esquema != "RCV5x10"].pivot_table(index=["cfg"], columns=["esquema", "subconjunto"], values="I").round(3).to_string())
