"""04b_bloo_analysis.py - RMSE del LOO con buffer (radio de exclusion r): ¿cuanto depende cada modelo de tener vecinos cercanos?"""
from oofio import *
D = Data(); O = load_oof("bloo")
O["radio_km"] = O.scheme.str.replace("BLOO_", "").astype(float)
rows = []
for (c, r), t in O.groupby(["cfg", "radio_km"]):
    t = t.drop_duplicates("id"); rows.append(dict(cfg=c, radio_km=r, RMSE=rmse(t.y, t.pred), MAE=mae(t.y, t.pred), R2=r2(t.y, t.pred), n_train_mediana=t.n_train.median(), n=len(t)))
B = pd.DataFrame(rows).sort_values(["cfg", "radio_km"]); B.round(4).to_csv(f"{PATH['res']}/04b_bloo_rmse.csv", index=False)
pd.set_option("display.width", 220); print(B.pivot(index="cfg", columns="radio_km", values="RMSE").round(3).to_string())
