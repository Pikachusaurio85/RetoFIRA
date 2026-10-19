"""10_figures.py - Genera todas las figuras (PNG) a partir de results/*.csv y oof/. Paleta de referencia validada (dataviz), superficie clara."""
import sys, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
from matplotlib.lines import Line2D
from oofio import *

D = Data(); RES = PATH["res"]; FIG = PATH["fig"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 100})
REG = pd.read_csv(f"{PATH['cfg']}/regimen_captura_por_municipio.csv", index_col=0).regimen
SHORT = {"Chignahuapan": "Chignahuapan", "Calpulalpan": "Calpulalpan", "Singuilucan": "Singuilucan", "Almoloya": "Almoloya", "Cuautepec de Hinojosa": "Cuautepec", "Emiliano Zapata": "E. Zapata",
         "Apan": "Apan", "Nanacamilpa de Mariano Arista": "Nanacamilpa", "Tepeapulco": "Tepeapulco"}


def save(fig, name):
    fig.savefig(f"{FIG}/{name}", dpi=170, bbox_inches="tight"); plt.close(fig); print("fig", name)


def safe_run(fn):
    try:
        fn()
    except Exception as e:
        print("FALLO", fn.__name__, repr(e)); traceback.print_exc()


def fig01():
    tr = D.tr; order = tr.groupby("loc_municipio").RENDIMIENTO_T_HA.mean().sort_values().index.tolist()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    r = np.random.default_rng(0)
    for i, m in enumerate(order):
        v = tr[tr.loc_municipio == m].RENDIMIENTO_T_HA.values; col = ORANGE if REG[m].startswith("gruesa") else BLUE
        ax.scatter(i + r.uniform(-.18, .18, len(v)), v, s=14, color=col, alpha=.75, linewidths=0)
        ax.plot([i - .3, i + .3], [v.mean()] * 2, color=INK, lw=1.6)
        ax.text(i, 1.72, f"n={len(v)}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([SHORT[m] for m in order], rotation=30, ha="right"); ax.set_ylabel("Rendimiento (t/ha)"); ax.set_ylim(1.6, 5.8)
    ax.set_title("Rendimiento por municipio y régimen de captura del dato"); ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="captura fina (continua)"), Line2D([], [], marker="o", ls="", color=ORANGE, label="captura gruesa (múltiplos de 0.5)"),
                       Line2D([], [], color=INK, lw=1.6, label="media")], loc="upper left", fontsize=8)
    v = pd.read_csv(f"{RES}/02_regimen_vs_municipio_secuencial.csv", index_col=0).valor; vv = pd.read_csv(f"{RES}/02_varianza_entre_dentro_municipio.csv", index_col=0)
    a, b = v["R2_regimen"], v["R2_incremental_municipio_dado_regimen"]; c = 1 - a - b
    left = 0
    for val, col, lab in ((a, ORANGE, "régimen de captura"), (b, MUTED, "municipio | régimen"), (c, BLUE, "dentro de municipio")):
        ax2.barh(0, val, left=left, color=col, height=.5, edgecolor=SURF, linewidth=2); ax2.text(left + val / 2, 0, f"{val*100:.1f}%" if val > .05 else "", ha="center", va="center", color="white", fontsize=8.5)
        left += val
    ax2.set_xlim(0, 1); ax2.set_ylim(-.6, .6); ax2.set_yticks([]); ax2.set_xlabel("fracción de la suma de cuadrados del rendimiento"); ax2.set_title("Descomposición de la varianza"); ax2.grid(False)
    ax2.legend(handles=[matplotlib.patches.Patch(color=ORANGE, label="régimen de captura (fina/gruesa)"), matplotlib.patches.Patch(color=MUTED, label="municipio dado el régimen"), matplotlib.patches.Patch(color=BLUE, label="dentro de municipio")], fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.28))
    save(fig, "fig01_rendimiento_municipio_y_varianza.png")


def fig02():
    allf = load_all_features(); fig, ax = plt.subplots(figsize=(6.6, 6.2))
    h = allf[allf.CONJUNTO == "PREDICCION"]; l = allf[allf.CONJUNTO == "ENTRENAMIENTO"]
    ax.scatter(h.loc_x_utm / 1000, h.loc_y_utm / 1000, s=34, facecolors="none", edgecolors=MUTED, linewidths=1.1, label="59 sin etiqueta (solo posición)")
    sc = ax.scatter(l.loc_x_utm / 1000, l.loc_y_utm / 1000, s=26, c=l.RENDIMIENTO_T_HA, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("s", SEQ), vmin=2, vmax=5.6, linewidths=0, label="138 etiquetadas")
    cb = fig.colorbar(sc, ax=ax, shrink=.7); cb.set_label("Rendimiento (t/ha)"); ax.set_aspect("equal"); ax.set_xlabel("X UTM 14N (km)"); ax.set_ylabel("Y UTM 14N (km)")
    for m, g in allf.groupby("loc_municipio"):
        if len(g) >= 9:
            ax.text(g.loc_x_utm.mean() / 1000, g.loc_y_utm.mean() / 1000 + 2.6, SHORT[m], ha="center", fontsize=8, color=INK2)
    ax.set_title("Las parcelas ocultas están intercaladas con las etiquetadas"); ax.legend(loc="lower left", fontsize=8)
    save(fig, "fig02_mapa_parcelas.png")


def logticks(ax, ticks):
    ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks])); ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_locator(FixedLocator([]))


def fig03():
    vg = pd.read_csv(f"{RES}/02_semivariograma.csv"); cg = pd.read_csv(f"{RES}/02_correlograma_moran.csv"); pw = pd.read_csv(f"{RES}/02_potencia_moran_chignahuapan.csv")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    g = vg[vg.serie.str.startswith("Chignahuapan")]; x = (g.d_lo_km + g.d_hi_km) / 2
    axs[0].fill_between(x, g.env_p2_5, g.env_p97_5, color=GRID, label="envolvente 95 % (permutación intra-municipio)"); axs[0].plot(x, g.env_p50, color=MUTED, lw=1, ls="--")
    axs[0].plot(x, g.semivarianza, "-o", color=BLUE, ms=4, lw=1.6, label="observada"); logticks(axs[0], [0.125, 0.5, 1, 2, 4]); axs[0].set_xlabel("distancia (km, centro del anillo)"); axs[0].set_ylabel("semivarianza (t/ha)²")
    axs[0].set_title("Semivariograma, Chignahuapan (n=71)"); axs[0].legend(fontsize=7.5, loc="lower right")
    for s_, col, lab in (("Chignahuapan", BLUE, "Chignahuapan"), ("intra-municipio (pooled, centrado)", ORANGE, "pooled intra-municipio")):
        c = cg[cg.serie == s_]; xs = (c.d_lo_km + c.d_hi_km) / 2
        axs[1].plot(xs, c.I, "-o", color=col, ms=4, lw=1.4, label=lab)
        sig = c[c.p_one_sided < .05]; axs[1].scatter((sig.d_lo_km + sig.d_hi_km) / 2, sig.I, s=70, facecolors="none", edgecolors=INK, linewidths=1.2)
    axs[1].axhline(0, color=MUTED, lw=.8); logticks(axs[1], [0.25, 0.75, 1.5, 3, 6]); axs[1].set_xlabel("centro del anillo de distancia (km)"); axs[1].set_ylabel("I de Moran"); axs[1].set_title("Correlograma (círculo: p<0.05 sin corregir)"); axs[1].legend(fontsize=7.5)
    for share, col in ((0.2, MUTED), (0.4, BLUE), (0.6, ORANGE), (0.8, AQUA)):
        p = pw[pw.proporcion_varianza_espacial == share]; axs[2].plot(p.rango_practico_km, p.potencia, "-o", color=col, ms=4, label=f"{int(share*100)} % de varianza espacial")
    axs[2].axhline(.8, color=INK2, lw=.8, ls=":"); logticks(axs[2], [.5, 1, 2, 4]); axs[2].set_ylim(0, 1.02)
    axs[2].set_xlabel("rango práctico (km)"); axs[2].set_ylabel("potencia (p<0.05)"); axs[2].set_title("Potencia de Moran con n=71"); axs[2].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); save(fig, "fig03_autocorrelacion_espacial.png")


def fig04():
    FE = pd.read_csv(f"{RES}/02_eta2_por_feature.csv"); idn = pd.read_csv(f"{RES}/02_identificabilidad_municipio_por_familia.csv")
    order = ["LOC", "CLIM", "T_PRE", "T_VEG", "S2", "PL", "SAT", "T_PHTIM", "T_PHLEV", "T_PICO", "T_SEN", "LS", "TOPO", "T_HET", "GEOM"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    r = np.random.default_rng(1)
    for i, f in enumerate(order):
        v = FE[FE.familia == f].omega2_municipio.values; a1.scatter(v, i + r.uniform(-.2, .2, len(v)), s=6, color=BLUE, alpha=.45, linewidths=0); a1.plot([np.median(v)] * 2, [i - .3, i + .3], color=INK, lw=1.6)
    a1.set_yticks(range(len(order))); a1.set_yticklabels(order); a1.invert_yaxis(); a1.set_xlabel("ω² entre municipios (variable)"); a1.set_title("¿Cuánto de cada variable es 'municipio'?"); a1.axvline(0, color=MUTED, lw=.8)
    d2 = idn.set_index("familia").loc[[o for o in order if o in set(idn.familia)]]
    a2.barh(range(len(d2)), d2.acc_municipio_4clases, color=BLUE, height=.6, edgecolor=SURF, linewidth=1.5); a2.axvline(d2.acc_mayoritaria.iloc[0], color=ORANGE, lw=1.4, ls="--")
    a2.text(d2.acc_mayoritaria.iloc[0] + .01, len(d2) - .4, "clase mayoritaria", color=ORANGE, fontsize=8)
    a2.set_yticks(range(len(d2))); a2.set_yticklabels(d2.index); a2.invert_yaxis(); a2.set_xlim(.4, 1.02); a2.set_xlabel("exactitud CV para reconocer el municipio (4 municipios, sin usar rendimiento)"); a2.set_title("Cada familia 'delata' el municipio")
    fig.tight_layout(); save(fig, "fig04_features_identifican_municipio.png")


NICE = {"ref_global": "media global", "ref_muni": "prior municipal", "ref_knn": "kNN espacial (k=5)", "ref_muni_knnres": "prior mun.+kNN del residuo",
        "direct|ET|LOC": "ET · solo ubicación", "direct|ET|CLIM": "ET · solo clima", "direct|ET|SAT": "ET · solo satélite", "direct|ET|B1": "ET · baseline 1 (sat+topo+geom)",
        "direct|ET|ALLNOLOC": "ET · todo sin ubicación", "direct|ET|ALL": "ET · todo", "direct|Ridge|SAT": "Ridge · solo satélite", "direct|Ridge|ALLNOLOC": "Ridge · todo sin ubic.",
        "resid_muni|Ridge|CUR": "prior mun.+Ridge(CUR24)", "resid_muni|Ridge|SAT": "prior mun.+Ridge(sat)", "resid_muni|ET|SAT": "prior mun.+ET(sat)", "resid_muni|ET|ALL": "prior mun.+ET(todo)",
        "resid_muni|ET|ALLNOLOC": "prior mun.+ET(todo s/ubic.)", "direct|CB|ALL": "CatBoost · todo", "resid_muni|CB|ALL": "prior mun.+CatBoost(todo)"}


def fig05():
    W = pd.read_csv(f"{RES}/04_rmse_ancho.csv", index_col=0); keys = [k for k in NICE if k in W.index]
    sch = ["RCV5x10", "SPA_GRID3B05", "SPA_GRID8B1", "SPA_BAND6", "SPA_KM6", "LMO"]; ttl = {"RCV5x10": "CV aleatoria 5×10", "SPA_GRID3B05": "bloques 3 km + buffer", "SPA_GRID8B1": "bloques 8 km + buffer", "SPA_BAND6": "6 bandas",
                                                                                         "SPA_KM6": "6 bloques k-means", "LMO": "leave-municipio-out"}
    fig, axs = plt.subplots(1, 6, figsize=(16, 5.6), sharey=True)
    for ax, s in zip(axs, sch):
        y = np.arange(len(keys)); v = W.loc[keys, s].values
        col = [MUTED if k.startswith("ref_") else (BLUE if k.startswith("direct") else ORANGE) for k in keys]
        ax.hlines(y, W.loc["ref_muni", s], v, color=GRID, lw=1.2); ax.scatter(v, y, c=col, s=34, zorder=3, linewidths=0)
        ax.axvline(W.loc["ref_muni", s], color=MUTED, lw=.8, ls=":"); ax.set_title(ttl[s], fontsize=9); ax.set_xlabel("RMSE (t/ha)")
    axs[0].set_yticks(range(len(keys))); axs[0].set_yticklabels([NICE[k] for k in keys]); axs[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label="referencia"), Line2D([], [], marker="o", ls="", color=BLUE, label="modelo directo"), Line2D([], [], marker="o", ls="", color=ORANGE, label="prior + residuo")], fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("RMSE fuera de muestra con las MISMAS particiones (línea punteada: prior municipal)", x=.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig05_rmse_por_esquema.png")


def fig06():
    C = pd.read_csv(f"{RES}/04_contrastes_pareados.csv"); fams = ["LOC", "S2", "PL", "LS", "SAT", "CLIM", "TOPO", "GEOM", "T_PRE", "T_VEG", "T_PICO", "T_SEN", "T_PHTIM", "T_PHLEV", "T_HET", "CUR", "B1", "ALLNOLOC", "ALL"]
    sch = [("RCV5x10", "CV aleatoria"), ("SPA_GRID3B05", "bloques 3 km"), ("SPA_KM6", "6 bloques k-means (regional)")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 5.6), sharey=True)
    for ax, (s, t) in zip(axs, sch):
        ax.axvspan(-.02, .02, color=GRID, alpha=.9, zorder=0)
        for j, (model, col, off) in enumerate((("Ridge", BLUE, -.17), ("ET", ORANGE, .17))):
            sub = C[(C.grupo == f"C1 residuo[F] vs prior municipal | {model}") & (C.esquema == s)].set_index("nota")
            for i, f in enumerate(fams):
                if f not in sub.index:
                    continue
                r = sub.loc[f]; ax.plot([r.ci_lo_clu, r.ci_hi_clu], [i + off] * 2, color=col, lw=1.4); ax.scatter(r.dRMSE, i + off, color=col, s=22, zorder=3, linewidths=0)
        ax.axvline(0, color=INK2, lw=.8); ax.set_title(t); ax.set_xlabel("ΔRMSE vs prior municipal (t/ha; <0 = mejor)")
    axs[0].set_yticks(range(len(fams))); axs[0].set_yticklabels(fams); axs[0].invert_yaxis()
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="Ridge"), Line2D([], [], color=ORANGE, lw=2, label="Extra Trees"), matplotlib.patches.Patch(color=GRID, label="zona de equivalencia ±0.02")], fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Aporte de cada familia SOBRE el prior municipal (IC95 bootstrap por cluster espacial)", x=.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig06_aporte_incremental_sobre_prior.png")


def fig07():
    fn = f"{RES}/04b_bloo_rmse.csv"
    B = pd.read_csv(fn); fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    cmap = {"ref_muni": (MUTED, "prior municipal"), "ref_knn": (AQUA, "kNN espacial"), "direct|ET|LOC": (MAGENTA, "ET · solo ubicación"), "direct|ET|ALL": (BLUE, "ET · todo"), "resid_muni|ET|SAT": (ORANGE, "prior mun.+ET(sat)"), "resid_muni|Ridge|CUR": (YELLOW, "prior mun.+Ridge(CUR24)")}
    for k, ax in enumerate(axs):
        for c, (col, lab) in cmap.items():
            s_ = B[B.cfg == c].sort_values("radio_km")
            if k == 1:
                s_ = s_[s_.radio_km <= 2]
            if len(s_):
                ax.plot(s_.radio_km, s_.RMSE, "-o", color=col, ms=4, lw=1.6, label=lab)
        ax.set_xlabel("radio de exclusión alrededor de cada parcela de prueba (km)"); ax.set_ylabel("RMSE LOO con buffer (t/ha)")
        ax.axvline(0.35, color=INK2, lw=.8, ls=":")
    axs[0].set_title("Rendimiento al alejar a los vecinos etiquetados", fontsize=10, fontweight="bold", loc="left")
    axs[1].set_title("Zoom 0-2 km (línea: distancia mediana oculta→etiquetada, 0.35 km)", fontsize=10, fontweight="bold", loc="left")
    axs[1].set_ylim(0.49, 0.62); axs[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); save(fig, "fig07_decaimiento_con_buffer.png")


def fig09():
    WC = pd.read_csv(f"{RES}/06_dentro_de_municipio_cv_y_permutacion.csv"); PT = pd.read_csv(f"{RES}/06_permutacion_intra_municipio.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ch = WC[(WC.municipio == "Chignahuapan") & (WC.modelo == "Ridge")].sort_values("skill_cv_vs_media_municipal", ascending=False)
    y = np.arange(len(ch)); axs[0].hlines(y, ch.ic95_lo, ch.ic95_hi, color=BLUE, lw=1.6); axs[0].scatter(ch.skill_cv_vs_media_municipal, y, color=BLUE, s=24, zorder=3, linewidths=0)
    axs[0].scatter(ch.null_p95, y, marker="|", color=ORANGE, s=120, zorder=4, label="percentil 95 de la nula (y permutada)"); axs[0].axvline(0, color=INK2, lw=.8)
    axs[0].set_yticks(y); axs[0].set_yticklabels(ch.conjunto); axs[0].invert_yaxis(); axs[0].set_xlabel("skill CV (1 − MSE/MSE de la media municipal)"); axs[0].set_title("Dentro de Chignahuapan (n=71), Ridge", fontsize=10, fontweight="bold", loc="left"); axs[0].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    pr = PT[PT.modelo == "Ridge"].sort_values("dRMSE_observado_vs_prior"); y = np.arange(len(pr))
    axs[1].hlines(y, pr.null_p5, pr.null_p95, color=GRID, lw=7, label="nula: X permutada dentro del municipio (p5–p95)"); axs[1].scatter(pr.dRMSE_observado_vs_prior, y, color=BLUE, s=26, zorder=3, linewidths=0, label="observado")
    axs[1].axvline(0, color=INK2, lw=.8); axs[1].set_yticks(y); axs[1].set_yticklabels(pr.conjunto); axs[1].invert_yaxis(); axs[1].set_xlabel("ΔRMSE vs prior municipal (CV aleatoria; <0 = mejor)"); axs[1].set_title("Prueba de permutación intra-municipio (Ridge)")
    axs[1].set_title("Permutación intra-municipio (Ridge, todos los municipios)", fontsize=10, fontweight="bold", loc="left"); axs[1].legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1); fig.tight_layout(); save(fig, "fig09_senal_dentro_de_municipio.png")


def fig10():
    S = pd.read_csv(f"{RES}/07_shap_grupos.csv"); P = pd.read_csv(f"{RES}/07_permutacion_grupos.csv")
    mods = [("direct|ET|ALL", "RCV5x10", "ET directo"), ("resid_muni|ET|ALL", "RCV5x10", "prior mun.+ET"), ("direct|Ridge|ALL", "RCV5x10", "Ridge directo"), ("direct|CB|ALL", "RCV5x10", "CatBoost directo")]
    fams = ["LOC", "CLIM", "S2", "PL", "LS", "TOPO", "GEOM", "MUN_prior"]; cols = [MAGENTA, ORANGE, BLUE, "#6da7ec", "#184f95", AQUA, YELLOW, MUTED]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    left = np.zeros(len(mods))
    for f, c in zip(fams, cols):
        vals = []
        for cf, sc, _ in mods:
            t = S[(S.cfg == cf) & (S.esquema == sc) & (S.agrupacion == "familia")]; tot = t.mean_abs_shap.sum(); r = t[t.grupo == f]
            vals.append(r.mean_abs_shap.sum() / tot if len(r) else 0)
        axs[0].barh(range(len(mods)), vals, left=left, color=c, edgecolor=SURF, linewidth=1.5, label=f); left += np.array(vals)
    axs[0].set_yticks(range(len(mods))); axs[0].set_yticklabels([m[2] for m in mods]); axs[0].invert_yaxis(); axs[0].set_xlabel("participación en |SHAP| medio (fuera de entrenamiento)"); axs[0].set_title("SHAP agrupado por familia"); axs[0].legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.18))
    for ax, cf, ttl in ((axs[1], "direct|ET|ALL", "Permutación de grupos · ET directo"), (axs[2], "resid_muni|ET|ALL", "Permutación de grupos · prior mun.+ET")):
        t = P[(P.cfg == cf) & (P.esquema == "RCV5x10") & (P.agrupacion == "familia")]
        gs = ["LOC", "CLIM", "S2", "PL", "LS", "TOPO", "GEOM"]; y = np.arange(len(gs))
        for off, v, col, lab in ((-.22, "global", BLUE, "global"), (0, "intra_municipio", ORANGE, "intra-municipio"), (.22, "intra_cluster", AQUA, "intra-cluster 25")):
            m = [t[(t.grupo == g) & (t.tipo_permutacion == v)].dRMSE_media.mean() for g in gs]
            ax.barh(y + off, m, height=.2, color=col, label=lab)
        ax.set_yticks(y); ax.set_yticklabels(gs); ax.invert_yaxis(); ax.set_xlabel("ΔRMSE al permutar el grupo"); ax.set_title(ttl); ax.axvline(0, color=INK2, lw=.8)
    axs[1].legend(fontsize=8)
    fig.tight_layout(); save(fig, "fig10_shap_y_permutacion_por_familia.png")


def fig11():
    R = pd.read_csv(f"{RES}/08_seleccion_anidada_vs_apriori.csv"); F = pd.read_csv(f"{RES}/08_frecuencia_seleccion.csv", index_col=0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    sch = [("RCV5x10", "CV aleatoria", BLUE), ("SPA_GRID3B05", "bloques 3 km", ORANGE), ("SPA_KM6", "6 bloques k-means", AQUA)]
    mods = [("nested_k5", "selección anidada k=5"), ("nested_k10", "selección anidada k=10"), ("nested_kopt", "anidada, k por CV interna"), ("apriori5", "5 a priori"), ("apriori10", "10 a priori")]
    for j, (s, lab, col) in enumerate(sch):
        for i, (m, _) in enumerate(mods):
            r = R[(R.esquema == s) & (R.modelo == m)].iloc[0]; yy = i + (j - 1) * .24
            a1.plot([r.ic_lo_cluster, r.ic_hi_cluster], [yy] * 2, color=col, lw=1.5); a1.scatter(r.dRMSE_vs_prior, yy, color=col, s=24, zorder=3, linewidths=0)
    a1.axvspan(-.02, .02, color=GRID, zorder=0); a1.axvline(0, color=INK2, lw=.8); a1.set_yticks(range(len(mods))); a1.set_yticklabels([m[1] for m in mods]); a1.invert_yaxis()
    a1.set_xlabel("ΔRMSE vs prior municipal solo (t/ha)"); a1.set_title("Prior municipal + pocas variables interpretables")
    fig.legend(handles=[Line2D([], [], color=c, lw=2, label=l) for _, l, c in sch], fontsize=8.5, loc="lower left", ncol=3, bbox_to_anchor=(0.02, -0.005))
    top = F.head(12).iloc[::-1]; a2.barh(range(len(top)), top.freq_top10_RCV, color=BLUE, height=.6); a2.set_yticks(range(len(top))); a2.set_yticklabels(top.index, fontsize=8)
    a2.set_xlabel("frecuencia de selección entre 50 folds externos (top-10)"); a2.set_title("¿Qué variables elige la selección anidada?"); a2.set_xlim(0, 1)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "fig11_pocas_variables_interpretables.png")


def fig12():
    T = pd.read_csv(f"{RES}/09_persistencia_temporal_sentinel2.csv"); fig, ax = plt.subplots(figsize=(8, 3.8))
    vars_ = ["s2_ndvi_base_ene_may", "s2_evi_base_ene_may", "s2_ndvi_media_estacional", "s2_ndvi_pico_estacional", "s2_lai_pico_estacional"]; r = np.random.default_rng(2)
    for i, v in enumerate(vars_):
        x = T[T.variable == v].rho_intra_municipio.values; ax.scatter(x, i + r.uniform(-.12, .12, len(x)), s=22, color=BLUE, alpha=.8, linewidths=0); ax.plot([np.median(x)] * 2, [i - .25, i + .25], color=INK, lw=1.6)
    ax.set_yticks(range(len(vars_))); ax.set_yticklabels(vars_); ax.invert_yaxis(); ax.set_xlabel("correlación de rangos entre años (dentro de municipio; 6 pares de años 2022–2025)"); ax.axvline(0, color=MUTED, lw=.8)
    ax.set_title("¿La posición de la parcela en cada índice persiste entre años? (sin usar rendimiento)"); save(fig, "fig12_persistencia_temporal.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig01", "fig02", "fig03", "fig04", "fig05", "fig06", "fig07", "fig09", "fig10", "fig11", "fig12"]
    for w in which:
        safe_run(globals()[w])
