# -*- coding: utf-8 -*-
"""
Actividad: Aprendizaje NO supervisado - Segmentación de clientes de supermercado.

Pasos:
 1. Generación de datos sintéticos (>=3000 clientes) con distintas distribuciones.
 2. Análisis exploratorio (EDA).
 3. Escalado y aplicación de K-means, K-medoids y clustering jerárquico (Ward).
 4. Selección de k con método del codo y coeficiente de silueta.
 5. Comparación de algoritmos (silueta, Davies-Bouldin, Calinski-Harabasz, ARI).
 6. Interpretación de los clusters (perfiles de clientes).

Semilla: 42
"""
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figuras")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid")


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, nombre), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 1. GENERACIÓN DE DATOS
# ---------------------------------------------------------------------------
# Se simulan 4 segmentos latentes (mezcla). La etiqueta latente NO se usa para
# agrupar; sólo se guarda para validar a posteriori (ARI).
#
# Variables:
#  - edad               : Normal(mu, sigma) o Uniforme(a, b) según segmento
#  - ingresos_anuales   : LogNormal(ln(mediana), sigma)      [USD/año]
#  - frecuencia_mensual : Poisson(lambda)                    [visitas/mes]
#  - ticket_promedio    : Gamma(forma, escala) * f(ingresos) [USD/visita]
#  - gasto_mensual      : frecuencia * ticket (+ ruido)      [USD/mes]
SEGMENTOS = {
    "Jóvenes/estudiantes": dict(n=800, edad=("uniforme", 18, 28),
                                ingreso_mediana=14000, ingreso_sigma=0.35,
                                lam=5, gamma_forma=4.0, gamma_escala=5.0),
    "Familias": dict(n=1000, edad=("normal", 42, 6),
                     ingreso_mediana=38000, ingreso_sigma=0.30,
                     lam=14, gamma_forma=6.0, gamma_escala=9.0),
    "Profesionales premium": dict(n=600, edad=("normal", 33, 4),
                                  ingreso_mediana=90000, ingreso_sigma=0.25,
                                  lam=3, gamma_forma=6.0, gamma_escala=18.0),
    "Adultos mayores": dict(n=800, edad=("normal", 68, 5),
                            ingreso_mediana=22000, ingreso_sigma=0.30,
                            lam=16, gamma_forma=5.0, gamma_escala=3.5),
}

partes = []
for seg_id, (nombre, p) in enumerate(SEGMENTOS.items()):
    n = p["n"]
    tipo, a, b = p["edad"]
    edad = rng.uniform(a, b, n) if tipo == "uniforme" else rng.normal(a, b, n)
    edad = np.clip(np.round(edad), 18, 90)
    ingresos = rng.lognormal(np.log(p["ingreso_mediana"]), p["ingreso_sigma"], n)
    frecuencia = np.maximum(rng.poisson(p["lam"], n), 1)
    # ticket: gamma, levemente modulado por los ingresos del cliente
    factor = (ingresos / p["ingreso_mediana"]) ** 0.3
    ticket = rng.gamma(p["gamma_forma"], p["gamma_escala"], n) * factor
    gasto = frecuencia * ticket * rng.normal(1.0, 0.05, n)
    partes.append(pd.DataFrame({
        "edad": edad, "ingresos_anuales": ingresos.round(2),
        "frecuencia_mensual": frecuencia, "ticket_promedio": ticket.round(2),
        "gasto_mensual": np.maximum(gasto, 1).round(2),
        "segmento_latente": seg_id}))

df = pd.concat(partes, ignore_index=True).sample(frac=1, random_state=SEED)
df = df.reset_index(drop=True)
df.insert(0, "id_cliente", np.arange(1, len(df) + 1))
df.to_csv(os.path.join(BASE, "datos_clientes.csv"), index=False)
print(f"Datos generados: {df.shape[0]} clientes -> datos_clientes.csv")

VARS = ["edad", "ingresos_anuales", "frecuencia_mensual",
        "ticket_promedio", "gasto_mensual"]

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
desc = df[VARS].describe().T
desc["asimetria"] = df[VARS].skew()
desc["curtosis"] = df[VARS].kurt()
desc.round(3).to_csv(os.path.join(BASE, "estadisticas_descriptivas.csv"))
print("\nEstadísticas descriptivas:\n", desc.round(2))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, v in zip(axes.ravel(), VARS):
    sns.histplot(df[v], bins=40, kde=True, ax=ax, color="#3b6ea5")
    ax.set_title(f"Distribución de {v}")
axes.ravel()[-1].axis("off")
guardar("01_histogramas.png")

fig, axes = plt.subplots(1, len(VARS), figsize=(16, 4))
for ax, v in zip(axes, VARS):
    sns.boxplot(y=df[v], ax=ax, color="#9ecae1")
    ax.set_title(v)
guardar("02_boxplots.png")

corr = df[VARS].corr()
corr.round(3).to_csv(os.path.join(BASE, "correlaciones.csv"))
plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1)
plt.title("Matriz de correlación (Pearson)")
guardar("03_correlaciones.png")

muestra_pp = df.sample(800, random_state=SEED)
g = sns.pairplot(muestra_pp[VARS], corner=True, plot_kws=dict(s=8, alpha=0.5))
g.figure.suptitle("Pairplot (muestra de 800 clientes, sin etiquetas)", y=1.02)
g.savefig(os.path.join(FIG, "04_pairplot.png"), dpi=100)
plt.close("all")

plt.figure(figsize=(7, 5))
plt.scatter(df["ingresos_anuales"], df["gasto_mensual"], s=6, alpha=0.4)
plt.xlabel("Ingresos anuales (USD)"); plt.ylabel("Gasto mensual (USD)")
plt.title("Ingresos vs gasto mensual")
guardar("05_scatter_ingresos_gasto.png")

# ---------------------------------------------------------------------------
# 3. PREPROCESAMIENTO: log de variables asimétricas + StandardScaler
# ---------------------------------------------------------------------------
X_df = df[VARS].copy()
for v in ["ingresos_anuales", "ticket_promedio", "gasto_mensual"]:
    X_df[v] = np.log1p(X_df[v])  # reduce la asimetría de lognormal / gamma
X = StandardScaler().fit_transform(X_df)
y_lat = df["segmento_latente"].to_numpy()


# --------------------------- K-medoids --------------------------------------
def kmedoids_numpy(X, k, max_iter=100, seed=SEED, D=None):
    """K-medoids (algoritmo 'alternate' / Voronoi iteration) con init k-medoids++.

    1) Inicialización tipo k-medoids++ (como k-means++ pero eligiendo puntos).
    2) Asignación de cada punto al medoide más cercano.
    3) Para cada cluster se elige como nuevo medoide el punto que minimiza la
       suma de distancias al resto de puntos del cluster.
    Se repite hasta que los medoides no cambian.
    """
    r = np.random.default_rng(seed)
    if D is None:
        sq = (X ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    n = X.shape[0]
    med = [r.integers(n)]
    for _ in range(1, k):
        dmin = D[:, med].min(1) ** 2
        med.append(r.choice(n, p=dmin / dmin.sum()))
    med = np.array(med)
    for _ in range(max_iter):
        labels = D[:, med].argmin(1)
        nuevo = med.copy()
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx):
                nuevo[c] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
        if np.array_equal(np.sort(nuevo), np.sort(med)):
            med = nuevo
            break
        med = nuevo
    labels = D[:, med].argmin(1)
    coste = D[np.arange(n), med[labels]].sum()
    return labels, med, coste


def kmedoids_mejor(X, k, n_init=5, D=None):
    mejor = None
    for i in range(n_init):
        res = kmedoids_numpy(X, k, seed=SEED + i, D=D)
        if mejor is None or res[2] < mejor[2]:
            mejor = res
    return mejor


def obtener_kmedoids():
    """Intenta scikit-learn-extra; si no está disponible usa la versión numpy."""
    try:
        from sklearn_extra.cluster import KMedoids  # noqa
        return "scikit-learn-extra"
    except Exception:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "scikit-learn-extra"], check=True, timeout=300,
                           capture_output=True)
            from sklearn_extra.cluster import KMedoids  # noqa
            return "scikit-learn-extra"
        except Exception:
            return "implementación propia (numpy)"


impl_kmedoids = obtener_kmedoids()
print(f"\nK-medoids: {impl_kmedoids}")


def ajustar_kmedoids(X, k, D):
    if impl_kmedoids == "scikit-learn-extra":
        from sklearn_extra.cluster import KMedoids
        m = KMedoids(n_clusters=k, method="pam", init="k-medoids++",
                     random_state=SEED).fit(X)
        return m.labels_, m.medoid_indices_
    lab, med, _ = kmedoids_mejor(X, k, D=D)
    return lab, med


sq = (X ** 2).sum(1)
D_full = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))

# ---------------------------------------------------------------------------
# 4. SELECCIÓN DE k: codo + silueta
# ---------------------------------------------------------------------------
ks = list(range(2, 11))
inercias, sil_km, sil_kmed, sil_jer = [], [], [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    inercias.append(km.inertia_)
    sil_km.append(silhouette_score(X, km.labels_))
    lab_m, _ = ajustar_kmedoids(X, k, D_full)
    sil_kmed.append(silhouette_score(X, lab_m))
    lab_h = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    sil_jer.append(silhouette_score(X, lab_h))
    print(f"k={k:2d} inercia={km.inertia_:9.1f} silueta KM={sil_km[-1]:.3f} "
          f"KMed={sil_kmed[-1]:.3f} Jer={sil_jer[-1]:.3f}")

# Codo: punto de máxima distancia a la recta que une k=2 y k=10 (kneedle)
xs = np.array(ks, float); ys = np.array(inercias)
xn = (xs - xs[0]) / (xs[-1] - xs[0]); yn = (ys - ys[-1]) / (ys[0] - ys[-1])
k_codo = ks[int(np.argmax(np.abs(yn - (1 - xn))))]
k_sil = ks[int(np.argmax(sil_km))]
# Regla: si la silueta en el codo está a <=0.02 del máximo, se prefiere el codo
# (más segmentos útiles con calidad equivalente); si no, el máximo de silueta.
k_opt = k_codo if sil_km[ks.index(k_codo)] >= max(sil_km) - 0.02 else k_sil
print(f"\nk codo={k_codo}, k máx. silueta={k_sil} -> k elegido = {k_opt}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(ks, inercias, "o-", color="#3b6ea5")
axes[0].axvline(k_opt, ls="--", color="gray")
axes[0].set(title="Método del codo (K-means)", xlabel="k",
            ylabel="Inercia (SSE intra-cluster)")
axes[1].plot(ks, sil_km, "o-", label="K-means")
axes[1].plot(ks, sil_kmed, "s-", label="K-medoids")
axes[1].plot(ks, sil_jer, "^-", label="Jerárquico (Ward)")
axes[1].axvline(k_opt, ls="--", color="gray")
axes[1].set(title="Coeficiente de silueta por k", xlabel="k", ylabel="Silueta")
axes[1].legend()
guardar("06_codo_silueta.png")

# ---------------------------------------------------------------------------
# 5. MODELOS FINALES Y COMPARACIÓN
# ---------------------------------------------------------------------------
km = KMeans(n_clusters=k_opt, n_init=10, random_state=SEED).fit(X)
lab_km = km.labels_
lab_kmed, medoides = ajustar_kmedoids(X, k_opt, D_full)
lab_jer = AgglomerativeClustering(n_clusters=k_opt, linkage="ward").fit_predict(X)

resultados_alg = {}
for nombre, lab in [("K-means", lab_km), ("K-medoids", lab_kmed),
                    ("Jerárquico (Ward)", lab_jer)]:
    resultados_alg[nombre] = {
        "silueta": round(float(silhouette_score(X, lab)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X, lab)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, lab)), 1),
        "ARI_vs_segmento_latente": round(float(adjusted_rand_score(y_lat, lab)), 4),
        "tamanos_clusters": [int(c) for c in np.bincount(lab)],
    }
comp = pd.DataFrame(resultados_alg).T
comp.to_csv(os.path.join(BASE, "comparacion_algoritmos.csv"))
print("\nComparación de algoritmos:\n", comp)
ari_entre = {
    "KMeans_vs_KMedoids": round(float(adjusted_rand_score(lab_km, lab_kmed)), 4),
    "KMeans_vs_Jerarquico": round(float(adjusted_rand_score(lab_km, lab_jer)), 4),
    "KMedoids_vs_Jerarquico": round(float(adjusted_rand_score(lab_kmed, lab_jer)), 4),
}

# Gráfico de barras de métricas
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, m, t in zip(axes, ["silueta", "davies_bouldin", "calinski_harabasz",
                           "ARI_vs_segmento_latente"],
                    ["Silueta (↑)", "Davies-Bouldin (↓)", "Calinski-Harabasz (↑)",
                     "ARI vs latente (↑)"]):
    vals = comp[m].astype(float)
    ax.bar(vals.index, vals.values, color=["#3b6ea5", "#e6873c", "#4f9d69"])
    ax.set_title(t); ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(vals.values):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
guardar("07_comparacion_metricas.png")

# PCA 2D
pca = PCA(n_components=2, random_state=SEED)
Z = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_
fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
for ax, (t, lab) in zip(axes, [("K-means", lab_km), ("K-medoids", lab_kmed),
                               ("Jerárquico (Ward)", lab_jer),
                               ("Segmento latente (referencia)", y_lat)]):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=lab, cmap="tab10", s=5, alpha=0.6)
    ax.set_title(t)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.0%})"); ax.set_ylabel(f"PC2 ({var_exp[1]:.0%})")
axes[0].scatter(*pca.transform(km.cluster_centers_).T, c="black", marker="X",
                s=120, label="Centroides"); axes[0].legend()
axes[1].scatter(Z[medoides, 0], Z[medoides, 1], c="black", marker="*", s=180,
                label="Medoides"); axes[1].legend()
guardar("08_pca_clusters.png")

# Dendrograma sobre una muestra
idx_m = rng.choice(len(X), 300, replace=False)
L = linkage(X[idx_m], method="ward")
plt.figure(figsize=(14, 5))
dendrogram(L, truncate_mode="lastp", p=30, show_leaf_counts=True,
           leaf_rotation=90, color_threshold=L[-(k_opt - 1), 2])
plt.axhline(L[-(k_opt - 1), 2], ls="--", color="gray",
            label=f"Corte en k={k_opt}")
plt.title("Dendrograma (Ward, muestra de 300 clientes)")
plt.xlabel("Clientes / grupos"); plt.ylabel("Distancia"); plt.legend()
guardar("09_dendrograma.png")

# ---------------------------------------------------------------------------
# 6. INTERPRETACIÓN DE CLUSTERS (sobre K-means)
# ---------------------------------------------------------------------------
df["cluster_kmeans"] = lab_km
df["cluster_kmedoids"] = lab_kmed
df["cluster_jerarquico"] = lab_jer
perfil = df.groupby("cluster_kmeans")[VARS].mean()
perfil["n_clientes"] = df["cluster_kmeans"].value_counts().sort_index()
perfil["pct"] = (perfil["n_clientes"] / len(df) * 100).round(1)

# Asignación automática de nombre según el perfil relativo a la media global
media = df[VARS].mean()


def nombrar(fila):
    if fila["edad"] >= 58:
        return "Adultos mayores: compra frecuente y de bajo ticket"
    if fila["ingresos_anuales"] > 1.4 * media["ingresos_anuales"]:
        return "Profesionales premium: pocas visitas, ticket alto"
    if fila["edad"] < 30 and fila["gasto_mensual"] < media["gasto_mensual"]:
        return "Jóvenes/estudiantes: bajo ingreso y bajo gasto"
    if fila["frecuencia_mensual"] > media["frecuencia_mensual"]:
        return "Familias: alta frecuencia y mayor gasto mensual"
    return "Cliente medio"


perfil["perfil"] = perfil.apply(nombrar, axis=1)
perfil.round(2).to_csv(os.path.join(BASE, "perfiles_clusters.csv"))
df.to_csv(os.path.join(BASE, "datos_clientes_con_clusters.csv"), index=False)
print("\nPerfiles (K-means):\n", perfil.round(1).to_string())

# Tabla cruzada clusters vs segmento latente (validación)
nombres_lat = list(SEGMENTOS.keys())
ct = pd.crosstab(df["cluster_kmeans"],
                 df["segmento_latente"].map(dict(enumerate(nombres_lat))))
ct.to_csv(os.path.join(BASE, "tabla_cruzada_cluster_vs_latente.csv"))

# Heatmap de perfiles estandarizados
zp = (perfil[VARS] - media) / df[VARS].std()
plt.figure(figsize=(9, 4))
sns.heatmap(zp, annot=perfil[VARS].round(0), fmt="g", cmap="RdBu_r", center=0,
            yticklabels=[f"C{i}: {p.split(':')[0]}" for i, p in
                         zip(perfil.index, perfil["perfil"])])
plt.title("Perfil de clusters K-means (color = desviación estándar vs media)")
guardar("10_perfiles_heatmap.png")

fig, axes = plt.subplots(1, len(VARS), figsize=(18, 4))
for ax, v in zip(axes, VARS):
    sns.boxplot(data=df, x="cluster_kmeans", y=v, ax=ax, hue="cluster_kmeans",
                palette="tab10", legend=False)
    ax.set_title(v)
guardar("11_boxplots_por_cluster.png")

# ---------------------------------------------------------------------------
# 7. RESULTADOS JSON
# ---------------------------------------------------------------------------
resultados = {
    "semilla": SEED,
    "n_clientes": int(len(df)),
    "variables": VARS,
    "preprocesamiento": "log1p(ingresos, ticket, gasto) + StandardScaler",
    "k_evaluados": ks,
    "inercia_kmeans": [round(float(v), 1) for v in inercias],
    "silueta_kmeans": [round(float(v), 4) for v in sil_km],
    "silueta_kmedoids": [round(float(v), 4) for v in sil_kmed],
    "silueta_jerarquico": [round(float(v), 4) for v in sil_jer],
    "k_codo": int(k_codo),
    "k_max_silueta": int(k_sil),
    "k_elegido": int(k_opt),
    "implementacion_kmedoids": impl_kmedoids,
    "metricas_por_algoritmo": resultados_alg,
    "ARI_entre_algoritmos": ari_entre,
    "pca_varianza_explicada": [round(float(v), 4) for v in var_exp],
    "perfiles_kmeans": {
        int(c): {**{v: round(float(perfil.loc[c, v]), 2) for v in VARS},
                 "n_clientes": int(perfil.loc[c, "n_clientes"]),
                 "pct": float(perfil.loc[c, "pct"]),
                 "perfil": perfil.loc[c, "perfil"]} for c in perfil.index},
    "correlaciones": corr.round(3).to_dict(),
}
with open(os.path.join(BASE, "resultados.json"), "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)
print("\nListo: resultados.json y figuras en", FIG)
