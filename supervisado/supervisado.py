# -*- coding: utf-8 -*-
"""
Actividad: Machine Learning Supervisado - Clasificación de diabetes (datos sintéticos)
===================================================================================

Este script:
  1. Genera un conjunto de datos sintético de pacientes (n = 2000, semilla 42) con
     5 variables predictoras, cada una con una distribución teórica conocida:
        - edad          ~ Normal(50, 15) truncada al intervalo [18, 90]
        - glucosa       ~ Normal(100, 20)            (mg/dL, en ayunas)
        - imc           ~ LogNormal(mu = ln 27, sigma = 0.18)   (kg/m^2)
        - presion       ~ Normal(125, 15)            (mmHg, sistólica)
        - antecedentes  ~ Bernoulli(p = 0.30)        (1 = antecedentes familiares)
  2. Construye la etiqueta "diabetes" (Sí/No) con una REGLA CONOCIDA (score logístico):

        z = -11.5 + 0.03*edad + 0.05*glucosa + 0.08*imc + 0.015*presion
                  + 1.0*antecedentes + e,        e ~ Normal(0, 0.75)   (ruido)
        p = 1 / (1 + exp(-z))                    (función logística / sigmoide)
        diabetes = "Sí"  si p >= 0.5  (equivalente a z >= 0),  "No" en otro caso

     El ruido e simula factores no observados (dieta, genética no medida, error de
     medición), de modo que la frontera de decisión no es perfectamente separable.
  3. Realiza un análisis exploratorio (EDA): estadísticas descriptivas, histogramas con la
     curva teórica, boxplots por clase, matriz de correlación, conteo de clases y pruebas
     de bondad de ajuste (Kolmogorov-Smirnov, Shapiro-Wilk, Chi-cuadrado).
  4. Separa X / y, hace un train_test_split 80/20 estratificado y escala con StandardScaler
     (solo para la Regresión Logística, que es sensible a la escala).
  5. Entrena Regresión Logística y Random Forest, y calcula accuracy, precision, recall y F1
     en el conjunto de prueba, con matrices de confusión.
  6. Experimento de tamaño de muestra: n = 500, 2000, 10000 (5 repeticiones por tamaño).
  7. Guarda figuras en ./figuras, tablas CSV y un resultados.json con los números clave.

Ejecución:  python supervisado.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # backend sin ventana (solo guarda archivos)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------------------
SEED = 42
N_PRINCIPAL = 2000
BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figuras")
os.makedirs(FIG, exist_ok=True)

# Parámetros teóricos de las distribuciones (se usan para generar Y para las pruebas)
PARAM = {
    "edad": {"dist": "Normal truncada", "mu": 50.0, "sigma": 15.0, "a": 18.0, "b": 90.0},
    "glucosa": {"dist": "Normal", "mu": 100.0, "sigma": 20.0},
    "imc": {"dist": "LogNormal", "mu_log": float(np.log(27.0)), "sigma_log": 0.18},
    "presion": {"dist": "Normal", "mu": 125.0, "sigma": 15.0},
    "antecedentes": {"dist": "Bernoulli", "p": 0.30},
}

# Coeficientes de la regla de etiquetado (score logístico)
REGLA = {
    "intercepto": -11.5,
    "coef": {"edad": 0.03, "glucosa": 0.05, "imc": 0.08, "presion": 0.015, "antecedentes": 1.0},
    "sigma_ruido": 0.75,
    "umbral_probabilidad": 0.5,
}

FEATURES = ["edad", "glucosa", "imc", "presion", "antecedentes"]
VARS_CONTINUAS = ["edad", "glucosa", "imc", "presion"]


def dist_edad():
    """Distribución Normal(50,15) truncada a [18, 90] (scipy usa límites estandarizados)."""
    p = PARAM["edad"]
    a, b = (p["a"] - p["mu"]) / p["sigma"], (p["b"] - p["mu"]) / p["sigma"]
    return stats.truncnorm(a, b, loc=p["mu"], scale=p["sigma"])


def dist_imc():
    """LogNormal: si X ~ LogNormal(mu, s) entonces ln X ~ Normal(mu, s)."""
    p = PARAM["imc"]
    return stats.lognorm(s=p["sigma_log"], scale=np.exp(p["mu_log"]))


DIST_TEORICA = {
    "edad": dist_edad(),
    "glucosa": stats.norm(PARAM["glucosa"]["mu"], PARAM["glucosa"]["sigma"]),
    "imc": dist_imc(),
    "presion": stats.norm(PARAM["presion"]["mu"], PARAM["presion"]["sigma"]),
}


# ----------------------------------------------------------------------------------
# 1. Generación de datos sintéticos
# ----------------------------------------------------------------------------------
def score_lineal(df):
    """Combinación lineal (sin ruido) de la regla de etiquetado."""
    z = REGLA["intercepto"]
    for var, c in REGLA["coef"].items():
        z = z + c * df[var]
    return z


def generar_datos(n, seed=SEED):
    """Genera n pacientes sintéticos y su etiqueta según la regla documentada arriba."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "edad": DIST_TEORICA["edad"].rvs(size=n, random_state=rng),
        "glucosa": rng.normal(PARAM["glucosa"]["mu"], PARAM["glucosa"]["sigma"], n),
        "imc": rng.lognormal(PARAM["imc"]["mu_log"], PARAM["imc"]["sigma_log"], n),
        "presion": rng.normal(PARAM["presion"]["mu"], PARAM["presion"]["sigma"], n),
        "antecedentes": rng.binomial(1, PARAM["antecedentes"]["p"], n),
    })
    ruido = rng.normal(0.0, REGLA["sigma_ruido"], n)
    z = score_lineal(df) + ruido
    prob = 1.0 / (1.0 + np.exp(-z))                       # sigmoide
    df["diabetes"] = np.where(prob >= REGLA["umbral_probabilidad"], "Sí", "No")
    return df


df = generar_datos(N_PRINCIPAL, SEED)
df_csv = df.copy()
for c in ["edad", "glucosa", "imc", "presion"]:
    df_csv[c] = df_csv[c].round(2)
df_csv.to_csv(os.path.join(BASE, "datos_diabetes.csv"), index=False, encoding="utf-8-sig")
print(f"[1] Datos generados: {df.shape[0]} registros x {df.shape[1]} columnas -> datos_diabetes.csv")

# ----------------------------------------------------------------------------------
# 2. Análisis exploratorio (EDA)
# ----------------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
COLORES = {"No": "#4C72B0", "Sí": "#DD8452"}

# 2.1 Estadísticas descriptivas
desc = df[FEATURES].describe().T
desc["asimetria"] = df[FEATURES].skew()
desc["curtosis"] = df[FEATURES].kurt()
desc = desc.rename(columns={"count": "n", "mean": "media", "std": "desv_std", "min": "minimo",
                            "25%": "Q1", "50%": "mediana", "75%": "Q3", "max": "maximo"})
desc.round(4).to_csv(os.path.join(BASE, "estadisticas_descriptivas.csv"), encoding="utf-8-sig")
desc_clase = df.groupby("diabetes")[FEATURES].mean().T
desc_clase.round(4).to_csv(os.path.join(BASE, "medias_por_clase.csv"), encoding="utf-8-sig")

# 2.2 Conteo / proporción de clases
conteo = df["diabetes"].value_counts().reindex(["No", "Sí"])
proporcion = (conteo / len(df)).round(4)
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(conteo.index, conteo.values, color=[COLORES["No"], COLORES["Sí"]])
for b, v, p in zip(bars, conteo.values, proporcion.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({p:.1%})", ha="center", va="bottom")
ax.set_title("Conteo de clases (diabetes)")
ax.set_xlabel("Diabetes")
ax.set_ylabel("Número de pacientes")
ax.set_ylim(0, conteo.max() * 1.18)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "conteo_clases.png"), dpi=130)
plt.close(fig)

# 2.3 Histogramas con curva teórica
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
axes = axes.ravel()
titulos = {
    "edad": "Edad ~ Normal(50, 15) truncada [18, 90]",
    "glucosa": "Glucosa ~ Normal(100, 20)",
    "imc": "IMC ~ LogNormal(ln 27, 0.18)",
    "presion": "Presión ~ Normal(125, 15)",
}
for ax, var in zip(axes, VARS_CONTINUAS):
    ax.hist(df[var], bins=40, density=True, alpha=0.6, color="#4C72B0", edgecolor="white",
            label="Datos simulados")
    x = np.linspace(df[var].min(), df[var].max(), 400)
    ax.plot(x, DIST_TEORICA[var].pdf(x), color="#C44E52", lw=2.2, label="Densidad teórica")
    ax.set_title(titulos[var])
    ax.set_xlabel(var)
    ax.set_ylabel("Densidad")
    ax.legend()
# Antecedentes: barras observadas vs probabilidad teórica Bernoulli
ax = axes[4]
p_teo = PARAM["antecedentes"]["p"]
obs = df["antecedentes"].value_counts(normalize=True).reindex([0, 1]).fillna(0)
xpos = np.arange(2)
ax.bar(xpos - 0.2, obs.values, width=0.4, color="#4C72B0", alpha=0.7, label="Observado")
ax.bar(xpos + 0.2, [1 - p_teo, p_teo], width=0.4, color="#C44E52", alpha=0.7, label="Teórico")
ax.set_xticks(xpos, ["0 (No)", "1 (Sí)"])
ax.set_title("Antecedentes ~ Bernoulli(0.30)")
ax.set_ylabel("Proporción")
ax.legend()
# Q-Q plot de log(IMC) frente a la normal como apoyo
ax = axes[5]
stats.probplot(np.log(df["imc"]), dist="norm", plot=ax)
ax.set_title("Q-Q plot de ln(IMC) vs Normal")
ax.set_xlabel("Cuantiles teóricos")
ax.set_ylabel("Valores ordenados de ln(IMC)")
fig.suptitle("Distribuciones de las variables predictoras vs. distribución teórica", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "histogramas_teoricos.png"), dpi=130)
plt.close(fig)

# 2.4 Boxplots por clase
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, var in zip(axes, VARS_CONTINUAS):
    sns.boxplot(data=df, x="diabetes", y=var, hue="diabetes", order=["No", "Sí"],
                palette=COLORES, legend=False, ax=ax)
    ax.set_title(f"{var} por clase")
    ax.set_xlabel("Diabetes")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "boxplots_por_clase.png"), dpi=130)
plt.close(fig)

# Proporción de diabetes según antecedentes (variable binaria, el boxplot no aplica)
tab_ant = pd.crosstab(df["antecedentes"], df["diabetes"], normalize="index")
fig, ax = plt.subplots(figsize=(5, 4))
tab_ant[["No", "Sí"]].plot(kind="bar", stacked=True, color=[COLORES["No"], COLORES["Sí"]], ax=ax)
ax.set_title("Proporción de diabetes según antecedentes")
ax.set_xlabel("Antecedentes familiares (0 = No, 1 = Sí)")
ax.set_ylabel("Proporción")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "antecedentes_por_clase.png"), dpi=130)
plt.close(fig)

# 2.5 Matriz de correlación (incluye la etiqueta codificada 0/1)
df_corr = df[FEATURES].copy()
df_corr["diabetes (1=Sí)"] = (df["diabetes"] == "Sí").astype(int)
corr = df_corr.corr()
corr.round(4).to_csv(os.path.join(BASE, "matriz_correlacion.csv"), encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Matriz de correlación de Pearson")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "matriz_correlacion.png"), dpi=130)
plt.close(fig)

# 2.6 Pruebas de bondad de ajuste
ALFA = 0.05
pruebas = []


def agregar(variable, prueba, estadistico, p_valor, h0):
    pruebas.append({
        "variable": variable, "prueba": prueba, "hipotesis_nula": h0,
        "estadistico": round(float(estadistico), 5), "p_valor": round(float(p_valor), 5),
        "decision_alfa_0.05": "No se rechaza H0" if p_valor >= ALFA else "Se rechaza H0",
    })


for var in VARS_CONTINUAS:
    # KS con parámetros teóricos conocidos (no estimados de la muestra -> la prueba es válida)
    ks = stats.kstest(df[var], DIST_TEORICA[var].cdf)
    agregar(var, "Kolmogorov-Smirnov", ks.statistic, ks.pvalue,
            f"{var} sigue la distribución teórica ({PARAM[var]['dist']})")
for var in ["glucosa", "presion"]:
    sw = stats.shapiro(df[var])
    agregar(var, "Shapiro-Wilk", sw.statistic, sw.pvalue, f"{var} es normal")
sw = stats.shapiro(np.log(df["imc"]))
agregar("imc", "Shapiro-Wilk sobre ln(IMC)", sw.statistic, sw.pvalue,
        "ln(IMC) es normal (equivale a IMC lognormal)")
sw = stats.shapiro(df["edad"])
agregar("edad", "Shapiro-Wilk (normal sin truncar)", sw.statistic, sw.pvalue,
        "edad es normal SIN truncar (se espera rechazo leve por el truncamiento)")
obs_cnt = df["antecedentes"].value_counts().reindex([0, 1]).values
esp_cnt = np.array([1 - p_teo, p_teo]) * len(df)
chi = stats.chisquare(obs_cnt, esp_cnt)
agregar("antecedentes", "Chi-cuadrado de bondad de ajuste", chi.statistic, chi.pvalue,
        "antecedentes ~ Bernoulli(0.30)")
# Corrección por comparaciones múltiples (Bonferroni): con 9 pruebas a alfa = 0.05 se espera
# rechazar alguna H0 verdadera por azar (error tipo I). Umbral corregido = 0.05 / n_pruebas.
ALFA_BONF = ALFA / len(pruebas)
for p in pruebas:
    p["decision_bonferroni"] = "No se rechaza H0" if p["p_valor"] >= ALFA_BONF else "Se rechaza H0"
df_pruebas = pd.DataFrame(pruebas)
df_pruebas.to_csv(os.path.join(BASE, "pruebas_bondad_ajuste.csv"), index=False, encoding="utf-8-sig")
print("[2] Pruebas de bondad de ajuste:")
print(df_pruebas[["variable", "prueba", "estadistico", "p_valor", "decision_alfa_0.05",
                  "decision_bonferroni"]].to_string(index=False))

# ----------------------------------------------------------------------------------
# 3-4. Preparación, entrenamiento y evaluación
# ----------------------------------------------------------------------------------
PARAMS_LR = {"C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 1000, "random_state": SEED}
PARAMS_RF = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt",
             "bootstrap": True, "n_jobs": -1, "random_state": SEED}


def construir_modelos():
    """Regresión Logística (con StandardScaler en pipeline) y Random Forest (sin escalar:
    los árboles son invariantes a transformaciones monótonas de las variables)."""
    return {
        "Regresión Logística": Pipeline([("escalador", StandardScaler()),
                                         ("modelo", LogisticRegression(**PARAMS_LR))]),
        "Random Forest": RandomForestClassifier(**PARAMS_RF),
    }


def metricas(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def preparar(data, seed=SEED):
    X = data[FEATURES]
    y = (data["diabetes"] == "Sí").astype(int)          # 1 = Sí (clase positiva)
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)


X_train, X_test, y_train, y_test = preparar(df)
print(f"[3] Train: {len(X_train)}  Test: {len(X_test)}  "
      f"(prop. Sí train={y_train.mean():.3f}, test={y_test.mean():.3f})")

resultados_modelos = {}
filas = []
modelos = construir_modelos()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (nombre, modelo) in zip(axes, modelos.items()):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    m = metricas(y_test, pred)
    m_train = metricas(y_train, modelo.predict(X_train))
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    resultados_modelos[nombre] = {
        "test": {k: round(v, 4) for k, v in m.items()},
        "train": {k: round(v, 4) for k, v in m_train.items()},
        "matriz_confusion": {"etiquetas": ["No", "Sí"], "valores": cm.tolist(),
                             "VN": int(cm[0, 0]), "FP": int(cm[0, 1]),
                             "FN": int(cm[1, 0]), "VP": int(cm[1, 1])},
    }
    filas.append({"modelo": nombre, **{k: round(v, 4) for k, v in m.items()},
                  "accuracy_train": round(m_train["accuracy"], 4)})
    ConfusionMatrixDisplay(cm, display_labels=["No", "Sí"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{nombre}\nF1 = {m['f1']:.3f} | Accuracy = {m['accuracy']:.3f}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.grid(False)
    # Matriz individual
    f2, a2 = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay(cm, display_labels=["No", "Sí"]).plot(ax=a2, cmap="Blues", colorbar=False)
    a2.set_title(f"Matriz de confusión - {nombre}")
    a2.set_xlabel("Predicción")
    a2.set_ylabel("Valor real")
    a2.grid(False)
    f2.tight_layout()
    archivo = "matriz_confusion_" + ("regresion_logistica" if "Log" in nombre else "random_forest") + ".png"
    f2.savefig(os.path.join(FIG, archivo), dpi=130)
    plt.close(f2)
fig.suptitle("Matrices de confusión en el conjunto de prueba (n = 2000, test = 400)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "matrices_confusion.png"), dpi=130)
plt.close(fig)

# Referencia: la regla verdadera SIN ruido (el mejor clasificador posible con estas 5 variables)
pred_oraculo = (score_lineal(X_test) >= 0).astype(int)
m_oraculo = metricas(y_test, pred_oraculo)
filas.append({"modelo": "Referencia: regla verdadera sin ruido",
              **{k: round(v, 4) for k, v in m_oraculo.items()}, "accuracy_train": np.nan})
tabla = pd.DataFrame(filas)
tabla.to_csv(os.path.join(BASE, "comparacion_modelos.csv"), index=False, encoding="utf-8-sig")
print("[4] Métricas en test:")
print(tabla.to_string(index=False))

# Coeficientes LR (estandarizados) e importancias RF
lr = modelos["Regresión Logística"].named_steps["modelo"]
coef_lr = dict(zip(FEATURES, np.round(lr.coef_[0], 4)))
imp_rf = dict(zip(FEATURES, np.round(modelos["Random Forest"].feature_importances_, 4)))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pd.Series(coef_lr).sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0")
axes[0].set_title("Regresión Logística: coeficientes (variables escaladas)")
pd.Series(imp_rf).sort_values().plot(kind="barh", ax=axes[1], color="#55A868")
axes[1].set_title("Random Forest: importancia de variables (Gini)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "importancia_variables.png"), dpi=130)
plt.close(fig)

# Gráfico comparativo de métricas
fig, ax = plt.subplots(figsize=(8, 4.5))
t_plot = tabla.set_index("modelo")[["accuracy", "precision", "recall", "f1"]].iloc[:2]
t_plot.T.plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868"])
ax.set_ylim(0.5, 1.0)
ax.set_title("Comparación de métricas en test (n = 2000)")
ax.set_ylabel("Valor")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Modelo")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "comparacion_metricas.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------------
# 5. Experimento de tamaño de muestra
# ----------------------------------------------------------------------------------
TAMANOS = [500, 2000, 10000]
REPETICIONES = 5          # semillas 42..46 para estimar la variabilidad
registros = []
for n in TAMANOS:
    for r in range(REPETICIONES):
        semilla = SEED + r
        d = generar_datos(n, semilla)
        Xtr, Xte, ytr, yte = preparar(d, semilla)
        for nombre, modelo in construir_modelos().items():
            modelo.fit(Xtr, ytr)
            m = metricas(yte, modelo.predict(Xte))
            m_tr = metricas(ytr, modelo.predict(Xtr))
            registros.append({"n": n, "repeticion": r + 1, "semilla": semilla, "modelo": nombre,
                              "prop_si": round(float((d["diabetes"] == "Sí").mean()), 4),
                              **{k: round(v, 4) for k, v in m.items()},
                              "accuracy_train": round(m_tr["accuracy"], 4)})
df_exp = pd.DataFrame(registros)
df_exp.to_csv(os.path.join(BASE, "experimento_tamano_detalle.csv"), index=False, encoding="utf-8-sig")
resumen = (df_exp.groupby(["n", "modelo"])[["accuracy", "precision", "recall", "f1", "accuracy_train"]]
           .agg(["mean", "std"]).round(4))
resumen.columns = [f"{a}_{'media' if b == 'mean' else 'desv'}" for a, b in resumen.columns]
resumen = resumen.reset_index()
resumen.to_csv(os.path.join(BASE, "experimento_tamano.csv"), index=False, encoding="utf-8-sig")
print("[5] Experimento de tamaño (media de 5 repeticiones):")
print(resumen[["n", "modelo", "accuracy_media", "precision_media", "recall_media", "f1_media",
               "f1_desv", "accuracy_train_media"]].to_string(index=False))

fig, axes = plt.subplots(1, 4, figsize=(18, 4.3), sharey=True)
for ax, met in zip(axes, ["accuracy", "precision", "recall", "f1"]):
    for nombre, color in [("Regresión Logística", "#4C72B0"), ("Random Forest", "#55A868")]:
        s = resumen[resumen["modelo"] == nombre]
        ax.errorbar(s["n"], s[f"{met}_media"], yerr=s[f"{met}_desv"], marker="o", capsize=4,
                    color=color, label=nombre)
    ax.set_xscale("log")
    ax.set_xticks(TAMANOS, [str(t) for t in TAMANOS])
    ax.set_title(met.capitalize())
    ax.set_xlabel("Tamaño de la muestra n (escala log)")
axes[0].set_ylabel("Valor en test (media ± desv. estándar, 5 rep.)")
axes[0].legend()
fig.suptitle("Efecto del tamaño de la muestra en el desempeño")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "experimento_tamano.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------------
# 6. resultados.json
# ----------------------------------------------------------------------------------
mejor = max(["Regresión Logística", "Random Forest"], key=lambda k: resultados_modelos[k]["test"]["f1"])
pruebas_esperadas = [p for p in pruebas if "sin truncar" not in p["prueba"]]
rechazos_005 = [f"{p['variable']} ({p['prueba']}, p={p['p_valor']})" for p in pruebas_esperadas
                if p["decision_alfa_0.05"] == "Se rechaza H0"]
rechazos_bonf = [p for p in pruebas_esperadas if p["decision_bonferroni"] == "Se rechaza H0"]
if not rechazos_005:
    conclusion_ajuste = "Todas las variables son consistentes con su distribución teórica (alfa = 0.05)."
else:
    conclusion_ajuste = (
        "Sí, en general. Edad, glucosa, IMC y antecedentes no rechazan H0. Rechazos a alfa = 0.05: "
        + "; ".join(rechazos_005)
        + (". Tras la corrección de Bonferroni (alfa = %.4f) ninguna prueba rechaza H0, por lo que "
           "esos rechazos son compatibles con el error tipo I esperado al hacer varias pruebas "
           "(los datos se generaron exactamente con esas distribuciones)." % ALFA_BONF
           if not rechazos_bonf else
           ". Tras la corrección de Bonferroni persisten rechazos; revisar la tabla."))
conclusion_ajuste += (" La edad rechaza normalidad SIN truncar (Shapiro), lo cual es lo esperado "
                      "porque fue generada como normal truncada en [18, 90].")
resultados = {
    "semilla": SEED,
    "n_registros": int(len(df)),
    "variables_predictoras": FEATURES,
    "distribuciones_teoricas": PARAM,
    "regla_etiqueta": {
        "formula": "z = -11.5 + 0.03*edad + 0.05*glucosa + 0.08*imc + 0.015*presion + 1.0*antecedentes + e, "
                   "e ~ N(0, 0.75); p = 1/(1+exp(-z)); diabetes = 'Sí' si p >= 0.5",
        **REGLA,
    },
    "proporcion_clases": {"conteo": {k: int(v) for k, v in conteo.items()},
                          "proporcion": {k: float(v) for k, v in proporcion.items()}},
    "estadisticas_descriptivas": desc.round(4).to_dict(orient="index"),
    "pruebas_bondad_ajuste": pruebas,
    "alfa": ALFA,
    "alfa_bonferroni": round(ALFA_BONF, 5),
    "conclusion_bondad_ajuste": conclusion_ajuste,
    "particion": {"train": int(len(X_train)), "test": int(len(X_test)), "estratificada": True,
                  "prop_si_train": round(float(y_train.mean()), 4),
                  "prop_si_test": round(float(y_test.mean()), 4)},
    "parametros_modelos": {"Regresión Logística": {**PARAMS_LR, "escalado": "StandardScaler"},
                           "Random Forest": {**PARAMS_RF, "escalado": "ninguno"}},
    "metricas_modelos": resultados_modelos,
    "referencia_regla_sin_ruido_test": {k: round(v, 4) for k, v in m_oraculo.items()},
    "mejor_modelo_f1": mejor,
    "coeficientes_lr_escalados": coef_lr,
    "importancias_rf": imp_rf,
    "correlacion_con_diabetes": corr["diabetes (1=Sí)"].drop("diabetes (1=Sí)").round(4).to_dict(),
    "experimento_tamano": resumen.to_dict(orient="records"),
}


def _a_json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


with open(os.path.join(BASE, "resultados.json"), "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2, default=_a_json)
print(f"[6] Mejor F1: {mejor}. Resultados guardados en resultados.json y figuras/.")
