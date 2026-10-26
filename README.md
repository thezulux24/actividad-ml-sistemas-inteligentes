# Actividad guiada: ML supervisado y no supervisado

**Pontificia Universidad Javeriana Cali** · Sistemas Inteligentes
**Autor:** Brayan David Zuluaga

En esta actividad genero datos sintéticos con distintas distribuciones de probabilidad y los uso en dos problemas:

1. **ML supervisado:** clasificar si un paciente tiene diabetes a partir de su edad, glucosa, IMC, presión arterial y antecedentes familiares. Comparo una **Regresión Logística** con un **Random Forest**.
2. **ML no supervisado:** segmentar clientes de un supermercado según su edad, ingresos, frecuencia de compra y gasto. Uso **K-means**, **K-medoids** y **clustering jerárquico**.

El informe completo está en [`informe/informe.pdf`](informe/informe.pdf).

## Estructura

```
├── informe/
│   ├── informe.tex          # Informe en LaTeX
│   └── informe.pdf          # Informe compilado
├── supervisado/
│   ├── supervisado.py       # Generación de datos, EDA, modelos y métricas
│   ├── datos_diabetes.csv   # 2.000 pacientes sintéticos
│   ├── *.csv / resultados.json
│   └── figuras/
├── no_supervisado/
│   ├── no_supervisado.py    # Generación de datos, EDA, codo/silueta y clustering
│   ├── datos_clientes.csv   # 3.200 clientes sintéticos
│   ├── *.csv / resultados.json
│   └── figuras/
└── requirements.txt
```

## Resultados principales

### Parte 1: diabetes (conjunto de prueba, 20 %)

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Regresión Logística | **0,855** | 0,774 | **0,797** | **0,785** |
| Random Forest | 0,845 | 0,775 | 0,752 | 0,763 |

- Clases: 66,65 % sin diabetes y 33,35 % con diabetes.
- La regresión logística gana porque la etiqueta se construyó con una regla lineal-logística. El Random Forest sobreajusta: 0,98 en entrenamiento.
- Con más datos (500 → 2.000 → 10.000) las métricas se estabilizan, pero no suben más allá de 2.000 registros por el ruido de la etiqueta.

### Parte 2: clientes (k = 4)

| Algoritmo | Silueta | Davies-Bouldin | ARI vs. segmentos reales |
|---|---|---|---|
| K-means | 0,485 | 0,766 | 0,946 |
| K-medoids | 0,485 | 0,762 | 0,940 |
| Jerárquico (Ward) | 0,477 | 0,782 | 0,964 |

Segmentos encontrados: **adultos mayores**, **jóvenes/estudiantes**, **profesionales premium** y **familias**.

## Cómo ejecutarlo

Requiere Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python supervisado/supervisado.py
python no_supervisado/no_supervisado.py
```

Todos los resultados usan la semilla `42`, así que se pueden reproducir exactamente.

Para compilar el informe con [Tectonic](https://tectonic-typesetting.github.io/):

```bash
cd informe
tectonic informe.tex
```
