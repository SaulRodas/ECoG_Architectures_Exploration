# Plan experimental independiente: ablaciones de preprocesamiento y arquitectura para decodificación de flexión de dedos con ECoG

**Repositorio:** proyecto nuevo, independiente de cualquier implementación anterior
**Tipo de estudio:** exploratorio, orientado a un artículo científico
**Datos:** BCI Competition IV — Dataset 4; tres sujetos (S1, S2, S3)
**Objetivo:** analizar qué decisiones de preprocesamiento y arquitectura afectan la correlación de Pearson de la flexión continua de cinco dedos.
**Diseño:** 9 configuraciones entrenadas × 3 sujetos × 5 folds = **135 entrenamientos** (más 2 configuraciones de preprocesamiento verificadas por equivalencia, sin entrenar).
**Optimizador común:** Adam con L2 acoplada.
**Ejecución:** Google Colab, mediante `scripts/colab_experimentos.ipynb`.
**Fuera de alcance:** Optuna/TPE, ensambles, búsqueda de nuevas funciones de pérdida, selección de hiperparámetros tras observar resultados, evaluación adicional en el test oficial y los sistemas de referencia adaptados R-FF y R-DTC.

> Este archivo es la especificación **completa del experimento**: un equipo debe poder implementar el repositorio sin acceder al código ni a los documentos de un proyecto anterior. Se conservan prácticas simples de reproducibilidad (configuración, semillas, versiones y resultados por fold), pero **no** se exigen hashes, manifiestos complejos, etapas formales de aprobación ni pruebas de cientos de casos.

---

## 0. Registro de cambios respecto a la versión inicial del plan

Todos estos cambios se decidieron **antes** de entrenar cualquier modelo.

| # | Cambio | Motivo |
|---|---|---|
| 1 | Retardo ECoG(t) → guante(t + **20 ms**) = **2 columnas** a 100 Hz (antes 200 ms / 20 columnas). | Decisión del autor. |
| 2 | La campaña se ejecuta en **Google Colab** con `scripts/colab_experimentos.ipynb`, que lanza la campaña en segundo plano y permite monitorearla. | Infraestructura disponible. |
| 3 | **No se guardan pesos** (`model_last.pt`); `predictions.npz` conserva las predicciones exactas de la época 50. | Espacio en Drive (~9 GB evitados). |
| 4 | **TF32 permitido** en GPUs Ampere o posteriores: los tensores siguen en float32, pero las convoluciones y multiplicaciones de matrices usan TF32. Cada corrida registra la GPU y el estado de TF32. Se recomienda una sola clase de GPU para toda la campaña. | Coste de cómputo en Colab. |
| 5 | **P2 y P4 no se entrenan.** Bajo la cadena común son equivalentes a P1 por construcción (sección 5.1); se verifica numéricamente en cada sujeto/fold. | Evitar evaluaciones repetidas. |
| 6 | **Arquitecturas A0–A5 redefinidas** según la especificación de la sección 6. A3/A5 usan ConvTranspose1d depthwise; A4/A5 producen [512,512,256,128,64] canales y la cabeza recibe 64. | Especificación explícita sin cambios implícitos de canales. |
| 7 | **R-FF y R-DTC se eliminan** del alcance. | Decisión del autor. |
| 8 | Presupuesto: **135 entrenamientos** (antes 195). | Consecuencia de 5 y 7. |

---

## 1. Pregunta científica y antecedentes que motivan el estudio

**Pregunta:** manteniendo datos, entrenamiento y evaluación comparables, ¿cómo cambian las correlaciones de Pearson al modificar la referencia/normalización de ECoG y componentes concretos de una red encoder–decoder 1D?

Las dos referencias que motivan los componentes estudiados son:

- **FingerFlex** (Lomtev et al.): encoder–decoder convolucional con *skip connections*, LayerNorm, GELU, dropout y aumento de resolución por interpolación.
- **DTCNet** (Wang et al.): encoder–decoder con convoluciones dilatadas y convoluciones transpuestas, LayerNorm, GELU y dropout.

El modelo de control de **este** estudio (A0/P0) combina una U-Net 1D con convoluciones dilatadas, interpolación lineal, BatchNorm1d y dropout 0.3. Su configuración se especifica íntegramente más adelante; **no se importa de otro repositorio**.

### Antecedentes del estudio que originó estas hipótesis

Los siguientes hechos sirven solo para explicar por qué se diseñó el nuevo experimento:

1. Una implementación anterior utilizó CMR, características Morlet de 40–200 Hz y una U-Net dilatada con AdamW. **Aquí se vuelve a implementar todo desde cero y se utiliza Adam + L2.**
2. La revisión de ECoG de aquel trabajo había señalado como problemáticos los canales **S1-54, S2-20 y S2-37** (índices comenzando en cero).
3. En una evaluación previa del test oficial se detectó una alteración extrema en **S3-49**. Por ello, esta nueva investigación establece de antemano su exclusión y añade un tratamiento uniforme de amplitudes extremas en la señal cruda.
4. Dado que se conoce ese hallazgo procedente del test oficial, **este trabajo es exploratorio e informado por observaciones previas**. No se afirmará que excluir S3-49 sea una decisión obtenida de una evaluación ciega o que los resultados constituyan una validación confirmatoria independiente.

No se usarán modelos, checkpoints, archivos procesados, índices de pipeline, escaladores ni resultados numéricos del repositorio anterior.

---

## 2. Datos y validación cruzada

### 2.1. Datos de entrada

BCI Competition IV, Dataset 4 contiene tres sujetos con ECoG durante movimientos de cinco dedos.

| Sujeto | Canales ECoG originales | Canales excluidos | Canales utilizados |
|---|---:|---|---:|
| S1 | 62 | `[54]` | **61** |
| S2 | 48 | `[20, 37]` | **46** |
| S3 | 64 | `[49]` | **63** |

Todos los índices de canal son **base cero** y corresponden al orden del archivo original; no representan necesariamente proximidad anatómica.

Se leerán únicamente `train_data` y `train_dg` de los archivos originales `subX_comp.mat`. Cada sujeto cuenta con:

- 400 s de desarrollo (400 000 muestras de ECoG a 1000 Hz).
- Cinco trayectorias del guante, originalmente adquiridas a 25 Hz y almacenadas en una rejilla de 1000 filas/s con retención de orden cero. Se verificó que los cambios ocurren siempre en filas múltiplo de 40 (fase 0) en los tres sujetos.
- Objetivo de regresión: pulgar, índice, medio, anular y meñique.

El conjunto `test_data` y las etiquetas del test oficial **no intervienen en los entrenamientos ni en su selección**.

### 2.2. Cinco folds temporales por sujeto

Se divide cada registro de 400 s en cinco bloques **cronológicos de 80 s**:

| Fold | Intervalo validado | Entrenamiento |
|---|---|---|
| B1 | 0–80 s | B2 + B3 + B4 + B5 |
| B2 | 80–160 s | B1 + B3 + B4 + B5 |
| B3 | 160–240 s | B1 + B2 + B4 + B5 |
| B4 | 240–320 s | B1 + B2 + B3 + B5 |
| B5 | 320–400 s | B1 + B2 + B3 + B4 |

- Cada bloque se preprocesa **separadamente**: ningún filtro, interpolador o ventana cruza una frontera entre bloques.
- Las estadísticas para limpiar, centrar o escalar la señal se ajustan **solo con los bloques de entrenamiento del fold**: las de la señal cruda (limpieza y P) con todas sus muestras; los escaladores de características con las columnas que cubren las ventanas de entrenamiento.
- Se descartan los bordes sin contexto completo de filtrado (125 columnas a 100 Hz por extremo = 825 + 165 + 139 + 121 muestras de semilongitud de notch, pasabanda, Morlet a 40 Hz y antialias).
- Se utiliza una **purga adicional de 256 columnas a 100 Hz (2.56 s) en el entrenamiento junto al bloque validado**. La validación conserva sus tiempos elegibles.
- Esta modalidad es **validación cruzada temporal por bloques**, no *walk-forward*; el entrenamiento puede contener bloques posteriores al bloque validado.

Las mismas particiones, máscaras temporales, semillas por fold y reglas de evaluación se utilizan en todas las variantes.

---

## 3. Limpieza ECoG desde la señal cruda: solución para canales dañados

**Decisión central:** eliminar canales persistentemente problemáticos **antes de calcular CMR, los filtros o Morlet** y añadir una protección simple frente a transitorios extremos en los canales restantes. Esta limpieza se aplica **por igual** a todas las configuraciones; no es una variable que se optimice.

### 3.1. Exclusión preventiva y permanente

Inmediatamente después de cargar ECoG cruda:

```python
EXCLUDED_CHANNELS = {
    "S1": [54],
    "S2": [20, 37],
    "S3": [49],
}
```

Se retiran esos canales en **todos los bloques y folds**, tanto para entrenamiento como para validación. No se dejan como canales de valor cero: se eliminan del tensor y se ajustan las dimensiones de la capa de entrada al número efectivo de electrodos.

**Por qué así en S3:** el canal 49 exhibió una contaminación extrema en la evaluación histórica y podría producir una potencia Morlet y una salida de red enormes. Su eliminación previa evita que participe en la referencia espacial, los filtros, las características y el modelo. No se intenta reconstruir un electrodo cuya señal es poco confiable.

### 3.2. Protección frente a nuevos picos extremos

Para los canales conservados se aplica una **limitación robusta de amplitud en la ECoG cruda**, siempre antes de la operación P correspondiente.

En cada fold y canal conservado `c`, estimar **solo con los bloques de entrenamiento**:

\[
m_c=\operatorname{mediana}(x_c),\qquad
s_c=1.4826\,\operatorname{mediana}(|x_c-m_c|).
\]

Luego transformar tanto entrenamiento como validación con los mismos parámetros:

\[
x_c^{\mathrm{limpio}}(t)=
\operatorname{clip}(x_c(t),\,m_c-25s_c,\,m_c+25s_c).
\]

- `25` es un umbral **fijo y deliberadamente conservador**, no seleccionado buscando el mayor Pearson.
- Si `s_c` es nulo o casi nulo se usa un mínimo numérico (`mad_floor`) y el canal se registra en el reporte de calidad.
- Se guarda para cada sujeto y fold cuántas muestras se recortaron por canal y las amplitudes extremas antes y después. La señal original solo existe en `data/raw/`.
- Nunca se calculan `m_c` o `s_c` con el bloque de validación ni se reestiman por variante P.
- La limitación evita explosiones de amplitud; **no recupera información neuronal perdida** ni garantiza corregir todos los artefactos posibles.

La elección de excluir S3-49 es informada por el test histórico; el clip robusto es una protección adicional general. Ambos se describen transparentemente como parte de la preparación exploratoria de señales.

### 3.3. Revisión sencilla de calidad

Antes de los entrenamientos (`python scripts/run_experiments.py --quality`) se produce por sujeto:

1. Gráfica de ECoG cruda y limpiada para unos pocos segmentos representativos.
2. Tabla de canales excluidos y número de canales restantes.
3. Fracción de muestras recortadas por canal y fold.
4. Confirmación de que las señales producidas son finitas (y, durante la campaña, de que cada tensor de características lo es).

Si un canal adicional presenta una anomalía, se registra; **no se cambia la lista después de ver correlaciones**. Cualquier análisis adicional se considera un experimento distinto.

---

## 4. Procesamiento común de las señales

```text
ECoG cruda (1000 Hz)
   → división en bloques/folds
   → exclusión fija de canales dañados
   → clip robusto ajustado solo con train
   → operación de entrada P0–P5
   → FIR pasabanda + notch
   → potencia Morlet
   → antialias + reducción a 100 Hz
   → escalado de características ajustado con train
   → ventanas de 256 muestras
   → encoder–decoder 1D
   → predicción de cinco dedos
```

| Etapa | Especificación común |
|---|---|
| ECoG | 1000 Hz; separación de bloques antes de filtrar |
| Filtro pasabanda | 40–200 Hz; FIR Hamming de fase cero (`mne.filter.create_filter`, firwin), 331 coeficientes, transiciones de 10 y 20 Hz |
| Notch | 60, 120 y 180 Hz; FIR band-stop de 1651 coeficientes con las convenciones de MNE (anchos f/200, transición de 2 Hz por lado), aplicado sin atravesar bloques |
| Morlet | 40 frecuencias logarítmicas entre 40–200 Hz, 7 ciclos, `zero_mean=True` (`mne.time_frequency.morlet`) |
| Característica | Potencia espectral `abs(W)**2`, sin logaritmo |
| Antialias de potencia | FIR Kaiser, paso hasta 35 Hz y rechazo desde 50 Hz, 60 dB, 243 coeficientes |
| Frecuencia de trabajo | 100 Hz |
| Guante | Recuperar datos efectivos de 25 Hz (una muestra cada 40 filas) e interpolar con spline cúbico por bloque |
| Retardo | ECoG(t) → guante(t + **0.02 s**), **2 columnas** a 100 Hz |
| Escalado X | `RobustScaler(quantile_range=(10,90), unit_variance=True)` por canal y frecuencia, ajustado solo en train |
| Escalado Y | `MinMaxScaler` por dedo, ajustado solo en train |
| Ventanas de entrenamiento | Longitud 256 (2.56 s), stride 1; no cruzar bloques ni regiones inválidas |
| Ventanas de validación | Longitud 256, stride 64, incluir la ventana final si hace falta |
| Reconstrucción | Promedio de predicciones solapadas (*overlap-add*) sobre los instantes válidos |

Todas las convoluciones se calculan en float64 por FFT en modo *same* con relleno de ceros dentro de cada bloque (equivalente a MNE; una prueba lo verifica) y los bordes contaminados se descartan.

El retardo de **20 ms es una elección experimental propia y fija**; no se optimizará en este estudio.

Se mantienen idénticos filtros, frecuencias, escaladores, retardo, longitud de ventana y criterios temporales en todas las variantes. Las variantes P modifican únicamente la operación de normalización de **ECoG cruda, después de la limpieza y antes del filtrado**.

---

## 5. Familia P: normalización y referencia espacial de ECoG

La arquitectura de todas las variantes P es **A0** (sección 6).

- `mediana_espacial(t)`: mediana entre los canales disponibles en un mismo instante; esto es **CMR**.
- `mediana_temporal(c)`: mediana a lo largo de los tiempos de entrenamiento, calculada para un canal.
- `zscore`: \((x_c-\mu_{c,\text{train}})/\sigma_{c,\text{train}}\), con media/desviación (ddof = 0) ajustadas en train.

| ID | Transformación aplicada a ECoG ya limpiada | CMR espacial | Estadísticos temporales | ¿Se entrena? |
|---|---|---|---|---|
| **P0 = A0** | `x - mediana_espacial(x)` | Sí, sobre señal cruda limpiada | Ninguno | Sí (control) |
| **P1** | `x` | No | Ninguno | Sí |
| **P2** | `x - mediana_temporal_train(c)` | No | Mediana por canal | **No: ≡ P1** |
| **P3** | `zscore_train(x) - mediana_espacial(zscore_train(x))` | Sí, después de z-score | Media y desviación por canal | Sí |
| **P4** | `zscore_train(x) - mediana_temporal_train(zscore)` | No | Media, desviación y mediana por canal | **No: ≡ P1** |
| **P5** | `u - mediana_espacial(u)`, donde `u = zscore_train(x) - mediana_temporal_train(zscore)` | Sí | Media, desviación y mediana por canal | Sí |

- **P0** es el control de este repositorio y **no** hereda un archivo procesado anterior.
- **P3** representa una versión operacional de estandarización por canal seguida de CMR, inspirada en el código público de FingerFlex.
- **P4** es una interpretación implementable de la descripción textual de normalización temporal en DTCNet.
- Para desviaciones estándar nulas se usa un denominador mínimo fijo (`std_floor`), igual en train y validación.
- **No se ajusta nada con el fold validado.**

### 5.1. Equivalencia de P2 y P4 con P1 (por qué no se entrenan)

- **P2 ≡ P1.** P2 resta una constante por canal. El pasabanda de 40–200 Hz y la Morlet con `zero_mean=True` anulan cualquier constante en la región válida, de modo que la potencia es la misma.
- **P4 ≡ P1.** P4 = (x − mediana_c)/σ_c = P2/σ_c. La potencia Morlet queda multiplicada por 1/σ_c², un factor positivo por canal que el `RobustScaler` ajustado por canal×frecuencia cancela exactamente (centro y escala se multiplican por el mismo factor).
- Entrenarlas repetiría la evaluación de P1 con la misma semilla y el mismo orden de batches, por lo que sus diferencias solo medirían ruido numérico.
- **Regla pre-especificada:** en cada sujeto/fold se calculan los tensores escalados de P2 y P4 y se comparan con los de P1 en la región válida (`results/summary/feature_equivalence.csv`). Se declaran equivalentes si la diferencia relativa máxima es ≤ 1e-5. Si alguna no lo cumple se señala en el reporte; no se entrena dentro de esta campaña.
- P3 y P5 **no** son equivalentes: solo difieren en el offset por canal (media frente a mediana) antes del CMR, que sí altera la mediana espacial. Ambos se entrenan y el mismo diagnóstico reporta su cercanía con carácter informativo.

---

## 6. Familia A: modificaciones arquitectónicas

Se implementa una clase configurable `EncoderDecoder1D`, con bloques reutilizables para stem, encoder, decoder y cabeza. Se permiten diferencias explícitas entre arquitecturas, pero no cambios implícitos en sus canales o en el orden de las operaciones.

### 6.1. Parámetros configurables

`stem_channels`, `encoder_channels`, `kernel_sizes`, `dilations`, `decoder_channels`, `decoder_kernel_sizes`, `normalization` (`batch | layer`), `dropout`, `upsample` (`interp | convtranspose`), `decoder_order`, `convtranspose_mode` y `skip_position`.

Solo se implementan los valores que usan A0–A5:

- `decoder_order`: `upsample_conv_concat` o `upsample_concat_conv`.
- `convtranspose_mode`: `depthwise`.
- `skip_position`: `pre_pool`.

Los órdenes `conv_upsample_concat` y `concat_conv_upsample`, el modo `standard` y los skips en la entrada de bloque existían para R-FF/R-DTC y quedan fuera de alcance; la validación de la configuración los rechaza.

### 6.2. Entrada, salida y configuración común

- Entrada `(B, C, 40, 256)`, aplanada a `(B, C·40, 256)`; salida `(B, 5, 256)`, sin activación final.
- *Stem*: `Conv1d(C·40 → 64, kernel=3)` → normalización → GELU → Dropout.
- Encoder: canales `[64,64,128,256,512,512]` (stem + 5 bloques), kernels `[7,7,5,5,5]`, dilataciones `[1,2,3,1,2]`, `MaxPool1d(2)`; cada bloque es `Conv1d` → normalización → GELU → Dropout y el *skip* se guarda **antes del pooling**.
- Decoder: salidas de convolución `[512,512,256,128,64]` del nivel más profundo al más superficial, kernels espejo `[5,5,5,7,7]` y dilatación 1.

Convenciones comunes:

- Convoluciones sin bias cuando están seguidas inmediatamente de normalización; cabeza final con bias.
- Dropout después de GELU.
- LayerNorm sobre los canales de características en cada instante temporal.
- Padding `d*(k-1)//2` para kernels impares y stride 1.
- La normalización y el dropout de la variante aplican también al stem.
- Las dimensiones esperadas de cada skip se derivan de la configuración y se validan al concatenar; no se recorta, rellena ni proyecta para resolver incompatibilidades.

### 6.3. Matriz A0–A5

| Modelo | Normalización | Dropout | Upsampling | Orden del decoder |
|---|---|---|---|---|
| A0 | BatchNorm | 0.3 | Interpolación | Upsample → Conv → Concat |
| A1 | LayerNorm | 0.3 | Interpolación | Upsample → Conv → Concat |
| A2 | BatchNorm | 0.1 | Interpolación | Upsample → Conv → Concat |
| A3 | BatchNorm | 0.3 | ConvTranspose depthwise | Upsample → Conv → Concat |
| A4 | BatchNorm | 0.3 | Interpolación | Upsample → Concat → Conv |
| A5 | LayerNorm | 0.1 | ConvTranspose depthwise | Upsample → Concat → Conv |

**Flujo de canales.**

- En A0–A3 el decoder convoluciona y luego concatena. Las dimensiones después de cada concatenación son `[1024,1024,512,256,128]` y la cabeza recibe 128 canales.
- En A4–A5 la concatenación ocurre antes de la convolución. Las convoluciones de fusión reciben `[1024,1024,768,384,192]` canales y producen `[512,512,256,128,64]`; la cabeza recibe 64.
- Los anchos distintos se registran; no se añaden proyecciones ocultas para igualarlos.

**Convolución transpuesta (A3, A5).**

```python
nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
                   groups=in_channels, bias=False)
```

Realiza un upsampling aprendible depthwise y limita el incremento de parámetros. No debe identificarse con la convolución transpuesta estándar de DTCNet.

**Parámetros entrenables (S3, 63 × 40 entradas):**

| Modelo | Parámetros |
|---|---|
| A0, A1, A2 | ≈ 8.52 M |
| A3 | ≈ 8.53 M |
| A4 | ≈ 9.36 M |
| A5 | ≈ 9.37 M |

Entre sujetos solo varía el stem. Cada corrida guarda en `architecture.json` los parámetros por sección y las dimensiones reales por etapa.

Todos los modelos se inicializan desde cero; no se reutilizan pesos entre folds ni variantes. A5 no permite atribuir su efecto a una sola operación: es una combinación propuesta desde el principio.

---

## 7. Presupuesto y condiciones de entrenamiento

### 7.1. Presupuesto total

| Familia | Configuraciones entrenadas | Sujetos | Folds por sujeto | Entrenamientos |
|---|---:|---:|---:|---:|
| Control P0 = A0 | 1 | 3 | 5 | **15** |
| P1, P3, P5: preprocesamiento | 3 | 3 | 5 | **45** |
| A1–A5: arquitectura | 5 | 3 | 5 | **75** |
| **Total** | **9** | **3** | **5** | **135** |

P2 y P4 se verifican por equivalencia (sección 5.1) sin entrenamiento. Cada entrenamiento corresponde a **una variante × un sujeto × un fold**, con una semilla fija. No hay TPE, búsquedas internas ni repeticiones por semillas en esta fase.

### 7.2. Hiperparámetros comunes

| Parámetro | Valor |
|---|---|
| Optimizador | **Adam + L2 acoplada**, no AdamW |
| Learning rate | `8.42e-5` |
| `weight_decay` (L2) | `1e-6` |
| `betas` | `(0.9, 0.999)` |
| `eps` | `1e-8` |
| Scheduler | Ninguno |
| Batch | 128 (`drop_last=True`) |
| Épocas | **50** |
| Early stopping | No |
| Checkpoint para comparar | **Última época (50)** |
| Pérdida | `0.5*MSE + 0.5*(1 - coseno_temporal)` |
| Precisión numérica | Tensores float32; TF32 permitido en convoluciones/matmul de GPUs Ampere+ |
| Métrica principal | Pearson macro de las cinco trayectorias reconstruidas |

La similitud coseno se calcula a lo largo de las 256 posiciones temporales **por dedo y ventana** y después se promedia sobre dedos y batch, con protección `eps` para ventanas degeneradas. No se añade otra penalización L2 manual; en versiones de PyTorch que ofrecen `decoupled_weight_decay`, se usa `False`. Si la pérdida de entrenamiento deja de ser finita, el entrenamiento se detiene y la evaluación se marca inválida con su motivo.

Los hiperparámetros permanecen iguales aunque algunos modelos no estén perfectamente optimizados para ellos: se comparan componentes bajo un régimen común.

### 7.3. Semillas

```python
SUBJECT_INDEX = {"S1": 0, "S2": 1, "S3": 2}
FOLD_INDEX = {"B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4}
seed = 42 + 5 * SUBJECT_INDEX[subject] + FOLD_INDEX[fold]
```

Se fijan las semillas de Python, NumPy, CPU/CUDA de PyTorch y del orden de batches (idéntico entre variantes del mismo sujeto/fold). Se guardan las versiones y el modelo de GPU. Se activa el modo determinista cuando es viable (`cudnn.deterministic`, `use_deterministic_algorithms(warn_only=True)`), sin exigir identidad bit a bit entre plataformas.

---

## 8. Métricas y comparación de resultados

Por cada sujeto, fold y dedo, primero se reconstruye la trayectoria mediante *overlap-add* y después se calcula:

\[
r_{s,f,d}=\mathrm{corr}(y_{s,f,d}, \hat y_{s,f,d}),\qquad
r_{s,f}^{\mathrm{macro}}=\frac{1}{5}\sum_{d=1}^{5} r_{s,f,d},\qquad
\bar r_s=\frac{1}{5}\sum_{f=1}^{5}r_{s,f}^{\mathrm{macro}}.
\]

La **media global** da el mismo peso a cada uno de los tres sujetos. MSE, RMSE y CCC se reportan en unidades originales del guante.

**Para cada uno de los 135 fits se guarda** (`results/runs/<variante>/<sujeto>/<fold>/`):

- `config.json`: ID de variante, sujeto, fold, semilla, configuración completa, versiones, GPU, TF32 y commit.
- `metrics.json`: Pearson macro y por dedo, MSE, RMSE y CCC, número de parámetros, tiempo de entrenamiento, fracción recortada y validez.
- `history.csv`: pérdida y Pearson de validación por época.
- `architecture.json`: parámetros por sección y dimensiones por etapa.
- `predictions.npz`: trayectorias reales y predichas de la época 50.
- `status.json`: completada, interrumpida o fallida, con su motivo si aplica (`error.txt` en fallos).

Comparaciones:

- `P1`, `P3`, `P5` frente a `P0`: cambio de preprocesamiento con arquitectura A0 constante. P2 y P4 se reportan como "equivalentes a P1 (verificado)".
- `A1–A4` frente a `A0`: cambio aislado de un componente con preprocesamiento P0 constante.
- `A5`: variante multicomponente; comparación descriptiva, sin atribuir mejora a una única capa.

Se muestran **diferencias de Pearson apareadas por sujeto y fold** y los cinco folds completos. **No** se calcula Pearson por minibatch ni se tratan las ventanas solapadas como muestras independientes. Con tres sujetos y una semilla por fold, las diferencias pequeñas se interpretan con prudencia.

Si aparece una predicción no finita o constante, se registra como evaluación inválida; no se elimina ese dedo o fold silenciosamente (las tablas incluyen conteos de inválidas).

---

## 9. Estructura del repositorio

```text
ecog-architecture-ablations/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/experiment.yaml
├── data/raw/                          # archivos .mat; fuera de Git
├── docs/
│   ├── plan_experimental.md           # este documento
│   └── referencias_modelos.md
├── src/ecog_ablations/
│   ├── config.py                      # carga y validación del YAML
│   ├── data.py                        # lectura de train_data/train_dg y folds
│   ├── cleaning.py                    # exclusiones fijas + clip robusto
│   ├── preprocessing.py               # P0-P5, filtros, Morlet, escalado, ventanas, equivalencias
│   ├── models.py                      # A0-A5
│   ├── training.py                    # Adam+L2, pérdida, semillas
│   ├── evaluation.py                  # overlap-add y métricas
│   ├── runner.py                      # campaña reanudable y control de calidad
│   └── reporting.py                   # tablas, figuras y monitor
├── scripts/
│   ├── run_experiments.py
│   ├── make_report.py
│   └── colab_experimentos.ipynb       # ejecución y monitoreo en Colab
├── tests/test_smoke.py
└── results/                           # generado; no se versiona
    ├── runs/<variant>/<subject>/<fold>/
    ├── quality/
    └── summary/
        ├── results_all.csv
        ├── results_subjects.csv
        ├── paired_deltas.csv
        ├── paired_deltas_summary.csv
        ├── feature_equivalence.csv
        └── figures/
```

Comandos:

```bash
python scripts/run_experiments.py --quality
python scripts/run_experiments.py --all
python scripts/make_report.py
```

`runner.py` no vuelve a entrenar una combinación terminada correctamente, reentrena las interrumpidas y conserva las fallidas para diagnóstico (solo se reintentan con `--retry-failed`). Los intentos previos se archivan en `attempts/`.

### Comprobaciones mínimas antes de ejecutar (`pytest`)

1. **Datos:** lectura de S1/S2/S3, canales `61/46/63`, cinco bloques, cinco dedos y retención de orden cero alineada.
2. **Limpieza:** un pico sintético queda limitado; un canal excluido no entra a CMR ni al modelo.
3. **Sin contaminación:** alterar solo validación no cambia las estadísticas ni las ventanas de entrenamiento; las ventanas respetan bloques, purga y retardo de 2 columnas.
4. **Arquitecturas:** A0–A5 con entradas sintéticas de S1, S2 y S3 producen `(B,5,256)`, retropropagan sin NaN/Inf, concatenan en la posición indicada y tienen los anchos de la sección 6.
5. **Métrica:** una predicción idéntica produce Pearson 1; una predicción constante se marca inválida.
6. **DSP:** la potencia Morlet coincide con `mne.time_frequency.tfr_array_morlet`; P2 y P4 coinciden con P1.

---

## 10. Orden práctico de ejecución

1. Control de calidad (`--quality`).
2. *Smoke* real en Colab (`--smoke`): 1 época × 2 batches para medir tiempos.
3. Campaña completa en orden fold → sujeto → preprocesamiento → variante. Cada conjunto de características se calcula una vez por sujeto/fold y lo reutilizan todas las variantes que lo usan; al terminar B1 ya hay resultados de las 9 configuraciones en los tres sujetos.
4. Reporte (`make_report.py`). No se vuelven a optimizar parámetros a partir de esos resultados dentro de esta campaña.

En caso de error técnico se corrige el software y se repiten **solo las corridas afectadas**, dejando constancia breve en `README.md` (sección "Notas de ejecución").

---

## 11. Material que conviene conservar para el artículo

**En el artículo:**

- Pregunta de investigación y carácter exploratorio.
- Tratamiento de canales dañados, incluida **S3-49 y su origen en una observación previa de test**.
- Tabla de P0–P5 (con la equivalencia de P2/P4) y tabla de A0–A5.
- Los cinco folds, presupuesto de 135 entrenamientos, Adam + L2, TF32 y definición de Pearson.
- Resultados por sujeto, diferencias respecto a los controles y limitaciones.

**En el repositorio:**

- Este protocolo, configuración YAML, código y `requirements.txt`.
- Resultados individuales de los 135 entrenamientos, incluidos los negativos.
- Registro de canales excluidos, estadísticas de limpieza por fold, parámetros de entrada y verificación de equivalencias.
- Figuras y scripts que regeneran las tablas.
- Semillas, versiones, GPU, conteos de parámetros y errores de ejecución.

**Límite de interpretación:** la validación cruzada permite comparar alternativas con menor dependencia de un solo bloque, pero los folds comparten datos de entrenamiento. Además, la limpieza preventiva de S3 está motivada por conocimiento previo del test. El experimento ofrece evidencia **exploratoria y comparativa**, no una nueva evaluación independiente del rendimiento final en test.

---

## Referencias

1. BCI Competition IV, Dataset 4: https://bbci.de/competition/iv/
2. Lomtev, V., Kovalev, A. y Timchenko, A. *FingerFlex: Inferring Finger Trajectories from ECoG signals*. https://arxiv.org/abs/2211.01960
3. Código público de FingerFlex: https://github.com/Irautak/FingerFlex
4. Wang, F., Luo, Z., Lv, W. y Zhu, X. (2025). *DTCNet: finger flexion decoding with three-dimensional ECoG data*. https://doi.org/10.3389/fncom.2025.1627819
5. PyTorch: reproducibilidad y optimizador Adam. https://pytorch.org/docs/stable/notes/randomness.html ; https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
6. MNE-Python: filtros FIR y transformada Morlet. https://mne.tools/stable/generated/mne.filter.create_filter.html ; https://mne.tools/stable/generated/mne.time_frequency.tfr_array_morlet.html
