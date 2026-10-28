# Ablaciones de preprocesamiento y arquitectura — ECoG BCI IV-4

Experimento exploratorio: ¿cómo cambia el Pearson de la flexión continua de cinco dedos al modificar la referencia/normalización de la ECoG (familia P) y componentes concretos de un encoder–decoder 1D (familia A)? La especificación completa es [docs/plan_experimental.md](docs/plan_experimental.md) y [configs/experiment.yaml](configs/experiment.yaml) es la única fuente de parámetros.

**Campaña:** 9 configuraciones × 3 sujetos × 5 folds temporales = **135 entrenamientos**.

| Familia | Configuraciones | Mantiene fijo |
|---|---|---|
| Control | P0 = A0 (CMR + U-Net dilatada, BatchNorm, dropout 0.3, interpolación) | — |
| P | P1 (sin referencia), P3 (z-score + CMR), P5 (z-score − mediana + CMR) | arquitectura A0 |
| A | A1 (LayerNorm), A2 (dropout 0.1), A3 (ConvTranspose depthwise), A4 (concat antes de conv), A5 (combinación) | preprocesamiento P0 |

P2 y P4 **no se entrenan**: bajo la cadena común son equivalentes a P1. Se verifica numéricamente en cada sujeto/fold (`results/summary/feature_equivalence.csv`; véase la sección 5.1 del plan).

Cambios respecto al plan inicial: retardo de 20 ms, Colab, sin pesos guardados, TF32 permitido, sin P2/P4 entrenadas y sin R-FF/R-DTC. Están detallados en la sección 0 del plan.

## Uso local

```bash
pip install -r requirements.txt
# copiar sub1_comp.mat, sub2_comp.mat y sub3_comp.mat a data/raw/ (no se versionan)
python -m pytest -q                               # comprobaciones mínimas (~15 s en CPU)
python scripts/run_experiments.py --quality       # control de calidad → results/quality/
python scripts/run_experiments.py --smoke         # 1 época × 2 batches en S1/B1 → results/_smoke/
python scripts/run_experiments.py --all           # campaña completa (reanudable)
python scripts/make_report.py                     # tablas y figuras → results/summary/
```

Otras opciones:

- `--subjects S1 S2`, `--folds B1` y `--variants P0 A3` acotan la campaña.
- `--dry-run` muestra el plan y su estado.
- `--retry-failed` reintenta las corridas fallidas.
- `--data-dir` y `--results-dir` cambian las rutas.

## Uso en Google Colab desde VS Code

El notebook [scripts/colab_experimentos.ipynb](scripts/colab_experimentos.ipynb) se abre en VS Code y se ejecuta en un servidor de Colab con la extensión oficial *Google Colab* (kernel remoto). El `.ipynb` es local, pero las celdas corren en la máquina de Colab.

1. Sube el repositorio a GitHub (los datos y `results/` están en `.gitignore`):
   ```bash
   git add -A && git commit -m "Implementación del experimento"
   git remote add origin https://github.com/<usuario>/ecog-architecture-ablations.git
   git push -u origin main
   git tag campana-v1 && git push origin campana-v1   # etiqueta fija para la campaña
   ```
2. Sube los tres `.mat` a Google Drive (por defecto `MyDrive/BCICIV_4_mat`).
3. En VS Code, instala la extensión *Google Colab* y abre el notebook. Elige *Select Kernel → Colab → New Colab Server* y una máquina con GPU.
4. Ajusta la celda 2 (`GIT_URL`, `GIT_REF`, `PRIVATE_REPO`, `DATA_DIR`, `RESULTS_DIR`) y ejecuta las celdas en orden:
   - Celdas 4–6: tests, control de calidad y smoke con estimación de tiempo.
   - Celda 7: lanza la campaña en segundo plano.
   - Celda 8: monitor reejecutable.
   - Celda 9: detener la campaña.
   - Celda 10: reporte.

Particularidades de la extensión:

- **Drive:** `drive.mount()` funciona desde la versión 0.2.1. Si no, usa el comando *Colab: Mount Google Drive to Server...*.
- **Repo privado:** los *Secrets* de Colab (`userdata.get`) no funcionan en la extensión. Con `PRIVATE_REPO = True`, el notebook pide el token con un cuadro de entrada (`getpass`) y no lo guarda.
- **Origen del código:** el paquete se ejecuta desde el commit clonado en el servidor, no desde tu carpeta local. Si cambias `src/`, haz commit y push y vuelve a ejecutar la celda 3.
- **Sin GitHub:** con `GIT_URL = ""` puedes subir `src`, `scripts`, `configs`, `tests` y `pyproject.toml` con clic derecho → *Upload to Colab* y usar `REPO_DIR = "/content"`. En ese caso no queda commit registrado.
- **Conexión y sesiones:** mantén VS Code conectado durante la campaña.
  - Si el servidor se recicla, abre uno nuevo y reejecuta las celdas 1–3 y 7. Las corridas completadas se saltan, la interrumpida se reentrena desde cero y las fallidas se conservan para diagnóstico.
  - Varias sesiones pueden trabajar a la vez sobre el mismo `RESULTS_DIR` si usan sujetos o folds disjuntos.
- **Seguir el log:** desde la terminal experimental de la extensión (*Colab: Open Terminal*) puedes seguirlo con `tail -f`.

**Cómputo estimado** (orientativo; el smoke de la celda 6 da la cifra real):

| GPU | Tiempo estimado |
|---|---|
| T4 | ~25–35 h |
| L4 con TF32 | ~5–9 h |
| A100 con TF32 | ~4–6 h |

Conviene usar la misma clase de GPU en toda la campaña.

## Resultados

```text
results/
├── runs/<variante>/<sujeto>/<fold>/   status.json, config.json, architecture.json, history.csv,
│                                      metrics.json, predictions.npz, attempts/ (intentos previos)
├── quality/                           canales, fase del guante, recortes por canal/fold, parámetros de
│                                      entrada, estadísticas de características, equivalencias, figuras
└── summary/                           results_all.csv, results_subjects.csv, paired_deltas*.csv,
                                       feature_equivalence.csv, figures/
```

Cada `config.json` guarda la configuración completa, la semilla, las versiones de Python/PyTorch/MNE, la GPU, el estado de TF32 y el commit de git.

## Estructura del código

| Módulo | Contenido |
|---|---|
| `src/ecog_ablations/config.py` | carga y validación del YAML, semillas |
| `src/ecog_ablations/data.py` | lectura de `train_data`/`train_dg`, bloques, guante a 100 Hz |
| `src/ecog_ablations/cleaning.py` | exclusión fija de canales y clip robusto |
| `src/ecog_ablations/preprocessing.py` | P0–P5, FIR, Morlet, escalado, ventanas, equivalencias |
| `src/ecog_ablations/models.py` | A0–A5 (`EncoderDecoder1D`) |
| `src/ecog_ablations/training.py` | Adam + L2, pérdida, determinismo |
| `src/ecog_ablations/evaluation.py` | overlap-add y métricas |
| `src/ecog_ablations/runner.py` | campaña reanudable y control de calidad |
| `src/ecog_ablations/reporting.py` | tablas, figuras y monitor del notebook |

## Notas de ejecución

Registra aquí cualquier corrección de software y las corridas repetidas por ese motivo (fecha, cambio, corridas afectadas).

- _(sin incidencias todavía)_
