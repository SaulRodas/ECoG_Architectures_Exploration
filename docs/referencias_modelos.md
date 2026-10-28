# Referencias de modelos y elecciones de implementación

## Fuentes que motivan los componentes estudiados

| Fuente | Enlace | Componentes que inspira en este estudio |
|---|---|---|
| FingerFlex (Lomtev et al.) | https://arxiv.org/abs/2211.01960 · https://github.com/Irautak/FingerFlex | LayerNorm (A1, A5), dropout 0.1 (A2, A5), interpolación lineal + *skips*, z-score + CMR (P3) |
| DTCNet (Wang et al., 2025) | https://doi.org/10.3389/fncom.2025.1627819 | Convoluciones dilatadas `[1,2,3,1,2]`, upsampling aprendido (A3, A5), normalización temporal por canal (P4) |

Este repositorio **no** reimplementa ninguno de los dos modelos. Los sistemas de referencia adaptados (R-FF, R-DTC) se retiraron del alcance antes de entrenar. Las variantes A1–A5 toman componentes aislados de esas fuentes y los incorporan al control A0 bajo un régimen de entrenamiento común.

- La convolución transpuesta de A3/A5 es **depthwise** (`groups = canales`, kernel 4, stride 2, padding 1, sin sesgo). No es la convolución transpuesta estándar de DTCNet.
- La P4 de este estudio es una interpretación operacional del texto de DTCNet. Bajo la cadena común resulta equivalente a P1 (ver `plan_experimental.md`, sección 5.1).

## Elecciones de implementación de A0–A5

Las fuentes o el plan no fijan estas decisiones en detalle, así que se tomaron explícitamente:

1. Toda convolución seguida de normalización va **sin sesgo**: stem, encoder y decoder. La cabeza `Conv1d(·→5, k=1)` lleva sesgo y no tiene activación de salida.
2. Cada bloque sigue el orden `Conv1d → normalización → GELU → Dropout`. La normalización y el dropout de la variante también se aplican al stem.
3. **LayerNorm** normaliza los canales de cada instante: `(B,C,T) → (B,T,C) → LayerNorm(C) → (B,C,T)`.
4. Padding `d·(k−1)//2` (kernels impares) para conservar la longitud. El decoder usa dilatación 1 y kernels espejo `[5,5,5,7,7]`.
5. El *skip* se guarda antes del `MaxPool1d(2)`. Cinco poolings llevan la ventana de 256 a 8 y cinco upsamplings la devuelven a 256.
6. La interpolación es lineal con `align_corners=False` hasta la longitud del *skip*. La ConvTranspose depthwise duplica la longitud exactamente; una discrepancia de longitud produce un error y nunca se recorta.
7. Anchos del decoder:
   - **A0–A3** (upsample → conv → concat): convoluciones `[512,512,256,128,64]`; concatenaciones `[1024,1024,512,256,128]`; cabeza de 128.
   - **A4–A5** (upsample → concat → conv): entradas de fusión `[1024,1024,768,384,192]` → salidas `[512,512,256,128,64]`; cabeza de 64.
8. Inicialización por defecto de PyTorch, siempre desde cero, después de fijar la semilla del sujeto/fold.

`architecture.json`, en cada carpeta de corrida, registra la especificación, los parámetros por sección y las dimensiones reales de cada etapa.
