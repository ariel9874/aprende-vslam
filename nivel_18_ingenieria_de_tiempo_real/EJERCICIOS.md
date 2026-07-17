# Ejercicios — Nivel 18

## 1. GTSAM: el siguiente orden de magnitud (estrella; requiere conda)

La gemela de este nivel (numpy vectorizado) compra un ~4-6× sobre el BA
didáctico. El padre fue más lejos: GTSAM (factor graphs en C++ con
factorización dispersa de verdad) le dio ~2× de fps del SISTEMA completo
(su lección 30). Instálalo (`conda install -c conda-forge gtsam` — en
Windows pip no trae rueda para Python nuevos) y escribe `gtsam_ba.py` con
NUESTRA firma de `bundle_adjustment(...)` (la referencia: `backend/gtsam_ba.py`
del repo padre).

**Objetivo**: el test de equivalencia contra el didáctico... con tolerancias
más flojas que las del acto 1 del examen (¿por qué? GTSAM usa OTRA
parametrización interna y OTRO orden de eliminación — converge al mismo
óptimo por otro camino). Distinguir "equivalencia de resultado" de
"equivalencia de trayectoria" es parte del ejercicio.

## 2. La gemela C++ (difícil; requiere MSVC + CMake + pybind11)

El matching guiado es el segundo punto caliente del perfil (en el padre:
37%). Escríbelo en C++ con pybind11 (la referencia: `cpp/src/guided_match.cpp`
del padre) y mide.

**Objetivo**: el estándar de verificación del padre (su lección 31): la
equivalencia par a par EXACTA — hasta la semántica de desempate (¿qué par
gana cuando dos distancias empatan? np.argmin toma el primero; tu `<` en C++
debe hacer LO MISMO o el test truena con razón). Los bugs de las gemelas
viven en los empates y los bordes.

## 3. El perfil que miente (fácil, muy instructivo)

Repite el perfil del driver pero (a) incluyendo el BA global en el total, y
(b) midiendo sobre los primeros 200 frames en vez de 1000.

**Objetivo**: dos tablas distintas de LA MISMA verdad. El GBA es offline y
se amortiza una vez (¿debe contar como "tiempo del sistema"?); los primeros
frames tienen el mapa chico (el bucle aún no pesa). Un perfil es una foto
CONDICIONADA: saber qué condiciones fija el tuyo es parte de perfilar.

## 4. iSAM2, en concepto (lectura + ensayo corto)

El BA global re-optimiza TODO el mapa aunque el último keyframe solo tocó
una esquina. Lee `backend/gtsam_isam2.py` del padre y su lección 33.

**Objetivo**: explica con tus palabras qué re-usa iSAM2 entre llamadas (el
árbol de Bayes: solo se re-elimina lo que el nuevo factor toca) y por qué la
incrementalidad convirtió el punto caliente del padre en ruido de fondo
(25.7 → 46.7 fps con el hilo de mapeo). ¿Qué pieza de NUESTRO tracker
tendría que cambiar para aprovecharlo?

## 5. Las perillas del vocabulario (medio)

Barre `n_palabras` de la BolsaDePalabras: 64, 256, 1024, midiendo (a) si el
candidato del bucle sigue siendo el mismo que la fuerza bruta (recall) y
(b) el costo de consulta según crece la base de keyframes.

**Objetivo**: la curva recall-vs-costo. Con 64 palabras todo se parece a
todo (histogramas densos, coseno poco discriminante); con 1024 y pocos
descriptores por imagen, el histograma se vuelve ralo y frágil. Y la
pregunta de diseño: ¿por qué DBoW2 (ORB-SLAM) puede permitirse 1M de
palabras? (pista: vocabulario jerárquico k^L y pre-entrenado).
