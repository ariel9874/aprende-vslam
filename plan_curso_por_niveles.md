# Plan del curso por niveles — descomposición pedagógica del repo

> **Qué es este documento**: el diseño del repositorio HIJO que descompone
> este repo en un curso práctico paso a paso, de cero conocimientos a un
> Visual SLAM funcional. El padre NO se toca: el curso se construye copiando
> y simplificando archivos de aquí (duplicación deliberada, ver §2).
>
> **ESTADO (julio 2026): EJECUTADO.** El repo hijo existe y vive en
> `C:\Users\ariel\Documents\GitHub\aprende-vslam` (repo git propio, MIT, un
> commit por nivel). **Niveles 00-14 construidos y verificados**: de "una
> imagen es una matriz" a un SLAM completo con BA y cierre de bucle (corredor:
> ATE 5.8 cm; sin BA colapsa a 148 cm) corriendo sobre DATOS REALES (nivel 14,
> fr2_xyz entera: 0 perdidos, 245 KFs, 22 bucles, **1.4 cm** final-KF tras el
> GBA — dentro de la banda 0.4-1.5 del padre; fr1_desk documentado como límite
> del envelope: 562/613 perdidos, vs 560 del padre). Nota de construcción del
> 14: la primera versión heredó el mapa local por RECENCIA del 13 y reprodujo
> la lección 14 del padre (96k puntos duplicados, 35 cm que ningún BA bajaba);
> la covisibilidad + "observar antes que crear" entraron al código enseñado de
> ese nivel, con la historia medida en su README. **Nivel 15 también
> construido y verificado**: fr1_desk — la secuencia que derrotó al 14 — en
> METROS: 2.3 cm RÍGIDO, escala 1.012 (criterio del padre: 2.8 cm/1.005,
> cumplido); quedan 203 perdidos de blur (el padre con reloc: 0), documentados
> como el hueco que cubren la reloc (ejercicio) y el nivel 17. Restan los
> niveles 16-20 (estéreo, aprendidas, tiempo real, 3DGS, ROS 2), ya diseñados
> en §4.
>
> El curso NO importa nada de este repo (verificado: 0 imports de `vslam` en
> sus ~7 200 líneas). Se congela contra el tag v1.0.0: si el padre evoluciona,
> el curso no se rompe. Este documento se queda aquí como registro de diseño
> (documenta CÓMO se descompuso el sistema); la ejecución vive en el hijo.

---

## 1. Decisión de arquitectura: UN repo hijo, no varios

**Decidido y ejecutado: un único repositorio nuevo, `aprende-vslam`**, con una
carpeta por nivel (`nivel_00_...` a `nivel_20_...`). **No** un repo por nivel.

Por qué (mejores prácticas medibles de los tres referentes del género):

| Referente | Estructura | Qué copiamos |
|---|---|---|
| [slambook2](https://github.com/gaoxiang12/slambook2) (Gao Xiang, el curso de SLAM canónico) | Un repo, `ch2/`..`ch13/`, cada capítulo compila SOLO | Carpeta por capítulo, cero imports cruzados |
| [nand2tetris](https://www.nand2tetris.org/course) | Un repo/curso, 12 proyectos autocontenidos; único prerrequisito: intro a CS | Proyecto = unidad de progreso; prerrequisitos mínimos explícitos |
| [Ray Tracing in One Weekend](https://raytracing.github.io/) | Cada mini-capítulo añade UNA pieza y termina con una imagen: si tu salida coincide, vas bien | **El hito verificable por nivel** (nuestro `verificacion.py`) |

Razones operativas contra el multi-repo: 21 repos = 21 READMEs/issues/CI que
mantener, descubribilidad fragmentada (las estrellas no se acumulan), y la
narrativa del curso (nivel N motiva al N+1) se pierde. La independencia que
piden las reglas pedagógicas se logra POR CARPETA, no por repo: cada nivel
tiene sus propios `requirements.txt`, datos y scripts, y se puede copiar
suelto a otra máquina y correr.

**Relación con el padre**: el curso se congela contra el tag `v1.0.0` del
padre. No hay submódulos, no hay sincronización archivo a archivo: si el padre
evoluciona, el curso NO se rompe (duplicación asumida también entre padre e
hijo). El README raíz del curso enlaza al padre como "el sistema completo del
que salió este curso". Licencia MIT, igual que el padre.

---

## 2. Constitución del curso (las reglas de cada nivel)

Heredan la cultura del padre (docs/05 §2), adaptadas al alumno:

1. **Independencia total**: cada carpeta corre con
   `pip install -r requirements.txt` + `python descarga_datos.py` (si usa
   dataset) + `python <script>.py`. Cero imports entre niveles, cero paquete
   compartido, cero `sys.path` hacia fuera de la carpeta.
2. **Duplicar > importar**: el loader de TUM se copia en cada nivel que lo
   necesita, recortado a lo que el alumno ya sabe en ese punto. No hay DRY
   entre niveles: el contexto inmediato es la prioridad (regla del propio
   padre: examples/01 ya duplica deliberadamente — docs/05 §4.1).
3. **La matemática vive en el código**: bloques `─── La matemática ───` en
   español, identificadores en inglés — el estilo exacto del padre. El README
   del nivel da la intuición; la derivación va junto a la línea que la usa.
4. **Cada nivel cierra con un número**: `verificacion.py` corre el nivel y
   compara contra un valor esperado con tolerancia (el "cada versión cierra
   con números" del padre, vuelto examen del alumno). Donde existe, el número
   esperado ES el número histórico medido del padre (§5).
5. **Scripts planos, legibles de arriba a abajo**: sin paquetes, sin
   `__init__.py`; clases solo cuando la clase ES la lección (Frame, Map,
   Tracker). Un script principal por nivel, auxiliares solo si son cortos.
6. **Windows-proof**: prints solo ASCII (cp1252 — la lección del padre),
   rutas con pathlib, deps mínimas (numpy+opencv+matplotlib hasta que el
   nivel exija más).

**Anatomía de cada carpeta**:

```
nivel_XX_nombre/
├── README.md          # teoría del nivel: objetivo, conceptos, cómo correr
├── requirements.txt   # deps PROPIAS (aunque repitan las del vecino)
├── descarga_datos.py  # idempotente; solo si el nivel usa dataset
├── XX_<tema>.py       # el script principal, se lee de arriba a abajo
├── verificacion.py    # el examen: corre y compara contra números esperados
└── EJERCICIOS.md      # retos opcionales, cada uno con pista y número objetivo
```

**Prerrequisitos del alumno** (explícitos en el README raíz): Python básico
(funciones, listas, `pip`) y álgebra lineal introductoria (multiplicar
matrices; los autovalores se re-explican donde aparecen). Nada de visión,
nada de optimización: eso lo construye el curso.

**Hardware por nivel**: CPU sola para 00–16 y 18 (GTSAM va por conda);
webcam OPCIONAL en 01/04 (hay imágenes provistas como alternativa); GPU
NVIDIA solo en 17 y 19; Docker solo en 19 (gsplat full-res, por el mangling
nvcc↔MSVC — lección 40) y 20 (ROS 2).

---

## 3. Índice completo de niveles

Ruta troncal 00–15 (≈ un semestre a nivel/semana). Electivas 16–20:
independientes entre sí, cualquiera se cursa tras el 14/15.

| # | Carpeta | El alumno termina pudiendo… | Verificación |
|---|---|---|---|
| 00 | `nivel_00_entorno_y_primeros_pixeles` | Tratar una imagen como matriz numpy | shape/dtype/estadísticas esperadas |
| 01 | `nivel_01_el_sensor_de_imagen` | Explicar qué pasa entre el fotón y el ndarray | RMS del demosaicado propio |
| 02 | `nivel_02_camara_pinhole` | Proyectar 3D→2D con K a mano | píxeles exactos de puntos conocidos |
| 03 | `nivel_03_poses_y_transformaciones` | Componer/invertir T_w_c sin miedo | ida∘vuelta = identidad (1e-10) |
| 04 | `nivel_04_distorsion_y_calibracion` | Calibrar SU cámara y corregir la distorsión | reproyección < 0.5 px |
| 05 | `nivel_05_caracteristicas` | Detectar esquinas (Harris propio) y describirlas | tabla de invarianza medida |
| 06 | `nivel_06_matching` | Emparejar descriptores y filtrar con ratio test | conteos esperados de matches |
| 07 | `nivel_07_geometria_epipolar` | Recuperar R,t entre dos vistas (E + RANSAC) | error de rotación < 1° vs GT |
| 08 | `nivel_08_odometria_visual` | Encadenar poses y MEDIR la deriva (ATE) | **≈ 13 cm** (el v0.1 del padre) |
| 09 | `nivel_09_triangulacion` | Convertir matches en puntos 3D (PLY) | reproyección media < 1 px |
| 10 | `nivel_10_pnp_y_mapa_persistente` | Trackear 3D→2D contra un mapa que persiste | **13 → ~7 cm** (el salto v0.2) |
| 11 | `nivel_11_bundle_adjustment` | Optimizar poses y puntos a la vez (GN/LM/Schur) | error ↓ por iteración; gauge 2 anclas |
| 12 | `nivel_12_grafo_de_poses_y_cierre_de_bucle` | Cerrar un bucle y redistribuir la corrección | **1.09 m → 0.05 m** (examples/03) |
| 13 | `nivel_13_slam_completo` | Ensamblar un tracker INIT/TRACK/LOST con bucles | corredor **~2 cm** ON / colapso OFF |
| 14 | `nivel_14_datos_reales_tum` | Sobrevivir a datos reales (matching guiado) | fr2_xyz a pocos cm; fr1 falla y sabe por qué |
| 15 | `nivel_15_rgbd_escala_metrica` | SLAM en METROS con sensor de profundidad | fr1_desk **~3 cm métrico, escala ≈ 1.0** |
| 16 | `nivel_16_estereo_euroc` (electiva) | Rectificar un rig real y sacar profundidad | V1_01 **~7 cm métrico** |
| 17 | `nivel_17_features_aprendidas` (electiva) | Medir cuándo el deep gana a lo clásico | fr1_desk **560 → ~140 perdidos** |
| 18 | `nivel_18_ingenieria_de_tiempo_real` (electiva) | Perfilar → sustituir → test de equivalencia | su propia tabla de fps (4.3→46.7 de guía) |
| 19 | `nivel_19_mapa_denso_3dgs` (electiva) | Escribir un rasterizador 3DGS diferenciable | sobreajuste **> 30 dB**; real ~21 dB |
| 20 | `nivel_20_ros2` (electiva) | Sacar el SLAM al robot (nodos, TF, RViz) | demo RViz viva + lifecycle pausa/reanuda |

Bloques: **0** arranque (00) · **A** la cámara (01–04) · **B** dos vistas
(05–08) · **C** de VO a SLAM (09–15) · **D** electivas (16–20).

---

## 4. Detalle por nivel

### nivel_00_entorno_y_primeros_pixeles

- **Objetivo de aprendizaje**: montar el entorno (Python + numpy + opencv +
  matplotlib) y perder el miedo al array: una imagen digital ES una matriz.
- **Archivos involucrados**: ninguno de `vslam/` (deliberado — el alumno aún
  no debe ver el sistema). `descarga_datos.py` se adapta del comando curl de
  docs/05 §3.2 (fr1_xyz, ~450 MB). El formato/estilo de script se toma de
  `examples/01` (docstring-guía + pasos numerados).
- **Conceptos clave**: entorno e instalación de paquetes; `ndarray` H×W×3
  uint8; espacios de color; gris como combinación ponderada
  (0.299R+0.587G+0.114B, y por qué no es el promedio); histograma; indexado
  y slicing como edición de imagen.
- **Hito práctico**: `python 00_hola_pixeles.py` abre un frame de TUM,
  lo convierte a gris A MANO (sin `cv2.cvtColor`), dibuja su histograma y
  guarda un negativo y un recorte. `verificacion.py`: shape (480,640,3),
  dtype uint8, y la conversión manual coincide con cv2 a ±1 nivel de gris.

### nivel_01_el_sensor_de_imagen

- **Objetivo de aprendizaje**: entender qué hay entre el fotón y el ndarray —
  y por qué las imágenes reales traen ruido, blur y saturación.
- **Archivos involucrados**: código NUEVO (el padre no modela el sensor: es
  un hueco que el curso llena). Un simulador corto: escena → exposición →
  ruido shot/read → mosaico de Bayer → cuantización. El README conecta con la
  lección 41 del padre (motion blur y rolling shutter como techo fotométrico
  de fr1) y la 28 (por qué el handheld duele).
- **Conceptos clave**: fotosito y pozo de carga; exposición y saturación;
  ruido de disparo (Poisson) vs de lectura; mosaico de Bayer y demosaicado;
  rolling vs global shutter; gamma. Todo con el simulador, no con fórmulas
  sueltas.
- **Hito práctico**: demosaicar a mano (bilineal) una imagen Bayer generada
  por el propio script y medir el RMS contra el original; barrer la
  exposición y VER (gráfica) el ruido relativo subir al oscurecer.
  `verificacion.py`: RMS del demosaicado bajo umbral; el ruido medido escala
  como sqrt de la señal. Ejercicio opcional con webcam: capturar oscuro/claro
  y repetir la medición real.

### nivel_02_camara_pinhole

- **Objetivo de aprendizaje**: la cámara como función matemática — proyectar
  puntos 3D a píxeles con K.
- **Archivos involucrados**: `vslam/core/camera.py` → `camara.py` del nivel,
  RECORTADO: solo la clase pinhole con `project`/`backproject` y su bloque
  `─── La matemática ───`; fuera distorsión, `from_file` completo y
  `undistort_points` (llegan en el nivel 04).
- **Conceptos clave**: modelo pinhole; fx, fy, cx, cy y sus unidades
  (píxeles); plano de imagen y centro óptico; coordenadas homogéneas; la
  profundidad se PIERDE al proyectar (la semilla de todo el curso); ejes
  OpenCV (+Z delante, +Y abajo — la convención fija del padre, presentada
  aquí de una vez y para siempre).
- **Hito práctico**: renderizador alámbrico en numpy puro (sin OpenGL): un
  cubo proyectado con la K real de TUM fr1, animado en órbita y guardado como
  GIF/MP4. `verificacion.py`: la proyección de vértices conocidos da los
  píxeles esperados (exactos).

### nivel_03_poses_y_transformaciones

- **Objetivo de aprendizaje**: dominar T_w_c — componer, invertir y no
  perderse con los marcos de referencia ANTES de que aparezca la estimación.
- **Archivos involucrados**: `invert_se3` de `examples/01_monocular_vo.py`
  (su bloque de matemática ya es perfecto para el curso, se copia tal cual);
  `vslam/core/trajectory.py` → recortado a "escribir/leer formato TUM"
  (cuaterniones incluidos, con la conversión de Shepperd simplificada o
  scipy); la convención de subíndices de docs/02.
- **Conceptos clave**: R ortonormal (RᵀR=I, det=+1); SE(3) como matriz 4×4;
  componer = multiplicar y el ORDEN importa; subíndices que se cancelan
  (`T_w_c2 = T_w_c1 · T_c1_c2` — la notación del padre, que previene el 80%
  de los bugs de frames); inversa cerrada; extrínsecos vs intrínsecos;
  formato TUM. NOTA: Exp/Log y el espacio tangente NO van aquí — llegan en
  el nivel 12 cuando se necesitan (la abstracción se introduce al usarse).
- **Hito práctico**: volar la cámara del nivel 02 en círculo alrededor del
  cubo componiendo poses, exportar la trayectoria en formato TUM y graficarla
  en planta. `verificacion.py`: componer la vuelta entera y volver da la
  identidad a 1e-10; el archivo TUM re-leído reproduce las poses.

### nivel_04_distorsion_y_calibracion

- **Objetivo de aprendizaje**: pasar de la cámara ideal a la real — estimar
  K y la distorsión de una cámara física.
- **Archivos involucrados**: la parte de distorsión de `vslam/core/camera.py`
  (campo `distortion` Brown-Conrady + `undistort_points`, con su matemática);
  `tests/test_camera_distortion.py` como semilla de `verificacion.py`; script
  NUEVO de calibración con tablero (`cv2.calibrateCamera` — el padre no lo
  trae porque pre-rectifica; el curso llena el hueco con el flujo OpenCV
  estándar). Set de imágenes de tablero provisto para quien no tenga webcam.
- **Conceptos clave**: distorsión radial (k1..k3) y tangencial (p1,p2); por
  qué las rectas se curvan en el borde; el tablero como objeto de geometría
  conocida; el error de reproyección como métrica de calibración; corregir la
  imagen (undistort) vs corregir puntos (más barato — lo que hace el padre).
- **Hito práctico**: calibrar la webcam propia (o el set provisto) y ver el
  stream/las imágenes SIN distorsión; comparar la K obtenida con la K
  publicada de TUM fr1 entendiendo cada número. `verificacion.py` (set
  provisto): error de reproyección < 0.5 px y parámetros en rango esperado.

### nivel_05_caracteristicas

- **Objetivo de aprendizaje**: decidir QUÉ mirar de una imagen — esquinas y
  descriptores, y qué significa "invariante".
- **Archivos involucrados**: `vslam/frontend/features.py` → recortado al
  registro con 3 extractores (gftt-orb, orb, sift) manteniendo la interfaz
  `detect_and_compute`; `scripts/benchmark_frontends.py` como inspiración del
  comparador del nivel; docs/03 del padre como lectura recomendada (enlace,
  no copia).
- **Conceptos clave**: el problema de apertura (por qué esquinas y no bordes);
  Harris/Shi-Tomasi por autovalores del gradiente — implementarlo A MANO en
  numpy es EL ejercicio del nivel; FAST; descriptores binarios (BRIEF/ORB,
  distancia de Hamming) vs flotantes (SIFT, L2); invarianza a rotación y
  escala (pirámides).
- **Hito práctico**: Harris propio en numpy sobre un frame de TUM, comparado
  lado a lado con cv2 (GFTT/ORB/SIFT); tabla MEDIDA: % de keypoints
  re-detectados al rotar 30° y al escalar 0.5×, por detector.
  `verificacion.py`: los máximos de Harris propio coinciden con cv2 en ≥80%
  de las 50 esquinas más fuertes.

### nivel_06_matching

- **Objetivo de aprendizaje**: emparejar descriptores entre dos imágenes y
  filtrar mentiras — el ratio test como primera línea de defensa.
- **Archivos involucrados**: `vslam/frontend/matching.py` → recortado a
  fuerza bruta + ratio test + cross-check (fuera FLANN y la firma para
  aprendidos), con su bloque de matemática del ratio de Lowe.
- **Conceptos clave**: distancia de Hamming (XOR + popcount) y L2; vecino más
  cercano y segundo vecino; el ratio de Lowe (por qué 0.75: la ambigüedad se
  mide contra el segundo mejor); cross-check; los matches falsos son
  INEVITABLES sin geometría — motivación directa del nivel 07.
- **Hito práctico**: visualización antes/después del ratio test entre dos
  frames de TUM separados (líneas de match dibujadas); curva medida: número
  de matches y % de supervivencia vs umbral de ratio (0.6–0.9).
  `verificacion.py`: conteos dentro del rango esperado en el par provisto.

### nivel_07_geometria_epipolar

- **Objetivo de aprendizaje**: recuperar el movimiento de la cámara desde dos
  vistas — la matriz esencial, RANSAC y la descomposición en R, t.
- **Archivos involucrados**: el "paso 4" inline de
  `examples/01_monocular_vo.py` (el padre lo dejó deliberadamente didáctico
  ahí — es la semilla exacta); `tests/test_pose_recovery.py` como base de
  `verificacion.py`; las lecciones 1 (`distanceThresh` de recoverPose) y 17
  (MAGSAC en cuasi-planos) COPIADAS como comentario junto a la línea que las
  encarna, con sus números.
- **Conceptos clave**: la restricción epipolar x'ᵀEx = 0 y su dibujo (el
  plano epipolar); líneas epipolares; RANSAC como votación contra outliers;
  descomposición de E (4 soluciones) y quiralidad (los puntos deben quedar
  DELANTE); la ambigüedad de escala: t sale UNITARIO — el precio del
  monocular, que persigue al alumno hasta el nivel 15.
- **Hito práctico**: entre dos frames de TUM con GT: dibujar las líneas
  epipolares sobre ambas imágenes y recuperar R con error < 1° contra ground
  truth (t comparado solo en dirección). Reproducir la trampa de la lección 1
  (recoverPose sin `distanceThresh` colapsa inliers con profundidad/baseline
  > 50) como experimento del `verificacion.py`.

### nivel_08_odometria_visual

- **Objetivo de aprendizaje**: el primer SISTEMA completo — encadenar poses
  frame a frame, exportar la trayectoria y MEDIR la deriva contra GT.
- **Archivos involucrados**: `examples/01_monocular_vo.py` CASI ÍNTEGRO (ya
  es autocontenido por diseño del padre: la semilla natural del nivel);
  `vslam/evaluation.py` (Umeyama + ATE, con su matemática) copiado como
  `evaluacion.py`; `scripts/make_synthetic_sequence.py` (modo forward)
  copiado para generar la secuencia sintética; loader mínimo de carpeta de
  imágenes (recorte de `vslam/io/dataset.py`).
- **Conceptos clave**: composición de trayectoria (nivel 03 en acción);
  deriva sin límite — no hay mapa ni optimización que la contenga; la escala
  monocular como gauge (aquí ||t||=1 y por qué eso "funciona" solo a
  velocidad constante); alineación Umeyama (por qué hay que alinear antes de
  comparar) y ATE; ground truth y formato TUM.
- **Hito práctico**: correr la VO sobre la secuencia sintética y sobre TUM
  fr1_xyz; gráfico trayectoria-vs-GT en planta y error(t). Número de
  referencia: **ATE ≈ 13 cm en la sintética forward — el número real de v0.1
  del padre**. El alumno ve la deriva CRECER con la longitud del recorrido:
  la motivación medible de TODO lo que sigue.

### nivel_09_triangulacion

- **Objetivo de aprendizaje**: convertir matches en PUNTOS 3D — el primer
  mapa, y el error de reproyección como su control de calidad.
- **Archivos involucrados**: `vslam/core/geometry.py` →
  `triangulate_two_views` con sus filtros y su matemática DLT (copiado casi
  entero); la parte de triangulación de `tests/test_triangulation_pnp.py`
  como base de verificación; el export PLY de `vslam/mapping/sparse.py`
  extraído a una función suelta de ~20 líneas.
- **Conceptos clave**: triangulación DLT (el punto como mínimos cuadrados de
  dos rayos); paralaje — sin baseline no hay 3D (conexión con la ambigüedad
  del nivel 07); error de reproyección; quiralidad otra vez; filtros de
  calidad (ángulo, profundidad, reproyección) y por qué un mapa sucio es
  peor que un mapa chico (adelanto de la lección 8 del padre).
- **Hito práctico**: reconstrucción 3D de dos vistas de TUM exportada a PLY
  y abierta en MeshLab (o el visor matplotlib incluido). `verificacion.py`:
  error medio de reproyección < 1 px y fracción de puntos filtrados en rango.

### nivel_10_pnp_y_mapa_persistente

- **Objetivo de aprendizaje**: dejar de re-estimar desde cero — trackear
  3D→2D (PnP) contra un mapa que PERSISTE y crece con keyframes.
- **Archivos involucrados**: `examples/02_pnp_tracking.py` (simplificado);
  `solve_pnp` de `vslam/core/geometry.py`; versión MÍNIMA de
  `vslam/mapping/sparse.py` (posiciones + descriptores + observaciones;
  fuera covisibilidad, culling, re-anclaje — llegan en 13);
  `vslam/core/frame.py` (el contrato de datos, ya minimalista en el padre).
- **Conceptos clave**: PnP — por qué 3D-2D es más estable que 2D-2D (el mapa
  acumula evidencia; la escala se hereda en vez de re-inventarse);
  inicialización del mapa en 2 vistas + gauge mediana=1 (la convención del
  padre); qué es un keyframe y cuándo insertarlo (versión simple: cada N /
  por inliers); triangular puntos frescos al insertar.
- **Hito práctico**: **reproducir el salto v0.1→v0.2 del padre con sus
  números**: la misma secuencia sintética pasa de ~13 cm (nivel 08) a ~7 cm
  con tracking PnP — la primera mejora ARQUITECTÓNICA que el alumno mide él
  mismo. `verificacion.py`: ATE < 9 cm y mejora ≥ 30% sobre el nivel 08.

### nivel_11_bundle_adjustment

- **Objetivo de aprendizaje**: el corazón matemático de SLAM — mínimos
  cuadrados no lineales sobre poses Y puntos a la vez.
- **Archivos involucrados**: `vslam/backend/bundle_adjustment.py` — la
  referencia NumPy con jacobianos analíticos y complemento de Schur es EL
  material didáctico del padre; se copia casi entera. El bloque de teoría MAP
  de `vslam/backend/factor_graph.py` va al README. Base de verificación:
  `tests/test_bundle_adjustment.py`. Las lecciones 4–7 del padre (gauge de 7
  gdl, agujeros de costo, Huber no anula, puntos con 1 obs se deslizan) se
  copian como comentarios donde tocan, con sus números.
- **Conceptos clave**: residuo de reproyección; Gauss-Newton y
  Levenberg-Marquardt; jacobianos por regla de la cadena
  (proyección∘pose∘punto); estructura DISPERSA del problema y el complemento
  de Schur (por qué BA escala); kernel de Huber; el gauge monocular tiene 7
  gdl — fijar 2 cámaras, no 1 (lección 4, reproducible).
- **Hito práctico**: BA sobre una ventana sintética viendo el error de
  reproyección bajar por iteración (gráfica log). Reproducir el experimento
  del gauge: con 1 ancla la escala deriva (error idéntico en poses y puntos —
  la firma de la lección 4); con 2 anclas converge. `verificacion.py` =
  versión compacta del test del padre.

### nivel_12_grafo_de_poses_y_cierre_de_bucle

- **Objetivo de aprendizaje**: matar la deriva acumulada — reconocer un lugar
  ya visitado y REDISTRIBUIR la corrección por toda la trayectoria.
- **Archivos involucrados**: `vslam/core/lie.py` (ahora sí: Exp/Log de SE(3)
  y Sim(3) con su matemática — la abstracción llega cuando se necesita);
  `vslam/backend/pose_graph.py` (GN/LM genérico por grupo);
  `examples/03_pose_graph_loop.py` (el ejemplo simulado, casi tal cual);
  `tests/test_pose_graph.py` (incluye el experimento Strasdat). Lecciones
  10–11 copiadas junto al código.
- **Conceptos clave**: espacio tangente y por qué se optimiza ahí (suma no
  preserva SO(3)); grafo de poses: nodos, aristas relativas, información; el
  cierre de bucle como restricción adicional; la deriva monocular incluye
  ESCALA → SE(3) no puede repartirla y EMPEORA, Sim(3) sí (lección 10, el
  resultado de Strasdat reproducido en test); anclar el grafo (gauge otra
  vez, ahora en el grafo).
- **Hito práctico**: reproducir `examples/03`: trayectoria simulada con
  deriva **1.09 m → 0.05 m** tras optimizar el grafo con el bucle. Y el
  experimento SE(3)-vs-Sim(3) del test como demostración medida de la
  lección 10. `verificacion.py`: ambos números con tolerancia.

### nivel_13_slam_completo

- **Objetivo de aprendizaje**: ensamblar TODO lo anterior en un tracker con
  estados — el primer SLAM de verdad, entero, del alumno.
- **Archivos involucrados**: versión SIMPLIFICADA de
  `vslam/frontend/tracker.py` — el PnPTracker de ~850 líneas recortado a
  ~300: mapa local por RECENCIA (la covisibilidad queda como ejercicio
  estrella, con la lección 14 como guía), bucle por fuerza bruta contra la
  base de KFs (sin BoW), síncrono (sin hilo), sin matching guiado ni reloc
  (reloc = segundo ejercicio, con la lección 18). `examples/04_loop_closure.py`
  adaptado como driver; `make_synthetic_sequence.py --motion loop` (el
  corredor de carteles — lección 15); `sparse.py` ya con observaciones
  completas.
- **Conceptos clave**: la máquina de estados INIT/TRACK/LOST; validación de
  la inicialización (3ª vista — lección 2); política de keyframes: min/max
  gap y el piso de salud (lección 8: "nunca crear mapa desde pose incierta");
  mapa local; BA local de ventana (el nivel 11 enchufado); detección de
  bucle: matching + verificación geométrica + Sim(3) (el nivel 12
  enchufado); qué se congela al corregir (lección 11).
- **Hito práctico**: el corredor sintético con bucle: **ATE ~2 cm con bucle
  ON** y visiblemente peor OFF; el modo sin BA (~200 cm, colapso de escala)
  como demostración de que cada pieza carga peso. Gráfico x(t) con el bucle
  marcado (lección 16: las trayectorias de ida-y-vuelta se grafican como
  serie temporal). `verificacion.py`: ATE ON < 4 cm y ratio OFF/ON > 2.

### nivel_14_datos_reales_tum

- **Objetivo de aprendizaje**: el salto sim→real — y las dos palancas medidas
  que lo cruzan: el matching guiado y la métrica correcta.
- **Archivos involucrados**: `vslam/io/dataset.py` → `TUMRGBDLoader` +
  `associate_by_timestamp` + `tum_camera` (recortado, sin EuRoC);
  `examples/05_tum_rgbd.py` simplificado como driver; el tracker del nivel 13
  + `_guided_match` copiado del padre (la referencia Python, NO la C++);
  `global_bundle_adjustment` (GBA offline). El README del nivel resume las
  lecciones 21–28 con sus números (la historia real de v0.45).
- **Conceptos clave**: por qué lo real rompe lo sintético (distorsión → se
  pre-rectifica; blur; exposición; timestamps y asociación); la inanición de
  keyframes (lección 21) y su cura REAL: el matching guiado por reproyección
  (lección 24 — predecir pose, proyectar el mapa, buscar en ventana de 15
  px); evaluar la trayectoria FINAL de KFs, no la online (lección 25 — el
  artefacto de medición más caro del padre); GBA offline y convergencia
  (lecciones 26–27); el envelope de operación: qué secuencias caen fuera y
  POR QUÉ (lección 28).
- **Hito práctico**: fr2_xyz con ATE final-KF de POCOS cm (referencia del
  padre: 0.4–1.5 según config) y fr1_desk FALLANDO con dignidad: el alumno
  documenta el límite con números y frames perdidos, como hace el padre.
  `verificacion.py`: fr2_xyz < 5 cm final-KF; el guiado ON mejora
  measurablemente sobre OFF (ablación incluida).

### nivel_15_rgbd_escala_metrica

- **Objetivo de aprendizaje**: eliminar la ambigüedad de escala con un sensor
  de profundidad — SLAM en METROS de verdad.
- **Archivos involucrados**: `TUMRGBDLoader` con `with_depth=True` (PNG 16
  bits, factor 1/5000, 0 = sin dato); la init RGB-D del tracker
  (`_initialize_rgbd`: retro-proyección métrica desde el frame 0); el residuo
  de profundidad del BA (estéreo virtual u_R = u − bf/z, residuo [u,v,u_R] —
  la parte métrica de `bundle_adjustment.py`); el bucle métrico en SE(3);
  `examples/05 --depth` como driver; `evaluation.py` con Umeyama RÍGIDO.
  Lecciones 35–36 en el README (las dos moralejas grandes).
- **Conceptos clave**: cámaras RGB-D (luz estructurada/ToF) y sus agujeros;
  retro-proyección; la escala como MEDICIÓN vs como gauge — y la
  consecuencia: el bucle métrico va en SE(3), no Sim(3) (lección 35: Sim(3)
  re-escalaba el mapa métrico y COMPONÍA el error — la moraleja conceptual
  más bonita del padre); el residuo de profundidad como ancla métrica por
  observación (lección 36: es lo que cruza el episodio biestable); Umeyama
  rígido + escala de similitud como CHEQUEO (≈1.0 = metros de verdad); el
  bug del mapa MIXTO (init monocular accidental) como caso de estudio.
- **Hito práctico**: **fr1_desk ~3 cm MÉTRICO con escala Umeyama ≈ 1.005 —
  el criterio de v0.6 del padre, reproducido por el alumno** (referencia:
  2.8 cm). `verificacion.py`: ATE rígido < 5 cm y |escala − 1| < 0.05; la
  ablación sin residuo de profundidad empeora (12.8 cm en el padre).

### nivel_16_estereo_euroc (electiva)

- **Objetivo de aprendizaje**: la cámara derecha virtual se vuelve REAL — rig
  estéreo calibrado, rectificación y profundidad por disparidad.
- **Archivos involucrados**: `EuRoCStereoRig` y `EuRoCStereoLoader` de
  `vslam/io/dataset.py` (rectificación cv2.stereoRectify, bf desde P2, SGBM);
  `examples/06_euroc.py --stereo` como driver; `tests/test_stereo.py` como
  base de verificación sin dataset. Lección 37 en el README. Nota de datos
  del padre: host oficial caído → mirror HF `pepijn223/euroc-mirror`
  (V1_01_easy.zip ~1.1 GB), documentado en `descarga_datos.py`.
- **Conceptos clave**: rig calibrado y pose relativa cam0←cam1;
  rectificación epipolar (las epipolares se vuelven FILAS → búsqueda 1D);
  disparidad (SGBM) y z = bf/d; el ruido de profundidad crece con z² y el
  peso del residuo decae con z² — se cancelan (la simetría bonita de la
  lección 37); MISMO pipeline RGB-D: solo cambia de dónde sale z.
- **Hito práctico**: V1_01_easy (dron 6-DoF real): **~7 cm métrico, escala
  1.002** (referencia del padre: 6.9). `verificacion.py` sin dataset:
  geometría del rig (bf = fx·b; rectificar lo rectificado ≈ identidad) y
  profundidad por disparidad de un plano sintético — los tests del padre.

### nivel_17_features_aprendidas (electiva)

- **Objetivo de aprendizaje**: cuándo el deep learning gana a lo clásico — y
  cómo MEDIRLO en vez de creerlo.
- **Archivos involucrados**: `vslam/frontend/learned.py` (SuperPoint +
  LightGlue vía el paquete lightglue); el driver del nivel 14 con
  `--detector superpoint --matcher lightglue`. Lección 29 en el README
  (incluida la parte incómoda: el episodio 200-340 que NI el deep cruza).
  Requiere GPU (o paciencia en CPU — documentar ambos).
- **Conceptos clave**: SuperPoint (detección+descripción self-supervised,
  descriptores 256-D float robustos a blur); LightGlue (matching por atención
  espacial 2D-2D) y su restricción de integración (necesita keypoints de
  ambos lados → no sirve para 3D-2D contra el mapa: la asimetría que el
  padre resolvió con `_desc_matcher`); coste computacional real (~370
  ms/frame en GPU); la evaluación honesta: dónde rescata y dónde no.
- **Hito práctico**: fr1_desk (handheld, motion blur): **560 frames perdidos
  con ORB → ~140 con SuperPoint+LightGlue** — el rescate medido de la lección
  29. Y el análisis del residual: el episodio es estructural, no de umbrales
  (el puente conceptual hacia el nivel 15, donde el residuo de profundidad SÍ
  lo cruza).

### nivel_18_ingenieria_de_tiempo_real (electiva)

- **Objetivo de aprendizaje**: el MÉTODO de optimización de sistemas —
  perfilar, sustituir SOLO el punto caliente, verificar equivalencia. No
  "hacerlo rápido": saber DÓNDE.
- **Archivos involucrados**: `vslam/backend/gtsam_ba.py` y `gtsam_isam2.py`
  (+ `tests/test_gtsam_ba.py` / `test_isam2_ba.py` — los tests de
  equivalencia SON la lección); `cpp/src/guided_match.cpp` + `CMakeLists.txt`
  (pybind11, con la nota de toolchain Windows del padre);
  `vslam/frontend/place_recognition.py` (BoW: k-medias en Hamming por voto de
  mayoría, índice invertido, tf·idf); el `async_mapping` del tracker como
  lectura guiada (no se re-implementa). Lecciones 30–34: la escalera completa
  4.3→9.5→18.7→25.7→46.7 fps.
- **Conceptos clave**: perfilado primero — la intuición SE REFUTA (docs/04
  apostaba por cv2; el perfil dijo BA 57% + guiado 37%, cv2 8%); la gemela
  con test de equivalencia exacto (GTSAM↔NumPy, C++↔Python — hasta la
  semántica de desempate de np.argmin); iSAM2 e incrementalidad (intuición
  del árbol de Bayes); pybind11 y el GIL (qué suelta y qué no); BoW.
- **Hito práctico**: el alumno entrega SU tabla de perfilado de fr2_desk y
  reproduce al menos DOS peldaños de la escalera (p.ej. BA numpy→GTSAM ≈ 2×
  fps; BoW ON/OFF sobre el coste del keyframe), con paridad de ATE verificada
  por el test de equivalencia. La escalera del padre (4.3→46.7 fps) es la
  guía de qué esperar.

### nivel_19_mapa_denso_3dgs (electiva)

- **Objetivo de aprendizaje**: del mapa de puntos al mapa foto-realista — un
  rasterizador 3D Gaussian Splatting diferenciable desde cero.
- **Archivos involucrados**: `vslam/mapping/gaussian_render.py` — la
  referencia PyTorch pura y legible ES el material (proyección → covarianza
  EWA → α-blending por transmitancia); `vslam/mapping/gaussian.py` (el mapper:
  siembra desde nube dispersa, renderiza-y-compara, update_poses rígido);
  `examples/07_gaussian_mapping.py`; `tests/test_gaussian_render.py` (el
  gradiente por diferencias finitas como verificación). docs/06 del padre
  como lectura. Las gemelas tiled/gsplat NO se re-implementan: se mencionan
  con la lección 40 (Docker para gsplat; el bug del medio píxel como caso de
  estudio de convenciones).
- **Conceptos clave**: la gaussiana 3D como primitiva (media, covarianza =
  RSSᵀRᵀ, opacidad, color); proyección EWA (Σ' = JWΣWᵀJᵀ); α-blending
  front-to-back por transmitancia; optimizar por renderiza-y-compara
  (autograd); PSNR; la convención del centro de píxel (i+0.5 — el bug de 25→60
  dB del padre, contado como lección); el techo fotométrico de los datos
  reales (lección 41: la capacidad NO era el cuello).
- **Hito práctico**: sobreajustar UNA vista sintética a **PSNR > 30 dB** con
  el rasterizador propio (el test del padre como examen) y el gradiente
  verificado por diferencias finitas. Opcional con GPU/Docker: fr1_desk hacia
  ~21 dB full-res (paridad SOTA — el criterio recalibrado del padre, lección
  41).

### nivel_20_ros2 (electiva)

- **Objetivo de aprendizaje**: sacar el SLAM del script al robot — nodos,
  tópicos, TF y RViz, sin contaminar el núcleo.
- **Archivos involucrados**: `ros2/vslam_msgs` (Keyframe, PoseGraphEdge,
  TrackingState) y `ros2/vslam_ros` (dataset/frontend/backend/mapper — los 4
  nodos rclpy finos) + launch files; `docker/` (el contenedor es EL entorno
  del nivel: no hay ROS nativo en Windows); `conversions.py` con la
  conjugación de ejes como pieza central. Lecciones 43–44 en el README
  (incluido el orden de bringup consumidores→productor, medido).
- **Conceptos clave**: nodos/tópicos/QoS (reliable protege el transporte, no
  al suscriptor tardío); ejes óptico vs REP-103 y la conversión por
  CONJUGACIÓN (rotar un solo lado deja el mundo inconsistente — la
  trayectoria "de lado" en RViz); REP-105: map→odom→base_link y quién publica
  qué (T_map_odom = T_map_kf·T_odom_kf⁻¹); lifecycle nodes; la regla 4 del
  padre: la cáscara no importa nada del núcleo y viceversa.
- **Hito práctico**: demo RViz EN VIVO (trayectoria + Path + PointCloud2 +
  árbol TF completo) sobre TUM corriendo en el contenedor (WSLg en Windows);
  pausar/reanudar el frontend por lifecycle y verlo dejar de publicar.
  `verificacion.py` = adaptación del smoke_pipeline.py del padre (conteos de
  mensajes esperados, metric=True).

---

## 5. Mapa padre→curso (cobertura y omisiones deliberadas)

El curso re-recorre el historial medido del padre — cada hito de verificación
es un número que el padre ya midió (docs/05 §3.1):

| Nivel | Versión del padre que re-vive | Número heredado |
|---|---|---|
| 08 | v0.1 | ATE ~13 cm sintético |
| 10 | v0.2 | ~7 cm (PnP) |
| 11 | v0.35 | BA converge; gauge 2 anclas |
| 12 | v0.3/v0.4 | bucle 1.09 m → 0.05 m |
| 13 | v0.4 | corredor ~2 cm |
| 14 | v0.45 | fr2_xyz < 5 cm final-KF |
| 15 | v0.6 | fr1_desk 2.8 cm métrico, escala 1.005 |
| 16 | v0.6 hito 3 | V1_01 6.9 cm |
| 17 | v0.45 (lección 29) | fr1_desk 560→140 perdidos |
| 18 | v0.5 | 4.3→46.7 fps |
| 19 | v0.7 | >30 dB sintético / 21.0 dB real |
| 20 | v0.8 | smoke RViz completo |

**Qué queda deliberadamente FUERA del curso** (y se enlaza como "lectura en
el repo padre" para el alumno que quiera seguir):

| Pieza del padre | Por qué no entra |
|---|---|
| `config.py` (config declarativa) | Ingeniería de producto, no fundamento; cada nivel fija sus constantes con su porqué |
| `dense_thread.py`, épocas de mapa, test_concurrency | Concurrencia de producción (lecciones 42, 46); se MENCIONA en 18/19, no se re-implementa |
| `gaussian_render_tiled.py` / `gaussian_render_gsplat.py` | Gemelas de rendimiento; el nivel 19 enseña la referencia legible |
| Relocalización completa + compuerta (v0.4b) | Ejercicio avanzado del nivel 13 (con lecciones 13/18 como guía), no nivel propio |
| API freeze, PyPI, CI | Meta-ingeniería del release; se cuenta en el README raíz del curso |
| KITTI, MH_*/V2_*, webcam ROS | Los mismos pendientes del padre; el curso no promete lo no medido |

**Sugerencia de ejecución** (cuando Ariel lo construya): empezar por
`nivel_00` y `nivel_08` como pilotos — el 00 valida la fricción de entorno
con alumnos reales y el 08 es el primer nivel que copia código del padre en
serio (examples/01 + evaluation.py); si el patrón de carpeta funciona ahí,
escala al resto. El README raíz del curso lleva: el índice (§3), el grafo de
prerrequisitos (troncal lineal 00→15; electivas colgando de 14/15), los
requisitos de hardware por nivel y el enlace al padre.
