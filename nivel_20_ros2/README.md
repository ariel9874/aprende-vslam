# Nivel 20 — ROS 2 · electiva

**Objetivo**: sacar el SLAM del script al robot — nodos, tópicos, TF y RViz
— **sin contaminar el núcleo**. El tracker de este nivel es el del 14, sin
una sola línea nueva: todo lo de ROS vive en la cáscara.

## La regla de oro (la regla 4 del padre)

La cáscara no importa nada del núcleo salvo su API pública; el núcleo no
sabe que ROS existe. TODO cambio de convención pasa por **`conversiones.py`
en la frontera** — nunca dentro del tracker. Esa disciplina es lo que
permitió al padre tener el MISMO núcleo corriendo en scripts, benchmarks y
nodos.

## La pieza central: la conjugación de ejes

El núcleo habla en ejes ÓPTICOS (nivel 02: +Z delante, +Y abajo); ROS habla
REP-103 (+X delante, +Z arriba). La conversión correcta CONJUGA:

```
T_map_base = R̃_bo · T_w_c · R̃_bo⁻¹
```

porque conjugar preserva la estructura de grupo (los deltas se convierten
igual que las poses — el examen lo mide: 8.9·10⁻¹⁶). El bug clásico — rotar
SOLO un lado — el examen lo comete a propósito y lo mide: la posición parece
bien y la actitud miente **120° constantes** — el robot montado "de lado" en
RViz, invisible en un plot 2D de posiciones. Y la segunda trampa de la
frontera: geometry_msgs usa cuaterniones **xyzw** (el nivel 03 usaba wxyz;
mezclarlos rota 180°).

## REP-105: quién publica qué

TF exige que cada frame tenga UN padre. La odometría publica
`odom → base_link` (continua, deriva, nunca salta); el SLAM conoce
`map → base_link` pero NO puede publicarlo directo — publica la CORRECCIÓN:

```
T_map_odom = T_map_base · T_odom_base⁻¹
```

Cuando un bucle corrige, **salta `map → odom`** y `odom → base_link` sigue
continuo: los planificadores locales viven de esa continuidad. El nodo de
este nivel publica exactamente eso (y lo anuncia en el log cuando pasa).

## La arquitectura (en `nodos/`)

- `nodo_dataset.py` — publica fr2_xyz como cámara en vivo (`mono8` a mano,
  sin cv_bridge: un mensaje Image ES alto×ancho bytes).
- `nodo_slam.py` — la cáscara: suscribe imágenes, corre el SLAM del nivel
  14, publica el árbol TF de REP-105, el `Path` de keyframes y el mapa como
  `PointCloud2`. Servicio `/slam/pausa` (el lifecycle real es el ejercicio 2).
- `lanzar_slam.launch.py` — bringup **consumidores → productor** (lección 44
  del padre: QoS reliable protege el transporte, no al suscriptor tardío).

El padre divide en 4 nodos (dataset/frontend/backend/mapper) con mensajes
propios; aquí son 2 para que el patrón se vea entero — el split es el
ejercicio 5.

## La demo viva (Docker; en Windows, RViz sale por WSLg)

No hay ROS 2 nativo en Windows: **el contenedor es el entorno**.

```bash
python descarga_datos.py         # fr2_xyz en ./data (o copia la del nivel 14)
docker compose up --build        # dataset + slam
# en otra terminal:
docker compose exec slam bash -lc "rviz2"
#   Fixed frame: map · añade TF, Path=/slam/trayectoria, PointCloud2=/slam/mapa
# pausa/reanuda en vivo:
docker compose exec slam bash -lc \
  "ros2 service call /slam/pausa std_srvs/srv/SetBool '{data: true}'"
```

## El examen (sin ROS, en cualquier máquina)

`python verificacion.py` — 9 checks de la matemática de la frontera: R_BO
propia y a los ejes correctos, conjugación que preserva el grupo (8.9e-16),
el bug del lado único medido (120.0°), cuaterniones xyzw por las 4 ramas
(θ≈π incluida) y las dos identidades de REP-105. Con el daemon de Docker
corriendo, `--docker` añade el build de la imagen.

## Qué debes poder explicar al terminar

- Por qué la conversión de ejes es una CONJUGACIÓN y qué se rompe (y qué
  no) si rotas un solo lado.
- Por qué el SLAM publica map→odom y no map→base_link — y qué transformada
  salta cuando se cierra un bucle.
- Qué protege QoS reliable y qué no (el orden de bringup).
- Dónde vive cada conversión de convención en este nivel — y por qué esa
  disciplina de frontera es lo que mantiene el núcleo portable.
