# Aprende Visual SLAM — de cero a un sistema funcional

Un curso práctico por niveles: **de "una imagen es una matriz de números"
hasta un SLAM visual completo** con bundle adjustment y cierre de bucle.

Cada nivel es una carpeta **independiente y autoejecutable**, y termina con un
examen (`verificacion.py`) que compara tu resultado contra un número esperado.
Si el examen pasa, dominas el nivel.

> **Estado**: EN CONSTRUCCIÓN. La ruta **00 → 13 está completa y verificada**
> (14 niveles). Al terminar el nivel 13 tienes un SLAM funcional: máquina de
> estados, mapa por keyframes, bundle adjustment y cierre de bucle verificado,
> corriendo sobre un corredor de ida y vuelta. Los niveles 14-20 (datos reales,
> RGB-D, estéreo, features aprendidas, tiempo real, mapa denso 3DGS, ROS 2)
> están en camino.

## Empezar

```bash
cd nivel_00_entorno_y_primeros_pixeles

pip install -r requirements.txt      # deps del nivel
python descarga_datos.py             # solo si el nivel usa dataset
python 00_hola_pixeles.py            # el script: leelo de arriba a abajo
python verificacion.py               # el examen del nivel
```

`EJERCICIOS.md` trae retos opcionales, cada uno con su número objetivo.

**Prerrequisitos**: Python básico (funciones, listas, `pip`) y álgebra lineal
introductoria (multiplicar matrices). **Nada de visión por computadora**: eso
lo construye el curso.

**Hardware**: CPU sola hasta el nivel 16. Webcam opcional (niveles 01 y 04).
GPU NVIDIA sólo en 17 y 19; Docker sólo en 19 y 20.

## Las reglas del curso (la constitución)

1. **Independencia total**: cero imports entre niveles, cero paquete común.
   Copia una carpeta suelta a otra máquina y corre.
2. **Duplicar > importar**: el mismo loader aparece en varios niveles,
   recortado a lo que ya sabes en ese punto. Es deliberado — prima el contexto
   inmediato sobre el DRY.
3. **La matemática vive en el código**: bloques `─── La matemática ───` junto
   a la línea que la usa. El README del nivel da la intuición.
4. **Cada nivel cierra con un número.** Muchos son mediciones reales del
   sistema del que salió este curso (ver más abajo).
5. **Scripts planos**, legibles de arriba a abajo. Clases sólo cuando la clase
   ES la lección.
6. **Windows-proof**: prints ASCII, rutas con pathlib.

## Los niveles

### Bloque A — La cámara (formación de imagen)

| # | Nivel | Hito verificable |
|---|---|---|
| 00 | [entorno y primeros píxeles](nivel_00_entorno_y_primeros_pixeles/) | gris a mano == OpenCV (±1 nivel) |
| 01 | [el sensor de imagen](nivel_01_el_sensor_de_imagen/) | ruido shot ∝ √señal (pendiente 0.49) |
| 02 | [cámara pinhole](nivel_02_camara_pinhole/) | cubo alámbrico en numpy puro |
| 03 | [poses y transformaciones](nivel_03_poses_y_transformaciones/) | la vuelta entera compone la identidad |
| 04 | [distorsión y calibración](nivel_04_distorsion_y_calibracion/) | reproyección 0.24 px; la recta se endereza 24× |

### Bloque B — Dos vistas

| # | Nivel | Hito verificable |
|---|---|---|
| 05 | [características](nivel_05_caracteristicas/) | Harris propio == cv2 (100% del top-50) |
| 06 | [matching](nivel_06_matching/) | el ratio test: 30% → 8% de matches incoherentes |
| 07 | [geometría epipolar](nivel_07_geometria_epipolar/) | error de rotación 0.32° contra la verdad |
| 08 | [odometría visual](nivel_08_odometria_visual/) | **ATE 16 cm** — y la deriva, vista crecer |

### Bloque C — De la odometría al SLAM

| # | Nivel | Hito verificable |
|---|---|---|
| 09 | [triangulación](nivel_09_triangulacion/) | tu primer mapa 3D, exportado a `.ply` |
| 10 | [PnP y mapa persistente](nivel_10_pnp_y_mapa_persistente/) | **18.6 → 8.7 cm** cambiando sólo la arquitectura |
| 11 | [bundle adjustment](nivel_11_bundle_adjustment/) | Schur + el gauge de 7 gdl, medido |
| 12 | [grafo de poses y cierre de bucle](nivel_12_grafo_de_poses_y_cierre_de_bucle/) | Strasdat: SE(3) **empeora**, Sim(3) arregla |
| 13 | [SLAM completo](nivel_13_slam_completo/) | **5.8 cm**; sin bundle adjustment, 148 cm (colapso) |

### Bloque D — Electivas (en camino)

14 datos reales (TUM) · 15 RGB-D y escala métrica · 16 estéreo (EuRoC) ·
17 features aprendidas · 18 ingeniería de tiempo real · 19 mapa denso 3DGS ·
20 ROS 2

## De dónde sale este curso

Es la descomposición pedagógica de
[Visual-slam](https://github.com/arielvazquez/Visual-slam) (`vslam-edu` en
PyPI): un sistema de SLAM visual con arquitectura de producción, del mismo
autor. **Aquel repo es el destino; éste es el camino.**

Muchos de los números que vas a reproducir son mediciones reales de aquel
sistema: su ATE, sus colapsos, sus lecciones aprendidas por las malas. Cuando
un nivel dice *"el repo padre midió esto"*, es literal — y suele venir con la
historia de lo que costó descubrirlo.

## Para quien mantiene el curso

```bash
python verifica_todos.py                      # el examen de TODOS los niveles
python verifica_todos.py --root <ruta_TUM>    # reutiliza un dataset ya descargado
```

Sobre los datasets: los niveles 00, 05 y 06 descargan cada uno su propia copia
de TUM fr1_xyz (~450 MB). Es el precio de la independencia estricta. Si ya lo
tienes, todos los scripts aceptan `--root <carpeta>` y no descargan nada.

## Licencia

MIT.
