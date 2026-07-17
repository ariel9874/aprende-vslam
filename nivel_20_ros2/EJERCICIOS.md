# Ejercicios — Nivel 20

Todos requieren el contenedor (`docker compose up --build`) salvo el 4.

## 1. El bug, en vivo (fácil y catártico)

En `nodo_slam.py`, sustituye `optico_a_rep103` por `optico_a_rep103_MAL`
(el bug del examen) y abre RViz.

**Objetivo**: VE lo que el acto 3 del examen midió: la trayectoria avanza
"bien" pero los ejes del robot (TF) van montados de lado — 120° de actitud
falsa. Ahora entiendes por qué este bug sobrevive semanas en proyectos
reales: en un plot 2D de posiciones es invisible.

## 2. Lifecycle de verdad (medio)

El servicio `/slam/pausa` es un apagador casero. Conviértelo en un
**lifecycle node** de ROS 2 (configure → activate → deactivate), donde
deactivate deja de procesar Y de publicar TF.

**Objetivo**: `ros2 lifecycle set /slam deactivate` en vivo, y el árbol TF
congelándose en RViz. El bringup ordenado de sistemas reales se construye
sobre estos estados (el padre lo usa para pausar el frontend sin matar el
proceso).

## 3. QoS: el suscriptor tardío (fácil, medible)

Invierte el orden del launch (dataset primero, SLAM 3 s después) y compara
en cuántos frames inicializa el SLAM en ambos órdenes.

**Objetivo**: la lección 44 del padre, medida por ti: reliable garantiza
que lo TRANSMITIDO llega, no que lo publicado antes de que te suscribas
exista. Después prueba `durability=TRANSIENT_LOCAL` con `depth` grande en
el publisher y explica qué cambia (y qué costaría en memoria con imágenes).

## 4. El brazo del sensor (medio; sin Docker)

El nodo publica `base_link` == cámara. En un robot real la cámara va
montada con un desplazamiento fijo `T_base_cam` (el brazo de palanca del
nivel 16, ahora del lado de TF). Añade el frame `camara` como TF ESTÁTICO
`base_link → camara` y convierte las poses del núcleo en consecuencia.

**Objetivo**: la cadena `map → odom → base_link → camara` completa en
`ros2 run tf2_tools view_frames`, con la matemática: ¿dónde se inserta
`T_base_cam` en la conjugación? (Escríbelo primero en papel; el examen del
nivel 16 te recuerda qué pasa si lo ignoras.)

## 5. El split del padre (difícil)

Divide `nodo_slam.py` en frontend (tracking → publica pose y keyframes) y
backend (BA y bucle → publica correcciones), comunicados por tópicos.

**Objetivo**: el mismo split del padre (4 nodos con `vslam_msgs` propios).
Te obligará a decidir QUÉ viaja en el mensaje de keyframe (¿descriptores?
¿observaciones?) y descubrirás por qué el padre diseñó mensajes propios en
vez de abusar de los estándar. Compara la latencia por frame contra el nodo
monolítico: ¿cuánto cuesta la serialización?
