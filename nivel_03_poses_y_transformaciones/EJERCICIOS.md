# Ejercicios — Nivel 03

## 1. El bug de los subíndices, a propósito (fácil)

En `render_desde`, usa `T_w_c` directamente en vez de `invert_se3(T_w_c)`
y mira el resultado.

**Objetivo**: describe QUÉ ves (¿dónde acabó el cubo?) y explica con la
cadena de subíndices por qué ese error rara vez da un crash — da una imagen
plausible y MAL, que es peor. Es el bug número 1 de todo principiante en
robótica; la notación existe para cazarlo antes de correr.

## 2. Ruido que se acumula (medio — el trailer del nivel 08)

Toma los 60 relativos exactos de la órbita (`T_ck_ck+1`) y perturba cada
uno con un pequeño giro aleatorio (0.5°) y 5 mm de traslación. Compón la
cadena y compara la pose final con la real.

**Objetivo**: repite 100 veces y grafica el error final. Después hazlo con
el DOBLE de pasos (120 poses en el mismo círculo): ¿el error final crece o
decrece? Acabas de descubrir por qué la odometría deriva y por qué menos
composiciones (keyframes, nivel 10) ayudan.

## 3. Interpolar poses (medio)

Implementa la interpolación entre dos poses: lineal para t, y para R vía
slerp de cuaterniones (impleméntala: `q(u) = q1·sin((1−u)θ)/sin θ +
q2·sin(uθ)/sin θ`, cuidando el signo q2 → −q2 si el producto punto es
negativo).

**Objetivo**: genera una órbita "suavizada" con 10 poses clave
interpoladas a 60 y renderiza el video. Verifica que en u=0 y u=1
recuperas las poses originales exactas y que ‖q(u)‖=1 siempre.

## 4. Ejes de otros mundos (difícil, para robóticos)

ROS usa ejes X-delante/Z-arriba (REP-103); OpenCV usa Z-delante/Y-abajo.
Escribe la conversión de una trayectoria completa de un convenio al otro.
Pista del repo padre (su lección de v0.8): NO basta rotar un lado — es una
CONJUGACIÓN, `T_ros = R̃ · T_cv · R̃⁻¹`, o el mundo queda inconsistente.

**Objetivo**: convierte la órbita, verifica que las distancias entre poses
consecutivas se conservan (son invariantes de conjugación) y que la
trayectoria "en planta" ahora vive en el plano XY en vez de XZ.
