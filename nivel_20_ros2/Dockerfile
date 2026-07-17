# Nivel 20: el contenedor ES el entorno (no hay ROS 2 nativo en Windows).
# Construye:  docker build -t aprende-vslam-ros2 .
# Corre:      docker compose up   (ver docker-compose.yml)
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip ros-jazzy-rviz2 ros-jazzy-sensor-msgs-py \
    && rm -rf /var/lib/apt/lists/*

# --break-system-packages: la imagen es desechable; el venv seria ceremonia.
# OpenCV fijado a 4.10: la imagen ROS trae numpy 1.26 empaquetado por DEBIAN
# (pip no puede desinstalarlo: "RECORD file not found") y OpenCV 5 arrastra
# numpy 2. La regla de todo contenedor sobre una base con paquetes de
# sistema: NO pelear con el numpy del sistema — elegir dependencias que lo
# acepten. (Nos paso construyendo este nivel; el error exacto queda aqui
# para el siguiente que lo vea.)
RUN pip3 install --no-cache-dir --break-system-packages \
        "opencv-python-headless==4.10.0.84"

COPY nodos/ /nodos/
COPY conversiones.py /nodos/conversiones.py

# El dataset se monta como volumen en /data (docker-compose.yml).
CMD ["bash", "-lc", "ros2 launch /nodos/lanzar_slam.launch.py"]
