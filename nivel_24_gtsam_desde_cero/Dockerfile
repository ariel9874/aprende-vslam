# Nivel 24, acto 5: GTSAM real. El contenedor ES el entorno (patron del
# nivel 20): no hay wheel de gtsam para Windows/Python 3.13 — verificado
# 2026-07: pip solo publica manylinux y mac. Python 3.11 porque la wheel
# oficial de gtsam 4.2 cubre cp38-cp311.
#
# Construye:  docker build -t aprende-vslam-gtsam .
# Corre:      docker compose up --build     (o: docker run --rm aprende-vslam-gtsam)
FROM python:3.11-slim

# numpy<2: la wheel de gtsam 4.2 esta compilada contra el ABI de numpy 1.x;
# con numpy 2 el import truena. La regla del nivel 20, otra vez: no pelear
# con el ABI — fijar dependencias que la wheel acepte. (Si pip no trajera
# gtsam, el fallback documentado es una imagen conda-forge; ver README.)
RUN pip install --no-cache-dir "numpy<2" "gtsam==4.2"

COPY mundo.py eliminacion.py con_gtsam_real.py /nivel/
WORKDIR /nivel

CMD ["python", "con_gtsam_real.py"]
