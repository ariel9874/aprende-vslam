#!/usr/bin/env python3
"""Nodo DATASET: publica una secuencia TUM como si fuera una cámara en vivo.

Corre DENTRO del contenedor (ver Dockerfile). Publica sensor_msgs/Image
(mono8) en /camara/imagen al ritmo de los timestamps reales del dataset.
Sin cv_bridge: una imagen mono8 es alto*ancho bytes — se arma a mano (menos
dependencias, y se ve exactamente qué es un mensaje).

    ros2 run  (via launch: ver lanzar_slam.launch.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SecuenciaTUM                       # el loader del nivel 14


class NodoDataset(Node):
    def __init__(self) -> None:
        super().__init__("dataset")
        self.declare_parameter("root", "/data/rgbd_dataset_freiburg2_xyz")
        self.declare_parameter("hz", 15.0)
        root = self.get_parameter("root").value
        # QoS por defecto (reliable, cola 10). La leccion del padre (su 44):
        # reliable protege el TRANSPORTE, no al suscriptor que llega tarde —
        # por eso el orden de bringup es consumidores -> productor.
        self.pub = self.create_publisher(Image, "/camara/imagen", 10)
        self._iter = iter(SecuenciaTUM(root))
        hz = float(self.get_parameter("hz").value)
        self.timer = self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(f"publicando {root} a {hz:.0f} Hz")

    def _tick(self) -> None:
        try:
            ts, gris = next(self._iter)
        except StopIteration:
            self.get_logger().info("secuencia terminada")
            self.timer.cancel()
            return
        msg = Image()
        msg.header.stamp.sec = int(ts)
        msg.header.stamp.nanosec = int((ts % 1.0) * 1e9)
        msg.header.frame_id = "camara"
        msg.height, msg.width = gris.shape
        msg.encoding = "mono8"
        msg.step = gris.shape[1]
        msg.data = gris.tobytes()
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    nodo = NodoDataset()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
