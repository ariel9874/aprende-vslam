#!/usr/bin/env python3
"""Nodo SLAM: suscribe imágenes, corre el tracker del nivel 14 y publica el
árbol TF de REP-105 + la trayectoria + el mapa para RViz.

La regla de oro (regla 4 del padre): este nodo es una CÁSCARA. El núcleo
(slam.py) no sabe que ROS existe; toda conversión de convención pasa por
conversiones.py en la frontera. Publica:

  - TF map -> odom      : la CORRECCIÓN del SLAM (REP-105; teoría en
                          conversiones.py). Salta cuando un bucle corrige.
  - TF odom -> base_link: la odometría integrada de pasos relativos —
                          continua, deriva, y NUNCA salta (los planificadores
                          locales viven de esa continuidad).
  - /slam/trayectoria   : nav_msgs/Path con los KEYFRAMES (frame map).
  - /slam/mapa          : sensor_msgs/PointCloud2 (frame map).
  - servicio /slam/pausa (std_srvs/SetBool): pausa/reanuda el procesamiento
    (el lifecycle de verdad — configure/activate — es el ejercicio 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_srvs.srv import SetBool
from tf2_ros import TransformBroadcaster

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conversiones import optico_a_rep103, rot_a_quat_xyzw, t_map_odom
from dataset import camara_tum
from slam import SLAM


def _tf(stamp, padre: str, hijo: str, T: np.ndarray) -> TransformStamped:
    m = TransformStamped()
    m.header.stamp = stamp
    m.header.frame_id = padre
    m.child_frame_id = hijo
    m.transform.translation.x, m.transform.translation.y, \
        m.transform.translation.z = map(float, T[:3, 3])
    (m.transform.rotation.x, m.transform.rotation.y,
     m.transform.rotation.z, m.transform.rotation.w) = \
        rot_a_quat_xyzw(T[:3, :3])
    return m


class NodoSLAM(Node):
    def __init__(self) -> None:
        super().__init__("slam")
        self.declare_parameter("secuencia", "rgbd_dataset_freiburg2_xyz")
        K, _ = camara_tum(self.get_parameter("secuencia").value)
        self.slam = SLAM(K)
        self.T_odom = np.eye(4)          # odometria: pasos relativos, deriva
        self.pausado = False
        self.sub = self.create_subscription(Image, "/camara/imagen",
                                            self._imagen, 10)
        self.pub_path = self.create_publisher(PathMsg, "/slam/trayectoria", 1)
        self.pub_mapa = self.create_publisher(PointCloud2, "/slam/mapa", 1)
        self.tf = TransformBroadcaster(self)
        self.create_service(SetBool, "/slam/pausa", self._pausa)
        self.get_logger().info("esperando imagenes en /camara/imagen ...")

    def _pausa(self, req, resp):
        self.pausado = req.data
        resp.success = True
        resp.message = "pausado" if req.data else "reanudado"
        self.get_logger().info(resp.message)
        return resp

    def _imagen(self, msg: Image) -> None:
        if self.pausado:
            return
        gris = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
        T, info = self.slam.procesar(gris)

        # Odometria pura: integrar el paso relativo (continua, con deriva).
        self.T_odom = self.T_odom @ self.slam.T_rel

        # FRONTERA: optico -> REP-103, por conjugacion. Nunca en el nucleo.
        T_map_base = optico_a_rep103(T)
        T_odom_base = optico_a_rep103(self.T_odom)
        stamp = msg.header.stamp
        self.tf.sendTransform(_tf(stamp, "odom", "base_link", T_odom_base))
        self.tf.sendTransform(_tf(stamp, "map", "odom",
                                  t_map_odom(T_map_base, T_odom_base)))

        if info["kf"] or info["loop"]:
            self._publicar_mapa(stamp)
        if info["loop"]:
            self.get_logger().info(f"BUCLE {info['loop']} -> map->odom salta;"
                                   " odom->base_link sigue continuo")

    def _publicar_mapa(self, stamp) -> None:
        path = PathMsg()
        path.header.frame_id = "map"
        path.header.stamp = stamp
        frames, pos = self.slam.trayectoria_kfs()
        for T_kf in (self.slam.kf_poses[k] for k in sorted(self.slam.kf_poses)):
            p = PoseStamped()
            p.header.frame_id = "map"
            Tb = optico_a_rep103(T_kf)
            p.pose.position.x, p.pose.position.y, p.pose.position.z = \
                map(float, Tb[:3, 3])
            (p.pose.orientation.x, p.pose.orientation.y,
             p.pose.orientation.z, p.pose.orientation.w) = \
                rot_a_quat_xyzw(Tb[:3, :3])
            path.poses.append(p)
        self.pub_path.publish(path)

        pts = np.array(list(self.slam.mapa.puntos.values()))
        if len(pts):
            from conversiones import R_BO
            pts = (R_BO @ pts.T).T           # posiciones: solo rotar (frame map)
            header = path.header
            self.pub_mapa.publish(create_cloud_xyz32(header,
                                                     pts[::3].tolist()))


def main() -> None:
    rclpy.init()
    nodo = NodoSLAM()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
