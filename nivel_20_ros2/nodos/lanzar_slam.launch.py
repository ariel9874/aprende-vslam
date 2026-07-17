"""Launch: el pipeline completo dentro del contenedor.

Orden de bringup: CONSUMIDORES antes que PRODUCTOR (la lección 44 del padre,
medida: QoS reliable protege el transporte, no al suscriptor que llega tarde
— si el dataset arranca primero, los primeros frames se pierden y el SLAM
inicializa más tarde de lo que crees). Por eso el dataset sale con retraso.

    ros2 launch /nodos/lanzar_slam.launch.py
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(cmd=["python3", "/nodos/nodo_slam.py"],
                       output="screen"),
        TimerAction(period=3.0, actions=[
            ExecuteProcess(cmd=["python3", "/nodos/nodo_dataset.py"],
                           output="screen"),
        ]),
    ])
