import sys
if sys.prefix == '/home/bebaran/micromamba/envs/ros_env':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/bebaran/Área de trabalho/programação/Módulo 6 - Eng. Comp/ROS 2 Teste/install/turtle_draw'
