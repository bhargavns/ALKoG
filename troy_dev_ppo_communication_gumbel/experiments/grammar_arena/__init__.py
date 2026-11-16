"""Frozen compositional communication in the existing kinematic MuJoCo arena."""
import os
if os.name=='nt':
    os.environ['MUJOCO_GL']='glfw'
else:
    os.environ.setdefault('MUJOCO_GL','egl')
