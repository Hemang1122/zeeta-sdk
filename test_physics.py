import pybullet as p
import pybullet_data
import time
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
print("? If you see the 3D window, PyBullet is working perfectly!")
time.sleep(5)
p.disconnect()
