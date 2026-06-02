import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/root/gpufree-data/code/tal-vla/robot_ws/install/robot_navigation'
