from daniel.jointCtlComp import *
from daniel.taskCtlComp import *

states_dict = dict()

# Controller in the joint space. The robot has to reach a fixed position.
# jointCtlComp(['P', 'PD', 'PID'], True)

# Same controller, but this time the robot has to follow a fixed trajectory.
# jointCtlComp(['P', 'PD', 'PID'], False)

# Controller in the task space.
taskCtlComp(['JacNullSpace'],resting_pos=np.mat([0, -pi]).T)

input('Press Enter to close')