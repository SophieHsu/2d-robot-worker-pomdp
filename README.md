# 2d-robot-worker-pomdp

The purpose of this project is to be familiarized with generating a PomdpX file and solving a pomdp problem with SARSOP.
 
## Problem description
We will be explore a problem named as the 2D Robot Worker problem. The problem models a robot moving towards its goal in a 2-dimensional simulated factory. Inside the factory, there is also a worker that operates to move towards its assigned goal. The robot does not know which one out of the two possible worker's goal is the worker assigned to. However, it is able to infer through observing the worker's trajectory.

\insert picture of the problem


## Execution instructions
1) install SARSOP solver
2) execute 2d_robot_worker.py until it asks for an policy file name
\screenshot of input
3) retrieve the 2d_robot_worker.pomdpx file from the file location you execute the 2d_robot_worker.py
4) use SARSOP to solve the pomdpx file and obtain a policy file
5) input the policy file name to the terminal

Finally, you will see outputs of the simulated results.
\screenshot of results

## POMDP model
Consider a map of size XXXXXXX \times XXXXXX as shown in the figure above. The two goals labed in XXXX are the goals that will be assigned to the worker. The XXXX is the robot's goal. We formulate the POMDP model as fellow:

States = {R_x, R_y, W_x, W_y, W_a, W_g}
Actions = {Stop, Up, Right, Down, Left}
Observations = {W_x, W_y, W_vel, W_ori}

The states include the robot's position (R_x and R_y), the workers position (W_x and W_y) and the worker's movement (W_a) including stop and going up, right, down or left one grid cell. The robot's action will also be stop and going up, right, down or left one grid cell. Both the robot and worker's position and motion is fully observable. The only unobservable state will be the worker's assigned goal (W_g).

As for the transition function, it describes the locmotion of both the robot as it can move to neighboring cells that are horizontally or vertically adjacent to the one it is in. Additionally, we include the locmotion of the worker with an extra constraint based on the worker's assigned goal. For example, the two trajectories shows the exact grids the worker will take when assigned reach one of the two goals. Therefore, the extra constriant is to express the influence of W_g (an information included in the state space).

\picture of two different trajectories

## Worker trajectories

We create two different trajectories respectivly for the two worker goals by solving an MDP model of the worker aiming to reach its goal destination.

\code snipet of the worker's mdp

It is advised that the worker should be able to reach the goal with multiple different trajectories. Therefore, the trajectory generating code can be altered to be more realistic.






