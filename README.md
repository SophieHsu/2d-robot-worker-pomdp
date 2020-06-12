# 2D World: Robot-Worker Coordination

The purpose of this project is to be familiarized with generating a PomdpX file and solving a pomdp problem with SARSOP.
 
## Problem description
We will be exploring a problem named as the 2D Robot Worker problem. The problem models a robot moving towards its goal in a 2-dimensional simulated factory. Inside the factory, there is also a worker that operates to move towards its assigned goal. The robot does not know which one out of the two possible worker's goal is the worker assigned to. However, it is able to infer through observing the worker's trajectory.

![Image of problem_map](/images/problem_map.png)

## Execution instructions
1) install SARSOP solver
2) execute 2d_robot_worker.py until it asks for a policy file name
![Image of input_policy](/images/input_policy.png)
3) retrieve the 2d_robot_worker.pomdpx file from the file location you execute the 2d_robot_worker.py
4) use SARSOP to solve the pomdpx file and obtain a policy file
5) input the policy file name to the terminal

Finally, you will see the outputs of the simulated results.
![Image of results](/images/results.png)

## POMDP model
Consider a map of size 3 x 7, as shown in the figure above. The two goals labeled in blue are the goals that will be assigned to the worker. The red is the robot's goal. We formulate the POMDP model as fellow:

States = {R_x, R_y, W_x, W_y, W_v, W_g, W_a}

Actions = {Stop, Up, Right, Down, Left}

Observations = {W_x, W_y, W_vel, W_ori}

The states include the robot's position (R_x and R_y), the worker's position (W_x and W_y), and the worker's movement (W_v), including stop and going up, right, down or left one grid cell. The robot's action will also be stop and going up, right, down or left one grid cell. Both the robot and worker's position and motion is fully observable. The unobservable states will be the worker's assigned goal (W_g) and its adaptiveness (W_a) to the robot.

As for the transition function, it describes the locomotion of both the robot as it can move to neighbor cells that are horizontally or vertically adjacent to the one it is in. Additionally, we include the locomotion of the worker with an extra constraint based on the worker's assigned goal. For example, the two trajectories show the exact grids the worker will take when assigned reach one of the two goals. Therefore, the extra constraint is to express the influence of W_g (information included in the state space).

![Image of worker_policies](/images/worker_policies.png)

## Worker trajectories

We create two different trajectories, respectively, for the two worker goals by solving an MDP model of the worker aiming to reach its goal destination.

\code snippet of the worker's mdp

### Worker adaptiveness design

The goal is to illustrate the influence of the robot's presents can influence the worker's choice of next step. This is a more realistic demonstration that includes the coupled influence of action choices between the robots and humans. The adaptiveness design in this project can be a projection of human preferences that are usually unobservable (can only be interpreted through inference of observable traits).

We will assume that adaptiveness can be divided into 5 levels. The higher the level is, the more possibility that worker adapts its trajector according to the robot's status. Additionally, the worker only adapts when seen the robot as a threat to safety, in other words, only when the robot is within a certain distance and possibilly will collide with the worker. The details of the condition can be seen in the function adaptiveWorkerModel().

Note: It is advised that the worker should be able to reach the goal with multiple different trajectories. Therefore, the trajectory generating code can be altered to be more realistic.



