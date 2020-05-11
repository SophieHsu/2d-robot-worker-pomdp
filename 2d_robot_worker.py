#!/usr/bin/env python
__author__ = 'Ya-Chuan (Sophie) Hsu'

import mdptoolbox
import numpy as np
import logging
import math
import xml.etree.cElementTree as ET

WORKERGOALIDX = 2 # 1 or 2
BELIEFGAMMA = 0.5 # the level of effect of the new observations (lower means lesser)

### Outputs and prints
def printValueMap(simMapFile, lowerX, lowerY, upperX, upperY, valueMatrix, matrixName):
	global rewardFile
	tmpString = matrixName + ":\n"
	print(matrixName, ":")
	rewardFile.write(tmpString)
	for j in reversed(range(lowerY, upperY)):
		for i in range(lowerX, upperX):
			print("%.2f\t" % valueMatrix[((j - lowerY) * (upperX - lowerX)) + (i-lowerX)], end="")
			rewardFile.write("%.2f\t" % valueMatrix[((j - lowerY) * (upperX - lowerX)) + (i-lowerX)])

		print("\n")
		rewardFile.write("\n")
def printCurrMap(simMapFile, spaceArr, currRobotPos, currWorkerPos, robotGoalIdx, workerGoalIdx, goalPositions):
	print("\nMap: robot\'s goal[" + str(goalPositions[robotGoalIdx][1]) + ',' + str(goalPositions[robotGoalIdx][0]) + '], worker\'s goal[' + str(goalPositions[workerGoalIdx][1]) + ',' + str(goalPositions[workerGoalIdx][0]) + ']')
	simMapFile.write("\nMap: robot\'s goal[" + str(goalPositions[robotGoalIdx][1]) + ',' + str(goalPositions[robotGoalIdx][0]) + '], worker\'s goal[' + str(goalPositions[workerGoalIdx][1]) + ',' + str(goalPositions[workerGoalIdx][0]) + ']\n')

	# if any agents overlap with a goal
	if set(currRobotPos) == set(goalPositions[robotGoalIdx]):
		robotPrint = 'R(G_r)'
	elif set(currRobotPos) == set(goalPositions[workerGoalIdx]):
		robotPrint = 'R(G_w)'
	else:
		robotPrint = 'R'
	if set(currWorkerPos) == set(goalPositions[robotGoalIdx]):
		workerPrint = 'W(G_r)'
	elif set(currWorkerPos) == set(goalPositions[workerGoalIdx]):
		workerPrint = 'W(G_w)'
	else:
		workerPrint = 'W'


	for j in reversed(range(spaceArr.shape[0])):
		for i in range(spaceArr.shape[1]):

			if j == currRobotPos[0] and i == currRobotPos[1]:
				print(robotPrint, end="\t")
				simMapFile.write(robotPrint+"\t")
			elif j == currWorkerPos[0] and i == currWorkerPos[1]:
				print(workerPrint, end="\t")
				simMapFile.write(workerPrint+"\t")
			elif j == goalPositions[robotGoalIdx][0] and i == goalPositions[robotGoalIdx][1]:
				print("G_r", end="\t")
				simMapFile.write("G_r\t")
			elif j == goalPositions[workerGoalIdx][0] and i == goalPositions[workerGoalIdx][1]:
				print("G_w", end="\t")
				simMapFile.write("G_w\t")
			else:
				print("o", end="\t")
				simMapFile.write("o\t")
		print("\n")
		simMapFile.write("\n")
def printStates(opfile, currRobotPos, currWorkerPos, goalPositions, goalProb):
	opfile.write('Robot[%d,%d] Worker[%d,%d]\nBelief of the worker\'s goal:\n' % (currRobotPos[0], currRobotPos[1], currWorkerPos[0], currWorkerPos[1]))
	print('Robot[' + str(currRobotPos[0]) + ',' + str(currRobotPos[1]) + '] Worker[' + str(currWorkerPos[0]) + ',' + str(currWorkerPos[1]) + ']\nBelief of the worker\'s goal:', end="\n")

	for j in range(1,len(goalPositions)):
		print('G_w'+str(j)+'[' + str(goalPositions[j][1]) + ',' + str(goalPositions[j][0]) + '] =', end=" ")
		opfile.write('G_w'+str(j)+'[' + str(goalPositions[j][1]) + ',' + str(goalPositions[j][0]) + '] = ')

		if goalProb[j-1] < 0.0001:
			goalProb[j-1] = 0.0001
		print('%.3f' % (goalProb[j-1]), end='\t')
		opfile.write('%.3f\t' % (goalProb[j-1]))

	opfile.write('\n')
	print('')
def printNxtAction(opfile, velocityArr, orientationArr, currWorkerVel, agentName):
	print(agentName + ' next move: ', end='')
	opfile.write(agentName + ' next move: ')
	if currWorkerVel > 0.0:
		vel = int((currWorkerVel-1) / orientationArr.shape[0])+1
		ori = int((currWorkerVel-1) % orientationArr.shape[0])
		if ori == 0:
			print('up ', end='')
			opfile.write('up ')
		elif ori == 1:
			print('right ', end='')
			opfile.write('right ')
		elif ori == 2:
			print('down ', end='')
			opfile.write('down ')
		else:
			print('left ', end='')
			opfile.write('left ')

		print(vel, 'grid')
		opfile.write(str(vel) + ' grid\n')

	else:
		print('stop')
		opfile.write('stop\n')

### Calculate transition probabilities in space representation
def spaceTransition2D(spaceArr, velocityArr, orientationArr):
	T = np.zeros((len(orientationArr), len(velocityArr), spaceArr.shape[0], spaceArr.shape[1], spaceArr.shape[0], spaceArr.shape[1]))

	for orientationIdx in range(orientationArr.shape[0]):	# up, right, down, left => 0, 1, 2, 3
		for velocityIdx in range(velocityArr.shape[0]):
			
			start = -1
			end = -1
			if orientationIdx == 0 or orientationIdx == 2:
				for columnIdx in range(spaceArr.shape[1]):
					for rowIdx in range(spaceArr.shape[0]):
						if spaceArr[rowIdx,0] == 0:
							if start >= 0 and end == -1:	# close braket
								end = rowIdx
						else:
							if start == -1:
								start = rowIdx
								end = -1

						if start >= 0 and end == -1 and rowIdx >= (spaceArr.shape[0]-1):
							end = spaceArr.shape[0]

						# if found range of avaliable space, start to calculate probability of transfering in space
						if start >= 0 and end >= 0:
							velocity = velocityArr[velocityIdx]
							if orientationIdx == 2 or orientationIdx == 3:
								velocity *= -1
							tempT = spaceWithVelocity(np.arange(start, end, 1.0), velocity) # "end" index excluded
							for i in range(start, end):
								for j in range(start, end):
									T[orientationIdx][velocityIdx][i][columnIdx][j][columnIdx] = tempT[i][j]
							start = -1
							end = -1

			else:
				for rowIdx in range(spaceArr.shape[0]):
					for columnIdx in range(spaceArr.shape[1]):
						if spaceArr[0,columnIdx] == 0:
							if start >= 0 and end == -1:	# close braket
								end = columnIdx
						else:
							if start == -1:
								start = columnIdx
								end = -1

						if start >= 0 and end == -1 and columnIdx >= (spaceArr.shape[1]-1):
							end = spaceArr.shape[1]

						# if found range of avaliable space, start to calculate probability of transfering in space
						if start >= 0 and end >= 0:
							velocity = velocityArr[velocityIdx]
							if orientationIdx == 2 or orientationIdx == 3:
								velocity *= -1
							tempT = spaceWithVelocity(np.arange(start, end, 1.0), velocity) # "end" index excluded
							for i in range(start, end):
								for j in range(start, end):
									T[orientationIdx][velocityIdx][rowIdx][i][rowIdx][j] = tempT[i][j]
							start = -1
							end = -1
	return T		
def spaceWithVelocity(spaceArr, velocity):
	twoDArrayLength = spaceArr.shape[0]
	if velocity == 0:
		twoDSpaceT = np.identity(twoDArrayLength)
	else:
		twoDSpaceT = np.zeros((twoDArrayLength, twoDArrayLength))

		# space array pre-processing for negative velocity case
		flip = False
		if velocity < 0:
			np.flip(spaceArr, 0)
			np.multiply(spaceArr, -1)
			velocity *= -1
			flip = True

		for i in range(twoDArrayLength):	# loop through row in 2d transition matrix
			probStateRange = spaceArr[i] + velocity
			lowerBoundArr = np.argwhere(spaceArr <= probStateRange)
			lowerBoundIdx = lowerBoundArr[-1]

			# Use lowerBoaundIdx as the index for calculating the probability
			if lowerBoundIdx == i:
				twoDSpaceT[i][i] = 1.0
			else:
				distance = spaceArr[lowerBoundIdx] - spaceArr[i]
				nextSpaceProb = velocity / distance[0]
				currentSpaceProb = 1.0 - nextSpaceProb
				twoDSpaceT[i][lowerBoundIdx] = nextSpaceProb
				twoDSpaceT[i][i] = currentSpaceProb

		if flip == True:
			twoDSpaceT = np.flip(np.flip(twoDSpaceT, axis=1), axis=0)

	return twoDSpaceT

### Assign robot rewards
def assignRewardsInPosition(spaceArr, goalPosition):
	rewardsMatrixInSpace = np.zeros(spaceArr.shape)

	# for row in range(spaceArr.shape[0]):
	# 	for column in range(spaceArr.shape[1]):
	# 		if spaceArr[row, column] == 1 and (abs(goalPosition[0] - row) + abs(goalPosition[1] - column)) > 0:
	# 			rewardsMatrixInSpace[row, column] = 10 / (abs(goalPosition[0] - row) + abs(goalPosition[1] - column))

	if spaceArr[goalPosition[0], goalPosition[1]] == 0:
		print("Error: goal position is assigned to an unavaliable space in the factory.")
	else:
		rewardsMatrixInSpace[goalPosition[0], goalPosition[1]] = 100

	return rewardsMatrixInSpace
def rewardEfficient():
	return -2

### Compress position representation to state representation
def compressRPositionToState(spaceArr, actionSize, stateSize, rewardsMatrixInSpace):
	rewardsMatrix = np.zeros((actionSize, stateSize, stateSize))

	for action in range(actionSize):
		velocity = 0
		orientation = 0

		if action > 0:
			velocity = 1
			orientation = action - 1

		for currentState in range(stateSize):
			tmp = currentState
			currentSpaceX = tmp % spaceArr.shape[1]
			tmp /= spaceArr.shape[1]
			currentSpaceY = int(tmp) % spaceArr.shape[0]

			rewardsMatrix[action, currentState] += rewardsMatrixInSpace[currentSpaceY, currentSpaceX] + rewardEfficient()

	return rewardsMatrix
def compressTPositionToState(spaceArr, actionSize, stateSize, positionMatrix):
	T = np.zeros((actionSize, stateSize, stateSize))

	for action in range(actionSize):
		velocity = 0
		orientation = 0

		if action > 0:
			velocity = 1
			orientation = action - 1

		for currentState in range(stateSize):
			tmp = currentState
			currentSpaceX = tmp % spaceArr.shape[1]
			tmp /= spaceArr.shape[1]
			currentSpaceY = int(tmp) % spaceArr.shape[0]

			for nextState in range(stateSize):
				tmp = nextState
				nextSpaceX = tmp % spaceArr.shape[1]
				tmp /= spaceArr.shape[1]
				nextSpaceY = int(tmp) % spaceArr.shape[0]

				T[action, currentState, nextState] += positionMatrix[orientation][velocity][currentSpaceY][currentSpaceX][nextSpaceY][nextSpaceX]

	return T

### Calculate matrices of T and R
def calculateTMatrix(spaceArr, velocityArr, orientationArr):
	actionSize = 1 + ((velocityArr.shape[0] - 1) * orientationArr.shape[0])
	stateSize = spaceArr.shape[0] * spaceArr.shape[1]

	positionTransitionMatrix = spaceTransition2D(spaceArr, velocityArr, orientationArr)
	T = compressTPositionToState(spaceArr, actionSize, stateSize, positionTransitionMatrix)

	return T
def calculateRMatrix(spaceArr, velocityArr, orientationArr, goalPosition):
	actionSize = 1 + ((velocityArr.shape[0] - 1) * orientationArr.shape[0])
	stateSize = spaceArr.shape[0] * spaceArr.shape[1]

	rewardsMatrixInSpace = assignRewardsInPosition(spaceArr, goalPosition)
	R = compressRPositionToState(spaceArr, actionSize, stateSize, rewardsMatrixInSpace)

	return R
def calculateTAndRMatrix(spaceArr, velocityArr, orientationArr, goalPosition):
	T = calculateTMatrix(spaceArr, velocityArr, orientationArr)
	R = calculateRMatrix(spaceArr, velocityArr, orientationArr, goalPosition)

	return T, R

### Form problem as a pomdpx file and solve with SARSOP
def genPomdpXFile(spaceArr, velocityArr, orientationArr, goalPositions, T, workerPolicies):
	pomdpXFile = open("2d_robot_worker.pomdpx", "w")

	headerString = '<?xml version="1.0" encoding="ISO-8859-1"?>\n' \
		+ '<pomdpx version="1.0" id="2d_robot_worker" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '+ 'xsi:noNamespaceSchemaLocation="pomdpx.xsd">\n' \
		+ '<Description>A sarsop solution for the pomdp inside mdp.</Description>\n' \
		+ '<Discount>0.95</Discount>\n\n\n' \


	variableInitString = genPomdpXVariables(spaceArr, velocityArr, orientationArr, goalPositions)
	initBeliefString = genPomdpXInitBelief(spaceArr, velocityArr, orientationArr)
	stateTranString = genPomdpXStateTran(spaceArr, velocityArr, orientationArr, goalPositions, T, workerPolicies)
	obsFuncString = genPomdpXObsFunc(spaceArr, goalPositions, workerPolicies)
	rewardFuncString = genPomdpXRewardFunc(spaceArr, goalPositions)

	footString = '</pomdpx>'

	pomdpXString = headerString + variableInitString + initBeliefString + stateTranString + obsFuncString + rewardFuncString + footString
	pomdpXFile.write(pomdpXString)

	pomdpXFile.close()
def genPomdpXVariables(spaceArr, velocityArr, orientationArr, goalPositions):
	variableInitString = '<Variable>\n'

	###------ State Variables ------###

	# Define robot position
	variableInitString += '\t<StateVar vnamePrev="robot_pos_0" vnameCurr= "robot_pos_1" fullyObs="true">\n' \
		+ '\t\t<ValueEnum>'
	spaceCount = 0
	for y in range(spaceArr.shape[0]):
		for x in range(spaceArr.shape[1]):
			variableInitString += 'Rpos_' + str(spaceCount) + '_' + str(x) +'_'+ str(y) + ' '
			spaceCount += 1
	variableInitString += '</ValueEnum>\n\t</StateVar>\n' 

	# Define worker position
	variableInitString += '\t<StateVar vnamePrev="worker_pos_0" vnameCurr= "worker_pos_1" fullyObs="true">\n' \
		+ '\t\t<ValueEnum>'
	spaceCount = 0
	for y in range(spaceArr.shape[0]):
		for x in range(spaceArr.shape[1]):
			variableInitString += 'Wpos_' + str(spaceCount) + '_' + str(x) +'_'+ str(y) + ' '
			spaceCount += 1
	variableInitString += '</ValueEnum>\n\t</StateVar>\n' 

	# Define worker actions
	variableInitString += '\t<StateVar vnamePrev="worker_vel_0" vnameCurr= "worker_vel_1" fullyObs="true">\n' \
		+ '\t\t<ValueEnum>'
	for i in range((len(velocityArr)-1) + len(orientationArr)):
		variableInitString += 'Wvel_' + str(i) + ' '
	variableInitString += '</ValueEnum>\n\t</StateVar>\n' 

	# Define worker goal
	variableInitString += '\t<StateVar vnamePrev="worker_goal_0" vnameCurr= "worker_goal_1" fullyObs="false">\n' \
		+ '\t\t<ValueEnum>'
	for goal in range(1,len(goalPositions)):	# starts from index 1, bc 0 is the robot's goal
		variableInitString += 'Wgoal_' + str(goal-1) + ' '
	variableInitString += '</ValueEnum>\n\t</StateVar>\n'


	###------ Observation Variables ------###
	# Define observations
	variableInitString += '\t<ObsVar vname="obs_sensor">\n'	\
		+ '\t\t<ValueEnum>'
	variableInitString += 'obs_g0 obs_g1'
	variableInitString += '</ValueEnum>\n\t</ObsVar>\n'


	###------ Action Variables ------###
	# Define robot actions
	variableInitString += '\t<ActionVar vname="action">\n' \
		+ '\t\t<ValueEnum>'
	for i in range(actionSize):
		variableInitString += 'A_' + str(i) + ' '
	variableInitString += '</ValueEnum>\n\t</ActionVar>\n'


	###------ Reward Variables ------###
	# Define robot actions
	variableInitString += '\t<RewardVar vname="reward" />\n'


	variableInitString += '</Variable>\n\n'

	return variableInitString
def genPomdpXInitBelief(spaceArr, velocityArr, orientationArr):
	spaceCount = spaceArr.shape[0] * spaceArr.shape[1]

	###------ Initial State Belief ------###
	initBeliefString = '<InitialStateBelief>\n'

	# robot_pos_0 starts from bottom left hand corner (first state)
	initBeliefString += '\t<CondProb>\n\t\t<Var>robot_pos_0</Var>\n\t\t<Parent>null</Parent>\n\t\t<Parameter type="TBL">\n\t\t\t<Entry>\n'
	initBeliefString += '\t\t\t\t<Instance> - </Instance>\n'
	initBeliefString += '\t\t\t\t<ProbTable>1.0'
	for y in range(spaceCount-1):
		initBeliefString += ' 0'
	initBeliefString += '</ProbTable>'
	initBeliefString += '\n\t\t\t</Entry>\n\t\t</Parameter>\n\t</CondProb>\n'

	# worker_pos_0 starts from upper right hand corner (last state)
	initBeliefString += '\t<CondProb>\n\t\t<Var>worker_pos_0</Var>\n\t\t<Parent>null</Parent>\n\t\t<Parameter type="TBL">\n\t\t\t<Entry>\n'
	initBeliefString += '\t\t\t\t<Instance> - </Instance>\n'
	initBeliefString += '\t\t\t\t<ProbTable>'
	for y in range(spaceCount-1):
		initBeliefString += '0 '
	initBeliefString += '1.0 </ProbTable>'
	initBeliefString += '\n\t\t\t</Entry>\n\t\t</Parameter>\n\t</CondProb>\n'

	# worker_vel_0 (include stop and 4 orientations)
	initBeliefString += '\t<CondProb>\n\t\t<Var>worker_vel_0</Var>\n\t\t<Parent>null</Parent>\n\t\t<Parameter type="TBL">\n\t\t\t<Entry>\n'
	initBeliefString += '\t\t\t\t<Instance> - </Instance>\n'
	initBeliefString += '\t\t\t\t<ProbTable>1.0'	# initially the worker is at stop
	for y in range(len(orientationArr)):
		initBeliefString += ' 0'
	initBeliefString += '</ProbTable>'
	initBeliefString += '\n\t\t\t</Entry>\n\t\t</Parameter>\n\t</CondProb>\n'

	# worker_goal_0
	initBeliefString += '\t<CondProb>\n\t\t<Var>worker_goal_0</Var>\n\t\t<Parent>null</Parent>\n\t\t<Parameter type="TBL">\n\t\t\t<Entry>\n'
	initBeliefString += '\t\t\t\t<Instance> - </Instance>\n'
	initBeliefString += '\t\t\t\t<ProbTable>uniform</ProbTable>'
	initBeliefString += '\n\t\t\t</Entry>\n\t\t</Parameter>\n\t</CondProb>\n'

	initBeliefString += '</InitialStateBelief>\n\n'

	return initBeliefString
def genPomdpXStateTran(spaceArr, velocityArr, orientationArr, goalPositions, T, workerPolicies):
	spaceCount = spaceArr.shape[0] * spaceArr.shape[1]

	###------ State Transition Functions ------###
	stateTranString = '<StateTransitionFunction>\n'

	stateTranString += '\t<CondProb>\n'
	stateTranString += '\t\t<Var>robot_pos_1</Var>\n'
	stateTranString += '\t\t<Parent>action robot_pos_0</Parent>\n'
	stateTranString += '\t\t<Parameter type="TBL">\n'
	for action in range(actionSize):
		for currentState in range(spaceCount):
			tmp = currentState
			currentSpaceX = tmp % spaceArr.shape[1]
			tmp /= spaceArr.shape[1]
			currentSpaceY = int(tmp) % spaceArr.shape[0]

			for nextState in range(spaceCount):
				tmp = nextState
				nextSpaceX = tmp % spaceArr.shape[1]
				tmp /= spaceArr.shape[1]
				nextSpaceY = int(tmp) % spaceArr.shape[0]

				if(T[action, currentState, nextState] > 0):
					stateTranString += '\t\t\t<Entry>\n'
					stateTranString += '\t\t\t\t<Instance>'
					stateTranString += 'A_' + str(action) + ' '
					stateTranString += 'Rpos_' + str(currentState) + '_' + str(currentSpaceX) + '_' + str(currentSpaceY) + ' '
					stateTranString += 'Rpos_' + str(nextState) + '_' + str(nextSpaceX) + '_' + str(nextSpaceY)
					stateTranString += '</Instance>\n'

					stateTranString += '\t\t\t\t<ProbTable>'
					stateTranString += str(T[action, currentState, nextState])
					stateTranString += '</ProbTable>\n' 
					stateTranString += '\t\t\t</Entry>\n' 
	stateTranString += '\t\t</Parameter>\n\t</CondProb>\n' 

	stateTranString += '\t<CondProb>\n' 
	stateTranString += '\t\t<Var>worker_pos_1</Var>\n' 
	stateTranString += '\t\t<Parent>worker_vel_0 worker_pos_0</Parent>\n' 
	stateTranString += '\t\t<Parameter type="TBL">\n' 
	for vel in range((len(velocityArr)-1) + len(orientationArr)):
		for currentState in range(spaceCount):
			tmp = currentState
			currentSpaceX = tmp % spaceArr.shape[1]
			tmp /= spaceArr.shape[1]
			currentSpaceY = int(tmp) % spaceArr.shape[0]

			for nextState in range(spaceCount):
				tmp = nextState
				nextSpaceX = tmp % spaceArr.shape[1]
				tmp /= spaceArr.shape[1]
				nextSpaceY = int(tmp) % spaceArr.shape[0]

				if(T[vel, currentState, nextState] > 0):
					stateTranString += '\t\t\t<Entry>\n' 
					stateTranString += '\t\t\t\t<Instance>'
					stateTranString += 'Wvel_' + str(vel) + ' '
					stateTranString += 'Wpos_' + str(currentState) + '_' + str(currentSpaceX) + '_' + str(currentSpaceY) + ' '
					stateTranString += 'Wpos_' + str(nextState) + '_' + str(nextSpaceX) + '_' + str(nextSpaceY)
					stateTranString += '</Instance>\n' 

					stateTranString += '\t\t\t\t<ProbTable>'
					stateTranString += str(T[vel, currentState, nextState])
					stateTranString += '</ProbTable>\n' 
					stateTranString += '\t\t\t</Entry>\n' 
	stateTranString += '\t\t</Parameter>\n\t</CondProb>\n' 


	# the workers velocity transition is based on the worker's pre-calculated policy
	stateTranString += '\t<CondProb>\n' 
	stateTranString += '\t\t<Var>worker_vel_1</Var>\n' 
	stateTranString += '\t\t<Parent>worker_pos_0 worker_goal_0</Parent>\n' 
	stateTranString += '\t\t<Parameter type="TBL">\n' 

	for workerGoalIdx in range(1,len(goalPositions)):

		for currentState in range(spaceCount):
			tmp = currentState
			currentSpaceX = tmp % spaceArr.shape[1]
			tmp /= spaceArr.shape[1]
			currentSpaceY = int(tmp) % spaceArr.shape[0]

			policyAction = workerPolicies[workerGoalIdx - 1][(currentSpaceY * spaceArr.shape[1] + currentSpaceX)]
			stateTranString += '\t\t\t<Entry>\n' 
			stateTranString += '\t\t\t\t<Instance>'
			stateTranString += 'Wpos_' + str(currentState) + '_' + str(currentSpaceX) + '_' + str(currentSpaceY) + ' '
			stateTranString += 'Wgoal_' + str(workerGoalIdx-1) + ' '
			stateTranString += 'Wvel_' + str(int(policyAction))
			stateTranString += '</Instance>\n' 

			stateTranString += '\t\t\t\t<ProbTable>1.0</ProbTable>\n' 
			stateTranString += '\t\t\t</Entry>\n' 
	stateTranString += '\t\t</Parameter>\n\t</CondProb>\n' 

	# the workers velocity transition is based on the worker's pre-calculated policy
	stateTranString += '\t<CondProb>\n' 
	stateTranString += '\t\t<Var>worker_goal_1</Var>\n' 
	stateTranString += '\t\t<Parent>worker_goal_0</Parent>\n' 
	stateTranString += '\t\t<Parameter type="TBL">\n' 
	stateTranString += '\t\t\t<Entry>\n' 
	stateTranString += '\t\t\t\t<Instance> - - </Instance>\n' 
	stateTranString += '\t\t\t\t<ProbTable>identity</ProbTable>\n' 
	stateTranString += '\t\t\t</Entry>\n' 
	stateTranString += '\t\t</Parameter>\n\t</CondProb>\n' 

	stateTranString += '</StateTransitionFunction>\n\n' 

	return stateTranString
def genPomdpXObsFunc(spaceArr, goalPositions, workerPolicies):
	spaceCount = spaceArr.shape[0] * spaceArr.shape[1]

	###------ Observation Functions ------###
	obsFuncString = '<ObsFunction>\n' 

	obsFuncString += '\t<CondProb>\n' 
	obsFuncString += '\t\t<Var>obs_sensor</Var>\n' 
	obsFuncString += '\t\t<Parent>worker_pos_1 worker_vel_1 worker_goal_1</Parent>\n' 
	obsFuncString += '\t\t<Parameter type="TBL">\n' 

	obsFuncString += '\t\t\t<Entry>\n' 
	obsFuncString += '\t\t\t\t<Instance> * * * -'
	obsFuncString += '</Instance>\n' 
	obsFuncString += '\t\t\t\t<ProbTable>'
	for workerGoalIdx in range(len(goalPositions)-1):
		obsFuncString += str(1/(len(goalPositions)-1)) + ' '
	obsFuncString += '</ProbTable>\n' 
	obsFuncString += '\t\t\t</Entry>\n' 

	for currentState in range(spaceCount):
		tmp = currentState
		currentSpaceX = tmp % spaceArr.shape[1]
		tmp /= spaceArr.shape[1]
		currentSpaceY = int(tmp) % spaceArr.shape[0]
		
		policyActions = np.zeros([(len(goalPositions)-1), actionSize])
		for workerGoalIdx in range(len(goalPositions)-1):
			policyActionIdx = workerPolicies[workerGoalIdx][(currentSpaceY * spaceArr.shape[1] + currentSpaceX)]
			policyActions[workerGoalIdx, int(policyActionIdx)] += 1

		for actionIdx in range(actionSize):
			sumOfActionFreq = policyActions[:, actionIdx].sum(axis=0)
			if sumOfActionFreq > 0:
				obsFuncString += '\t\t\t<Entry>\n' 
				obsFuncString += '\t\t\t\t<Instance>'
				obsFuncString += 'Wpos_' + str(currentState) + '_' + str(currentSpaceX) + '_' + str(currentSpaceY) + ' '
				obsFuncString += 'Wvel_' + str(actionIdx) + ' * -'
				obsFuncString += '</Instance>\n' 
				obsFuncString += '\t\t\t\t<ProbTable>'
				for workerGoalIdx in range(len(goalPositions)-1):
					obsFuncString += str(policyActions[workerGoalIdx, actionIdx] / sumOfActionFreq) + ' '
				obsFuncString += '</ProbTable>\n' 
				obsFuncString += '\t\t\t</Entry>\n' 

	obsFuncString += '\t\t</Parameter>\n\t</CondProb>\n' 

	obsFuncString += '</ObsFunction>\n\n' 

	return obsFuncString
def genPomdpXRewardFunc(spaceArr, goalPositions):

	###------ Reward Functions ------###
	rewardFuncString = '<RewardFunction>\n' 

	# reward robot's goal state
	rewardFuncString += '\t<Func>\n' 
	rewardFuncString += '\t\t<Var>reward</Var>\n' 
	rewardFuncString += '\t\t<Parent>action robot_pos_0 worker_pos_0</Parent>\n' 
	rewardFuncString += '\t\t<Parameter type="TBL">\n' 
	rewardFuncString += '\t\t\t<Entry>\n' 
	rewardFuncString += '\t\t\t\t<Instance>'
	currX = goalPositions[0,1]
	currY = goalPositions[0,0]
	rewardFuncString += '*' + ' Rpos_' + str(currY * spaceArr.shape[1] + currX) + '_' + str(currX) + '_' + str(currY) + ' *'
	rewardFuncString += '</Instance>\n' 
	rewardFuncString += '\t\t\t\t<ValueTable> 10 </ValueTable>\n' 
	rewardFuncString += '\t\t\t</Entry>\n' 

	for y in range(spaceArr.shape[0]):
		for x in range(spaceArr.shape[1]):
			rewardFuncString += '\t\t\t<Entry>\n' 
			rewardFuncString += '\t\t\t\t<Instance> * '
			rewardFuncString += 'Rpos_' + str(y * spaceArr.shape[1] + x) + '_' + str(x) + '_' + str(y) + ' '
			rewardFuncString += 'Wpos_' + str(y * spaceArr.shape[1] + x) + '_' + str(x) + '_' + str(y) + ' '
			rewardFuncString += '</Instance>\n' 
			rewardFuncString += '\t\t\t\t<ValueTable> -20 </ValueTable>\n' 
			rewardFuncString += '\t\t\t</Entry>\n' 

	rewardFuncString += '\t\t</Parameter>\n\t</Func>\n' 

	rewardFuncString += '</RewardFunction>\n\n' 

	return rewardFuncString
def sarsopPolicy(fileName):
	A = np.zeros([3134, (len(goalPositions)-1)+2])

	print ("Loading XML policy file")
	tree = ET.parse(fileName + ".policy")
	root = tree.getroot()
	numVectors = len(root.getchildren()[0].getchildren())
	print (numVectors)
	print (root.iter('Vector'))
	counter = 0
	for vector in root.iter('Vector'):
		obsValue  = vector.get('obsValue')
		action = vector.get('action')
		values = vector.text.split(' ')

		# vector format: obsValue, action, values
		
		A[counter][0] = float(obsValue)
		A[counter][1] = float(action)
		for vv in range(len(goalPositions)-1):
			A[counter][2+vv] = float(values[vv])
		counter = counter + 1
		
	return A

### Functions for simulation
def executeAction(originalPosition, action, spaceArrMax0, spaceArrMax1):
	nextPosition = originalPosition
	
	if action == 0:
		return originalPosition
	elif action == 1: # up one step
		nextPosition[0] += 1
	elif action == 2: # right one step
		nextPosition[1] += 1
	elif action == 3: # down one step
		nextPosition[0] -= 1
	elif action == 4: # left one step
		nextPosition[1] -= 1
	else:
		print('Error: executeAction does not map to an action choice.')

	if nextPosition[0] > (spaceArrMax0 - 1):
		nextPosition[0] = spaceArrMax0
	if nextPosition[0] < 0:
		nextPosition[0] = 0
	if nextPosition[1] > (spaceArrMax1 - 1):
		nextPosition[1] = spaceArrMax1
	if nextPosition[1] < 0:
		nextPosition[1] = 0

	return nextPosition
def policyLookUp(alphaVectorTbl, currRobotPos, currWorkerPos, currWorkerVel, goalProb):
	action = -1
	maxV = -100000

	currRobot = (currRobotPos[0] * spaceArr.shape[1] + currRobotPos[1])
	currWorker = (currWorkerPos[0] * spaceArr.shape[1] + currWorkerPos[1])
	currState = currRobot * (spaceArr.shape[0]*spaceArr.shape[1]) * ((len(velocityArr)-1) * len(orientationArr) + 1) \
		+ currWorker * ((len(velocityArr)-1) * len(orientationArr) + 1) \
		+ currWorkerVel


	for i in range(alphaVectorTbl.shape[0]):
		if alphaVectorTbl[i][0] == currState:
			V = np.dot(alphaVectorTbl[i][2:(len(goalPositions)-1)+2], goalProb)
			if V > maxV:
				maxV = V
				action = int(alphaVectorTbl[i][1])

	return action

### Main function
if __name__ == "__main__":
	# Factory information input:
	FactoryMaxX = 5
	FactoryMaxY = 5
	spaceArr = np.ones((FactoryMaxY,FactoryMaxX))

	# Worker's possible goal location
	goalPositions = np.array([[4,4], [0,2], [2,0]])

	# Robot information (create robot dynamics):
	velocityArr = np.arange(2)
	orientationArr = np.arange(4)
	robotInitPos = np.array([0,0])
	robotPos =  np.copy(robotInitPos)
	robotGoalIdx = 0
	robotGoalPosition = np.copy(goalPositions[robotGoalIdx])
	actionSize = 1 + ((velocityArr.shape[0] - 1) * orientationArr.shape[0])

	# Initiate worker information and status:
	workerPolicies = np.ndarray((2,), dtype=object)
	workerInitPos = np.array([FactoryMaxY-1, FactoryMaxX-1])
	workerPos = np.copy(workerInitPos)
	for goalIdx in range(1, goalPositions.shape[0]):	# 1 or 2
		workerGoalPosition = np.copy(goalPositions[goalIdx])
		workerP, workerR = calculateTAndRMatrix(spaceArr, velocityArr, orientationArr, workerGoalPosition)
		workerVi = mdptoolbox.mdp.ValueIterationGS(workerP, workerR, 0.95, epsilon=0.01)
		workerVi.run()
		workerPolicies[goalIdx-1] = workerVi.policy
		print("policy = ", workerVi.policy)

	# Solve pomdp with SARSOP 
	T, R = calculateTAndRMatrix(spaceArr, velocityArr, orientationArr, robotGoalPosition)
	genPomdpXFile(spaceArr, velocityArr, orientationArr, goalPositions, T, workerPolicies)

	### Simulation ###
	fileName = input("Input policy file name: ")
	alphaVectorTbl = sarsopPolicy(fileName)

	# simulation parameters
	currRobotPos = [0,0]
	currWorkerPos = [FactoryMaxY-1, FactoryMaxX-1]
	currWorkerVel = velocityArr[0] 
	prevRobotPos = currRobotPos
	prevWorkerPos = currWorkerPos
	prevWorkerVel = currWorkerVel
	workerGoalIdx = WORKERGOALIDX  # 1 or 2
	workerPolicy = workerPolicies[workerGoalIdx-1]
	goalProb = np.full(len(goalPositions)-1, 1.0/(len(goalPositions)-1))
	gamma = BELIEFGAMMA

	# simulation output files
	simMapFile = open("simulatedMap.txt", "w")
	opfile = open("positions"+ str(workerGoalIdx) + ".txt", "w")

	for i in range(int(spaceArr.shape[0]*spaceArr.shape[1])):

		prevRobotPos = currRobotPos
		prevWorkerPos = currWorkerPos
		prevWorkerVel = currWorkerVel

		robotNxtAction = policyLookUp(alphaVectorTbl, prevRobotPos, prevWorkerPos, prevWorkerVel, goalProb)
		currWorkerVel = workerPolicy[prevWorkerPos[0] * spaceArr.shape[1] + prevWorkerPos[1]]

		# belief update (based on current state and action to the next state)
		policyActions = np.zeros([(len(goalPositions)-1), actionSize])
		for goalIdx in range(len(goalPositions)-1):
			policyActionIdx = workerPolicies[goalIdx][(prevWorkerPos[0] * spaceArr.shape[1] + prevWorkerPos[1])]
			policyActions[goalIdx, int(policyActionIdx)] += 1

		sumOfActionFreq = policyActions[:, currWorkerVel].sum(axis=0)
		sumProb = 0.0
		if sumOfActionFreq > 0:
			for j in range(len(goalPositions)-1):
				tmp = (policyActions[j, currWorkerVel] / sumOfActionFreq)
				if tmp < 0.001:
					tmp = 0.001
				goalProb[j] = goalProb[j]*(1-gamma) + (tmp*gamma)
				sumProb += goalProb[j]
			goalProb = goalProb/sumProb

		printCurrMap(simMapFile, spaceArr, currRobotPos, currWorkerPos, robotGoalIdx, workerGoalIdx, goalPositions)
		printStates(opfile, prevRobotPos, prevWorkerPos, goalPositions, goalProb)
		printNxtAction(opfile, velocityArr, orientationArr, robotNxtAction, 'Robot\'s')
		printNxtAction(opfile, velocityArr, orientationArr, currWorkerVel, 'Worker\'s')
		opfile.write('\n')

		currRobotPos = executeAction(prevRobotPos, robotNxtAction, spaceArr.shape[0], spaceArr.shape[1])
		currWorkerPos = executeAction(prevWorkerPos, currWorkerVel, spaceArr.shape[0], spaceArr.shape[1])

		if currRobotPos[0] == goalPositions[0][0] and currRobotPos[1] == goalPositions[0][1]:
			printCurrMap(simMapFile, spaceArr, currRobotPos, currWorkerPos, robotGoalIdx, workerGoalIdx, goalPositions)
			printStates(opfile, prevRobotPos, prevWorkerPos, goalPositions, goalProb)
			print('\nEnd of simulation')
			break

	opfile.close()
	simMapFile.close()



