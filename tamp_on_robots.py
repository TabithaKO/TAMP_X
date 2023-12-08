# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse

def organize_tuples(str_list):
    try:
        remove_braces = (str_list.split("[")[1]).split("]")[0]
        values = []
        val_list = remove_braces.split(",")
        for k in val_list:
            values.append(int(k.strip("[,( ) ,]")))
        tuple_list = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
        return tuple_list
    except:
        values = []
        val_list = str_list.split(",")
        print(val_list)
        for k in val_list:
            values.append(int(k.strip("[,( ) ,]")))
        tuple_list = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
        return tuple_list

def get_data(file):
    file_content = open(file)
    file_content = csv.reader(open(file))
    file_values = []
    #split the content into the rows
    for row in file_content:
        file_values.append(row)

    time = float(file_values[1][1])
    num_collisions = int(file_values[2][1])
    dock_spot = organize_tuples(file_values[3][1])
    target_positions = organize_tuples(file_values[4][1])
    agent_start_positions = organize_tuples(file_values[5][1])
    num_agents = int(file.split("_")[1])
    agent_paths = []
    for i in range(6, num_agents+6):
        agent_paths.append(organize_tuples(file_values[i][1]))

    return time, dock_spot, num_collisions, num_agents, target_positions, agent_start_positions, num_agents, agent_paths

# program agenets to follow their timesteps from simulation synchronously
# At each step we'll need to determine what action to take: forward, backward, left, right

def step_look_up(agent, curr_step, next_step):
    # foward (assuming forward in the y axis)
    if tuple(np.subtract(curr_step, next_step)) == (0, 1):
        # go forward command
        print("-"*20)
        print("Agent:", agent.agent_name,"is preparing to go forward!")
        print("*"*20)
        dist = "{distance: 0.1,max_translation_speed: 1}"
        agent_spec = "/"+agent.name+"/drive_distance irobot_create_msgs/action/DriveDistance"
        !ros2 action send_goal $agent_spec $dist
        print("-"*20)
        print("Agent:", agent.agent_name,"moved forward!")
        print("*"*20)

    elif tuple(np.subtract(curr_step, next_step)) == (0, -1):
        # rotate 180 degrees and go forward (2 commands)
        print("-"*20)
        print("Agent:", agent.agent_name,"is preparing to move backwards!")
        print("*"*20)
        angle = "{angle: 3.14,max_rotation_speed: 6}"
        agent_spec = "/"+agent.name+"/rotate_angle irobot_create_msgs/action/RotateAngle"
        !ros2 action send_goal $agent_spec $angle
        dist = "{distance: 0.1,max_translation_speed: 1}"
        agent_spec = "/"+agent.name+"/drive_distance irobot_create_msgs/action/DriveDistance"
        !ros2 action send_goal $agent_spec $dist
        print("-"*20)
        print("Agent:", agent.agent_name,"moved backwards!")
        print("*"*20)

    elif tuple(np.subtract(curr_step, next_step)) == (1, 0):
        # go right
        print("-"*20)
        print("Agent:", agent.agent_name,"is preparing to go right!")
        print("*"*20)
        angle = "{angle: 1.57,max_rotation_speed: 6}"
        agent_spec = "/"+agent.name+"/rotate_angle irobot_create_msgs/action/RotateAngle"
        !ros2 action send_goal $agent_spec $angle
        dist = "{distance: 0.1,max_translation_speed: 0.3}"
        agent_spec = "/"+agent.name+"/drive_distance irobot_create_msgs/action/DriveDistance"
        !ros2 action send_goal $agent_spec $dist
        print("-"*20)
        print("Agent:", agent.agent_name,"turned right!")
        print("*"*20)

    elif tuple(np.subtract(curr_step, next_step)) == (-1, 0):
        print("-"*20)
        print("Agent:", agent.agent_name,"is preparing to turn left!")
        print("*"*20)
        # go right
        angle = "{angle: 4.71,max_rotation_speed: 6}"
        agent_spec = "/"+agent.name+"/rotate_angle irobot_create_msgs/action/RotateAngle"
        !ros2 action send_goal $agent_spec $angle
        dist = "{distance: 0.1,max_translation_speed: 0.3}"
        agent_spec = "/"+agent.name+"/drive_distance irobot_create_msgs/action/DriveDistance"
        !ros2 action send_goal $agent_spec $dist
        print("-"*20)
        print("Agent:", agent.agent_name,"turned left!")
        print("*"*20)
    else:
        print("-"*20)
        print("Agent:", agent.agent_name,"is preparing to go home!")
        print("*"*20)
        # go home, you might be lost
        loc = "{achieve_goal_heading: true,goal_pose:{pose:{position:{x: 0,y: 0.2,z: 0.0}, orientation:{x: 0.0,y: 0.0, z: 0.0, w: 1.0}}}}"
        agent_spec = "/"+agent.name+"/navigate_to_position irobot_create_msgs/action/NavigateToPosition"
        !ros2 action send_goal $agent_spec $loc
        print("-"*20)
        print("Agent:", agent.agent_name,"went home!")
        print("*"*20)

# This function check synchronizes agent steps
def check_step_count(global_step_count, my_step):
  new_step_count = list(filter(lambda x: x!= -1, global_step_count))
  return all(i == my_step for i in new_step_count)

class agent(object):
  def __init__(self, agent_id, agent_name, step_num, curr_path, status):
    self.agent_id = agent_id
    self.step_num = step_num
    self.curr_path = curr_path
    self.agent_name= agent_name
    self.status = status

# This function filters the current agent's position out of the list of all agent positions
def filter_positions(agent_id, path_list):
  return path_list[0:agent_id]+ path_list[agent_id+1:]

def set_up(file_name):
    global global_step_count
    time, dock_spot, num_collisions, num_agents, target_positions, agent_start_positions, num_agents, agent_paths = get_data(file_name)
    robot_1 =  agent(agent_id=0,agent_name="robot_1", step_num = 0, agent_paths = agent_paths[0], status ="work")
    robot_3 = agent(agent_id=1,agent_name="robot_3", step_num = 0, agent_paths = agent_paths[1], status ="work")
    global_step_count = [0,0]

    return robot_1, robot_3

def step(agent):
    global global_step_count
    if agent.status == "work":
        curr_step = agent.curr_path[0]
        next_step = agent.curr_path[1]
        step_look_up(curr_step, next_step)
        try:
            agent.curr_path = agent.curr_path[1:]
        except:
            agent.status = "go_home"
    else:
        # send the agent home
        global_step_count = filter_positions(agent.agent_id, global_step_count)

    while check_step_count(global_step_count, agent.step_num) is False:
        continue
    if agent.status == "go_home":
        step_look_up(agent,(0,0),(10,10))
    else:
        agent.step_num += 1
        global_step_count[agent.id] = agent.step_num
        step(agent)

def main(args):
    agent_1, agent2 = set_up(args.filename)
    step(agent_1)
    step(agent_1)
    print("!!! DONE WITH SIMULATION!!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str,
                        help='path to the input csv')
    args = parser.parse_args()
    main(args)

