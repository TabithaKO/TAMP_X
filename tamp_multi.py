import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# import pygraphviz
import argparse
import logging
import time
import pandas as pd
import threading
import math
import time
from threading import Thread
from random import choice
import pandas as pd
import random

# Defining some global variables
global_step_count = []
graph = []
target_spots = []

global_target_spots = []
global_graph = []
dock_spot = ()
collisions = 0

# This function defines the distance metric used in the A* algorithm
def np_euclidean(a,b):
  return np.linalg.norm(np.array(a) - np.array(b))

# To predict the agent futures I'm going to calculate the probabilities
# as they move about the space
# The action space is: up, down, left, right
# The probability of an action p(a|h,t) = SUM(p(a|t)*p(t|h))

# The probability that an agent is headed to a certain target, t, is p(t|h).
# We get this value from a table of probabilities that gets updated in every step
# If the distance between the agent and a target is decreasing in every step then the
# probability that the agent is going to that target location is high

# At every step, we compute p(a|h,t) = SUM(p(a|t)*p(t|h)), where p(a|t) is the inverse
# of the length of the A* path to that targetif we begin with the action 'a'
def get_next_steps(Graph, val):
  global graph
  node_up = (val[0]+ 1, val[1])
  node_down = (val[0]- 1, val[1])
  node_left = (val[0], val[1]- 1)
  node_right = (val[0], val[1]+ 1)
  options = [node_up, node_down, node_left, node_right]
  next_steps = []
  for i in options:
    if graph.has_node(i):
      next_steps.append(i)

  return next_steps

# This function gets the distance between two nodes on the graph
def get_dist(start, dest):
  global target_spots, graph
  try:
    if graph.has_node(dest) == False:
        # This is not a viable target
        return graph.size() + 1

    dist = nx.astar_path_length(graph, start, dest, heuristic=np_euclidean)
    return dist
  except:
     # No path exists between nodes
     return graph.size() + 1

# This function predicts the future steps of the other agents
def predict_agent_futures(Graph, agent, current_agent_positions, init_agent_positions):
  global global_target_spots, global_graph, global_step_count, dock_spot
  print("I am agent:", agent.agent_id,"with target:", agent.target, "in step:", agent.step_num)
  agent_futures = []
  
  for agent_i in range(0,len(init_agent_positions)):
    p_T_H = []
    # calculate p(t|h)
    for target_j in range(0,len(global_target_spots)):
      # calculate the change in distance between all the available targets
      curr_distance = get_dist(current_agent_positions[agent_i], global_target_spots[target_j])
      p_t = 1/(curr_distance + math.ulp(1.0))
      p_T_H.append(p_t)

    # calculating some p(a|t) = SUM(p(a|t)*p(t|h))
    p_A_T = []
    possible_directions = get_next_steps(graph, current_agent_positions[agent_i])
    # print("possible next steps:", possible_directions)
     # calculating p(a|t) for every target
    for pos_step in range(0,len(possible_directions)):
      p_a_curr_t = []
      for t in range(0, len(global_target_spots)):
        future_dist = get_dist(possible_directions[pos_step], global_target_spots[t])
        future_prob = (1/(future_dist + math.ulp(1.0))) * p_T_H[t]
        p_a_curr_t.append(future_prob)
      p_A_T.append(sum(p_a_curr_t))

    step_idx = p_A_T.index(max(p_A_T))
    predicted_step = possible_directions[step_idx]
    agent_futures.append(predicted_step)

  return agent_futures

# curr_path : The path in the agent
# path : A future path
def check_collision(curr_path, path):
    if curr_path[0] == path[0]:
      return True
    return False

# main_graph : This is a copy of the global graph
# other_agent_futures : Future predicted paths of the other agents
def block_collisions(main_graph, other_agent_futures):
  for path in other_agent_futures:
    for node in path:
      try:
        main_gaph = main_graph.remove_node(node)
      except:
        continue

  return main_graph

# curr_path : The current path of this agent
# other_agent_futures : Future predicted paths of the other agents
# target : The goal point of every agent
# This function check the validity of a path for the agent's future steps
def check_path(agent, curr_path, other_agent_futures, target):
  global graph, collisions
  new_graph = block_collisions(graph.copy(), other_agent_futures)
  print("In 'check path', the other agent futures are:", other_agent_futures)
  if curr_path[0] == other_agent_futures[0]:
    print("!! Possible collision! Searching for a new path!!")
    collisions += 1
    option_node = choice(get_next_steps(graph, curr_path[0]))
    print("The option node is:", option_node)
    out_path = nx.astar_path(new_graph, option_node, target, heuristic=np_euclidean)
    print("New Generated Path:", out_path)
    return out_path, 0
  print("!! No Collision! Moving on as planned !!")
  return curr_path, 1

# This is a custom thread that allows us to probe the agents after task completion
class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return

    def Lock(self):
        Thread.Lock(self)
        return self._return

# This class defines the agent's properties
class agent(object):
  def __init__(self, agent_id, step_num, curr_path, target, has_crazyflie, curr_agent_pos, former_agent_pos):
    self.agent_id = agent_id
    self.step_num = step_num
    self.curr_path = curr_path
    self.target = target
    self.crazyflie = has_crazyflie
    self.curr_agent_pos = curr_agent_pos
    self.former_agent_pos = former_agent_pos
    self.paths = []
    self.assigned_targets = []
   
# This function filters the current agent's position out of the list of all agent positions
def filter_positions(agent_id, path_list):
  return path_list[0:agent_id]+ path_list[agent_id+1:]


# This function randomly positions the targets on the graph
def agent_target_pos(graph_nodes, n_targets):
   print("Node List:", graph_nodes)
   target_pos = [choice(graph_nodes) for i in range(n_targets)]
   unique_list = []
   [unique_list.append(val) for val in target_pos if val not in unique_list]
   print("Target positions:", target_pos)
   [graph_nodes.remove(i) for i in unique_list]
   return unique_list

# This function outputs a list of agent start positions
def agent_start_pos(graph_nodes, n_agents):
   start_pos =  [choice(graph_nodes) for i in range(n_agents)]
   unique_list = []
   [val for val in start_pos if val not in unique_list]
   return start_pos
   
# This function check synchronizes agent steps
def check_step_count(global_step_count, my_step):
  new_step_count = list(filter(lambda x: x!= -1, global_step_count))
  return all(i == my_step for i in new_step_count)

# This function uses A* to get the shortest path between a pair of nodes
def get_path(start, dest):
  global graph
  return nx.astar_path(graph, start, dest, heuristic=np_euclidean)


# step_num : The current timestep in the program
# curr_path : The current path that the agent is following
# global_curr_agent_pos : The current positions of the other agents
# global_former_agent_pos : The positions of other agents at step(t - e)
# crazyflie : A binary value representing whether or not we are carrying a crazyflie
# target : The target location the agent is currently moving to


def step(agent, global_curr_agent_pos, global_former_agent_pos):
  step_time_start = time.time()
  global target_spots, graph, global_step_count, dock_spot, collisions
  lock = threading.Lock()
  if agent.crazyflie:
    print("In the crazyflie check condition!")
    print("Agent:", agent.agent_id,"current step is:",agent.curr_path[0] )
    # find a path to the drone drop-off point
    if agent.curr_path[0] == agent.target:
      print("Agent target is at crazyflie destination!")
      # Drop the crazyflie and return to the dock
      lock.acquire()
      print("Crazyflie, Lock Acquired!")
      print("Removing a node", agent.curr_path[0])
      try:
        agent.curr_path = get_path(agent.target, dock_spot)[1:]
        graph.remove_node(agent.target)
        agent.target = dock_spot
      except:
        print("Node", agent.curr_path[0], "has already been removed!")
      
      agent.crazyflie = 0
      print("Crazyflie, Lock Released!")
      print("Agent", agent.agent_id, "path:", agent.curr_path)
      lock.release()


  if agent.curr_path[0] == agent.target:
    print("Agent target is the Dock!")
    # Once you're at the dock spot, grab a crazyflie and update the new goal point
    lock.acquire()
    print("No crazyflie, Lock Acquired!")
    if len(target_spots) > 0:
        agent.target = target_spots[0]
        agent.crazyflie = 1
        target_spots = target_spots[1:]
        agent.curr_path = get_path(dock_spot, agent.target)
        agent.assigned_targets.append(agent.target)
    else:
       agent.target = None
       agent.curr_path = []
       agent.crazyflie = 0
       
    print("*************** Target Spots:", target_spots)
    print("No crazyflie, Lock Released!")
    lock.release()
    print("Agent:", agent.agent_id, "path:", agent.curr_path)
    print("Global targets:", target_spots)

  print("Agent:", agent.agent_id," is going to predict other agent futures")
  other_agent_futures = predict_agent_futures(graph, agent, agent.curr_agent_pos, agent.former_agent_pos)
  filtered_futures = filter_positions(agent.agent_id, other_agent_futures)
  print("I am agent:", agent.agent_id, "I predicted the other agent's future as:", filtered_futures)
  # print("The other agent futures:",filtered_futures)
  # print("My path is currenlty:", agent.curr_path)
  if len(agent.curr_path) > 0:
    updated_path, idx = check_path(agent.agent_id, agent.curr_path, filtered_futures, agent.target)

    print("--path update successful!--")
    # update the global 'curr_agent_pos' list so that the other agents can predict your future
    agent.curr_path = (updated_path[idx:])
    agent.paths.append(updated_path[idx:])

  global_curr_agent_pos[agent.agent_id] = updated_path[0]
  agent.step_num += 1
  global_step_count[agent.agent_id] = agent.step_num

  # Keep looping until the targets have been completed and you have completed the tast that you're on
  step_time_end = time.time()
  print("++++++++++++TIME:", (step_time_end - step_time_start))
  print(">>>>>>>", agent.agent_id,": is done with this step!")
  while check_step_count(global_step_count, agent.step_num) is False:
    continue

  print("About to hit the end step conditions as agent:", agent.agent_id, "at step:", agent.step_num, "with target:", agent.target, "and curr_step:", agent.curr_path[0])
  # print("However the full path is:", agent.curr_path)
  if agent.crazyflie:
    print("CONDITION 1")
    # agent task not done, keep going
    step(agent, global_curr_agent_pos, global_former_agent_pos)
  elif len(target_spots ) > 0 and not(agent.crazyflie):
    print("condition check --> Target spots:", target_spots)
    print("CONDITION 2")
    # task not done, keep going
    step(agent, global_curr_agent_pos, global_former_agent_pos)
  elif len(target_spots) == 0 and not(agent.crazyflie):
    print("CONDITION 3")
    # global_step_count = filter_positions(agent.agent_id, global_step_count)
    print("Agent:", agent.agent_id, "has crazyflie?", agent.crazyflie)
    global_step_count[agent.agent_id] = -1
    return agent.paths

def multi_thread(num_agents, num_targets, dock_spot):
    global graph, target_spots, global_target_spots, global_step_count
    graph_node_list = list(graph.nodes)
    graph_node_list.remove(dock_spot)
    random.shuffle(graph_node_list)

    # Try running the function
    try:
      target_spots = agent_target_pos(graph_node_list, num_targets)
      global_target_spots = target_spots                                                                      
      threads = []

      # initialize the agent positions
      agent_init_positions = agent_start_pos(graph_node_list, num_agents)
      # Handle possible duplicates in the random sampling
      num_agents = len(agent_init_positions)
      global_step_count = [0 for i in range(num_agents)]
      print("Initial Positions:", agent_init_positions)
      # create the agent objects
      agents = [agent(agent_id=i, step_num=0, curr_path=get_path(agent_init_positions[i], dock_spot), target=dock_spot, has_crazyflie=0, curr_agent_pos=agent_init_positions, former_agent_pos = agent_init_positions) for i in range(num_agents)]                                                                
      
      # begin timing from here
      t0 = time.time()
      for i in range(num_agents):                                                             
          # create the thread                                                     
          threads.append(                                                         
              CustomThread(target = step, args = (agents[i], agent_init_positions, agent_init_positions)))        
          threads[-1].start() # start the thread we just created                  
      print("Threads Initiated!")

      # wait for all threads to finish                                            
      for t in threads:                                                           
          t.join()   

      t1 = time.time()
      # returns all the paths 
      df = pd.DataFrame([t1-t0]+[collisions]+[dock_spot]+[global_target_spots] + [agent_init_positions] +[i.paths for i in agents])
      df.to_csv(args.output_path)
    except:
       df = pd.DataFrame(["Failed Execution!"])
       df.to_csv(args.output_path)
   
def main(args):
   global graph, global_step_count, dock_spot
   try:
    graph = nx.grid_2d_graph(args.graph_size, args.graph_size)
    dock_spot = random.choice(list(graph.nodes))
    multi_thread(args.num_agents, args.num_targets, dock_spot)
   except:
    print("Whoops!")
  

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str,
                        help='path to the output csv')
    parser.add_argument('--num_agents', type=int,
                        help='The number of agents in the experiment')
    parser.add_argument('--num_targets', type=int,
                        help='The number of targets in the experiment')
    parser.add_argument('--graph_size', type=int,
                        help='The length of one side of the square graph')
    args = parser.parse_args()
    main(args)

