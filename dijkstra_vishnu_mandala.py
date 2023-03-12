import numpy as np
import time
import cv2
from queue import PriorityQueue

# Define the move functions
move_up = lambda node: ((node[0] - 1, node[1]), 1)
move_down = lambda node: ((node[0] + 1, node[1]), 1)
move_left = lambda node: ((node[0], node[1] - 1), 1)
move_right = lambda node: ((node[0], node[1] + 1), 1)
move_up_left = lambda node: ((node[0] - 1, node[1] - 1), np.sqrt(2))
move_up_right = lambda node: ((node[0] - 1, node[1] + 1), np.sqrt(2))
move_down_left = lambda node: ((node[0] + 1, node[1] - 1), np.sqrt(2))
move_down_right = lambda node: ((node[0] + 1, node[1] + 1), np.sqrt(2))

#Define the Obstacle Equations and Map Parameters
eqns = {
    "Rectangle1": lambda x, y: 0 <= y <= 100 and 100 <= x <= 150,
    "Rectangle2": lambda x, y: 150 <= y <= 250 and 100 <= x <= 150,
    "Hexagon": lambda x, y: (75/2) * abs(x-300)/75 + 50 <= y <= 250 - (75/2) * abs(x-300)/75 - 50 and 225 <= x <= 375,
    "Triangle": lambda x, y: (200/100) * (x-460) + 25 <= y <= (-200/100) * (x-460) + 225 and 460 <= x <= 510
}

map_width, map_height, clearance = 600, 250, 5
pixels = np.full((map_height, map_width, 3), 255, dtype=np.uint8)

for i in range(map_height):
    for j in range(map_width):
        is_obstacle = any(eqn(j, i) for eqn in eqns.values())
        if is_obstacle:
            pixels[i, j] = [0, 0, 0]  # obstacle
        else:
            is_clearance = any(
                eqn(x, y)
                for eqn in eqns.values()
                for y in range(max(i-clearance, 0), min(i+clearance+1, map_height))
                for x in range(max(j-clearance, 0), min(j+clearance+1, map_width))
            )
            if i < clearance or i >= map_height - clearance or j < clearance or j >= map_width - clearance:
                pixels[i, j] = [192, 192, 192]  # boundary
            elif is_clearance:
                pixels[i, j] = [192, 192, 192]  # clearance
            else:
                pixels[i, j] = [255, 255, 255]  # free space

# Define the start and goal nodes
def is_valid_node(node):
    x, y = node
    y = map_height - y - 1
    return 0 <= x < map_width and 0 <= y < map_height and (pixels[y, x] == [255, 255, 255]).all()

# Define a function to check if current node is the goal node
def is_goal(current_node, goal_node):
    return current_node == goal_node

# Define a function to find the optimal path
def backtrack_path(parents, start_node, goal_node, pixels):
    path, current_node = [goal_node], goal_node
    while current_node != start_node:
        path.append(current_node)
        current_node = parents[current_node]
        pixels[map_height-1-current_node[1], current_node[0]] = (0, 255, 0)  # Mark path (in green)
    path.append(start_node)
    return path[::-1]

# Define the Dijkstra algorithm
def dijkstra(start_node, goal_node, display_animation=True):
    open_list = PriorityQueue()
    closed_list = set()
    cost_to_come = {start_node: 0}
    cost = {start_node: 0}
    parent = {start_node: None}
    open_list.put((0, start_node))
    visited = set([start_node])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('animation.mp4', fourcc, 15.0, (map_width, map_height))
    # Loop until the open list is empty
    while not open_list.empty():
        _, current_node = open_list.get()
        closed_list.add(current_node)
        pixels[map_height -1- current_node[1], current_node[0]] = (255, 0, 0)  # Mark current node as visited (in blue)
        out.write(pixels)
        if display_animation:
            cv2.imshow('Explored', pixels)
            cv2.waitKey(1)
        # Check if the current node is the goal node
        if is_goal(current_node, goal_node):
            path = backtrack_path(parent, start_node, goal_node, pixels)
            if display_animation:
                cv2.imshow('Optimal Path', pixels)
                cv2.waitKey(0)
            print("Final Cost: ", cost[goal_node])
            out.release()
            cv2.destroyAllWindows()
            return path
        # Generate the children of the current node
        for move_func in [move_up, move_down, move_left, move_right, move_up_left, move_up_right, move_down_left, move_down_right]:
            new_node, move_cost = move_func(current_node)
            # Check if the new node is valid and not already visited
            if is_valid_node(new_node) and new_node not in closed_list:
                new_cost_to_come = cost_to_come[current_node] + move_cost
                # Check if the new node is already in the open list
                if new_node not in cost_to_come or new_cost_to_come < cost_to_come[new_node]:
                    cost_to_come[new_node] = new_cost_to_come
                    cost[new_node] = new_cost_to_come
                    parent[new_node] = current_node
                    open_list.put((new_cost_to_come, new_node))
                    visited.add(new_node)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
    out.release()
    cv2.destroyAllWindows()
    return None

# Get valid start and goal nodes from user input
while True:
    start_node = tuple(map(int, input("\nEnter the start node (in the format 'x y'): ").split()))
    if not is_valid_node(start_node):
        print("Error: Start node is in the obstacle space, clearance area or out of bounds. Please input a valid node.")
        continue
    goal_node = tuple(map(int, input("Enter the goal node (in the format 'x y'): ").split()))
    if not is_valid_node(goal_node):
        print("Error: Goal node is in the obstacle space, clearance area or out of bounds. Please input a valid node.")
        continue
    break

# Run Dijkstra's algorithm
start_time = time.time()
path = dijkstra(start_node, goal_node)
if path is None:
    print("\nError: No path found.")
else:
    print("\nGoal Node Reached!\nShortest Path:", path, "\n")
end_time = time.time()
print("Runtime:", end_time - start_time, "seconds\n")