from search import *
import random
import time

def make_rand_8puzzle():
    input = [1,2,3,4,5,6,7,8,0]
    while (True):
        random.shuffle(input)
        puzz = EightPuzzle(input)
        if(puzz.check_solvability(input)):
            puzz.initial = tuple(puzz.initial)
            break
    return puzz

def display(state):
    new_state = list(state).copy()
    output = ''
    dictionary = {0: '*', 1: '1', 2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8'}
    for i in range(len(new_state)):
        output += dictionary[new_state[i]]
        if(i % 3 == 2 and i != len(new_state)):
            output += '\n'
        else:
            output += ' '
    print(output)
    

def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    # copied from the textbook code so that it can be modified
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print('path length', len(node.path())-1)
                print(len(explored)+1, "nodes removed from the frontier")
            path_len = len(node.path())-1
            nodes_removed = len(explored)+1
            return path_len, nodes_removed
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None 

def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    # unchanged from textbook
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


def manhattan(node):
    state = list(node.state)
    dist = 0
    for i in range(9):
        # the value that is currently stored in the index
        val = state[i]     
        if val == 0:
            continue
        # where the value should be located
        x1 = (val-1)%3
        y1 = (val-1)//3
        # where it is currently located
        x2 = i%3
        y2 = i//3
        # manhattan distance per piece of the puzzle
        dist += abs(x1-x2) + abs(y1-y2)
    return dist

def max_misplaced_manhattan(node):
    return max(sum(s != g for (s, g) in zip(node.state, (1, 2, 3, 4, 5, 6, 7, 8, 0))), manhattan(node))
# ____________________________________________________________



eight_puzzles = []
for i in range(10):
    eight_puzzles.append(make_rand_8puzzle())
eight_lens_missing = []
eight_rem_missing = []
eight_times_missing = []
eight_lens_manhattan = []
eight_rem_manhattan = []
eight_times_manhattan = []
eight_lens_max = []
eight_rem_max = []
eight_times_max = []
for puzz in eight_puzzles:
    start = time.time()
    length, rem = astar_search(problem=puzz,h=puzz.h, display=False)
    end = time.time()
    eight_lens_missing.append(length)
    eight_rem_missing.append(rem)
    eight_times_missing.append((end-start)*1000)            # time in ms
    start = time.time()
    length, rem = astar_search(problem=puzz,h=manhattan, display=False)
    end = time.time()
    eight_lens_manhattan.append(length)
    eight_rem_manhattan.append(rem)
    eight_times_manhattan.append((end-start)*1000) 
    start = time.time()
    length, rem = astar_search(problem=puzz,h=max_misplaced_manhattan, display=False)
    end = time.time()
    eight_lens_max.append(length)
    eight_rem_max.append(rem)
    eight_times_max.append((end-start)*1000) 
print('Eight puzzle, missing tile:')
for i in range(10):
    print(eight_times_missing[i],'ms,',eight_lens_missing[i],'path length,',eight_rem_missing[i],'nodes removed from frontier')
print('Eight puzzle, manhattan:')
for i in range(10):
    print(eight_times_manhattan[i],'ms,',eight_lens_manhattan[i],'path length,',eight_rem_manhattan[i],'nodes removed from frontier')
print('Eight puzzle, max:')
for i in range(10):
    print(eight_times_max[i],'ms,',eight_lens_max[i],'path length,',eight_rem_max[i],'nodes removed from frontier')