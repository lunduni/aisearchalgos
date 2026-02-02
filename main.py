from src.maze import maze 
import argparse

parser = argparse.ArgumentParser(description='Search Algorithm Parameters')
parser.add_argument('--maze_id', type=int, required=True,
                    help='the id of the maze')
parser.add_argument('--search_algorithm_name', type=str, required=True,
                    help='the search algorithm name') 
args = parser.parse_args()
my_algorithm_name = args.search_algorithm_name

if args.maze_id > 6: # there are 6 mazes predefined
        raise ValueError("Argument --maze_id: {args.maze_id} is greater than allowed threshold {6}")

my_maze = maze(maze_id=args.maze_id, save_img= False)
my_maze.loadSearchAlgorithm(algorithm_name=my_algorithm_name)
my_maze.runSearchAlgorithm()