import pygame
import numpy
from os.path import exists, dirname, join

colors = {
    -1: (206, 171, 147), #grid
    0: (255, 251, 233), #empty
    1: (227, 202, 165), #occupied
    2: (79, 189, 186), #start
    3: (246, 137, 137), #end
    4: (255, 0, 0), #frontier
    5: (0, 0, 255), #explored
    6: (0, 255, 0), #path
}

class maze:
    window_size = (706, 706)
    #window_size = (310, 200) # for mini mazes
    cell_properties = {"height":20,"width":20,"margin":2}
    sizes = (33,33)
    #sizes = (9,14) # for mini mazes
    timeout = 100000
    done = False
    found = False
    start_exist = False
    end_exist = False
    start = None
    end = None
    algorithm = None
    algorithm_name = "" 
    maze_id = 1
    path = [] 
    saveImages = False


    def __init__(self, maze_id=1, save_img = False) -> None:
        pygame.init()
        pygame.display.set_caption("LUND - LTH - MAZE {}".format(self.maze_id))
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()
        self.grid = numpy.zeros(self.sizes)
        self.maze_id=maze_id
        self.saveImages = save_img
        self.resetgrid(fill=False)
        maze_path = "./mazes/maze{}.txt".format(self.maze_id)
        self.loadgrid(maze_path)
        print(maze_path, "is loaded")  

    def __del__(self) -> None:
        pygame.quit()

    def loadSearchAlgorithm(self, algorithm_name, initial=False):
        module = "searchalgorithms."+algorithm_name
        if exists(join(dirname(dirname(__file__)),"searchalgorithms",algorithm_name)+".py"):
            m = __import__(module)
            algo = getattr(m, algorithm_name)
            self.algo = getattr(algo, algorithm_name)
            self.algorithm_name = algorithm_name
            print(algorithm_name, "is loaded")
            # Throw exception if nothing has been implemented yet
            try:
                self.algo() 
            except NotImplementedError as e:
                print(f"Error: {e}")
                # The program will stop here because the exception was raised
                exit(1)            
        else:
            self.algorithm_name = ""
            if not initial:
                print("No algorithm loaded. Could not find",algorithm_name, "algorithm")
            else:
                print("No algorithm loaded")
            return False

    def loadgrid(self, path) -> None:
        self.grid = numpy.loadtxt(path)
        for row in range(self.sizes[0]):
            for column in range(self.sizes[1]):
                if self.grid[row][column] == 2:
                    self.start_exist = True
                    self.start = (row,column)
                elif self.grid[row][column] == 3:
                    self.end_exist = True
                    self.end = (row,column)

    def resetgrid(self,fill=False) -> None:
        if fill:
            self.grid = numpy.ones(self.sizes)
        else:
            self.grid = numpy.zeros(self.sizes)
        self.start_exist = False
        self.end_exist = False

    def draw(self) -> None:
        f = []
        e = []
        if self.algorithm:
            f = self.algorithm.getFrontier()
            e = self.algorithm.getExplored()
            
        self.screen.fill(colors[-1])
        for row in range(self.sizes[0]):
            for column in range(self.sizes[1]):
                if self.grid[row][column] in colors:
                    if (row,column) in self.path:
                        color = colors[6]
                    elif (row,column) in e:
                        color = colors[5]
                    elif (row,column) in f:
                        color = colors[4]
                    else:
                        color = colors[self.grid[row][column]]
                pygame.draw.rect(self.screen, color, 
                [self.cell_properties["margin"] + (self.cell_properties["margin"] + self.cell_properties["width"]) * column, 
                self.cell_properties["margin"] + (self.cell_properties["margin"] + self.cell_properties["height"]) * row, 
                self.cell_properties["width"], self.cell_properties["height"]])
        pygame.display.flip() 
        self.clock.tick(60)
                                               
    def runSearchAlgorithm(self) -> None:
        pygame.display.set_caption("LUND - LTH - MAZE {} - Algorithm {}".format(self.maze_id, self.algorithm_name))
        frame_no = 0        
        # start running
        if self.algorithm_name == "":
            print("No algorithm selected")
            return
        self.algorithm = self.algo()
        if not self.start_exist:
            print("Start position is not set")
        elif not self.end_exist:
            print("End position is not set")
        else:
            count = 0
            self.algorithm.reset(self.grid, self.start, self.end)
            while count<self.timeout and not self.algorithm.isDone():
                pygame.event.get() #to prevent freezing
                self.algorithm.step()
                self.draw()
                if self.saveImages:
                    # save the output to generate gif
                    outputgifname = "./images/gif_images/maze_{0}_algorithm_{1}_frame_{2}.png".format(self.maze_id,self.algorithm_name,frame_no)
                    pygame.image.save(self.screen,outputgifname) 
                    frame_no +=1 
            if count>=self.timeout:
                print("Timeout")
            else:
                print("------------------------")
                print("Found Path with", self.algorithm.getCost(),"cost")
                print("Expanded", self.algorithm.getNumberOfExpanded(),"nodes")
                print("Max Frontier size", self.algorithm.getMaxFrontierSize())                
                print("Max node memory size", self.algorithm.getMaxMemoryUsage())            
                print("Max depth", self.algorithm.getMaxDepth())
                print("------------------------")
                all_path = self.algorithm.getPath()
                for p in all_path:
                    self.path.append(p)
                    self.draw()
                    if self.saveImages:
                        # save the output to generate gif
                        outputgifname = "./images/gif_images/maze_{0}_algorithm_{1}_frame_{2}.png".format(self.maze_id,self.algorithm_name,frame_no)
                        pygame.image.save(self.screen,outputgifname) 
                        frame_no +=1

        # save the output
        outputfilename = "./images/maze_{0}_algorithm_{1}.png".format(self.maze_id,self.algorithm_name)
        pygame.image.save(self.screen,outputfilename)  
        
        # show the final maze and computed solution until pressing Escape!
        while not self.done:
            self.draw()
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT:
                    self.done = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                            print("Exit")
                            pygame.quit()
                            self.done = True
                            break
                    else:
                        self.draw() 