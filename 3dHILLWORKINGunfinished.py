from direct.showbase.ShowBase import ShowBase
from panda3d.core import *

from direct.task import Task
import numpy as np
import noise

class Terrain:
    def __init__(self, size):
        self.size = size #store terrain size
        self.height_map = self.generate_height_map() 
        self.target_location = np.unravel_index(np.argmax(self.height_map, axis=None), self.height_map.shape)

    def generate_height_map(self):
        scale = 100.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        height_multiplier = 50 #scale height values

        world = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                world[i][j] = height_multiplier * noise.pnoise2(i/scale, 
                                            j/scale, 
                                            octaves=octaves, 
                                            persistence=persistence, 
                                            lacunarity=lacunarity, 
                                            repeatx=1024, 
                                            repeaty=1024, 
                                            base=42)
                #using perlin noise to generate smooth , natural looking heights
        return world

    def get_height(self, x, y):
        if x < self.size and y < self.size:
            return self.height_map[x, y]
        else:
            return -np.inf # when outside the terrain, return a very low height


class Agent:
    def __init__(self, start_location, goal_location, temperature=100):
        self.current_location = start_location
        self.goal_location = goal_location
        self.temperature = temperature
        self.trail = [self.current_location]
        self.visited = set()

    def sense_surroundings(self, terrain):
        x, y = self.current_location

        #create dictionary of heights of surrounding tiles, 3x3 grid including center
        surrounding_heights = {(dx, dy): terrain.get_height((x + dx) % terrain.size, (y + dy) % terrain.size)
                               for dx in range(-1, 2)
                               for dy in range(-1, 2)}
        
        #return dictionary
        return surrounding_heights

    def decide_next_move(self, terrain):
        #sense surrounding terrain to get heights of neighbours
        surroundings = self.sense_surroundings(terrain)
        direct_neighbors = {(dx, dy): height for (dx, dy), height in surroundings.items() if abs(dx)<=1 and abs(dy)<=1 and not (dx == 0 and dy == 0)} 
        # ignores current tile, direct neighbour any tile exactly 1 step away
        unvisited_neighbors = {(dx, dy): height for (dx, dy), height in direct_neighbors.items() if (self.current_location[0]+dx, self.current_location[1]+dy) not in self.visited}

        #if all neighbors have been visited: select from all neighbors
        #otherwise select from unvisited neighbors, not in the visited set
        chosen_neighbors = unvisited_neighbors if unvisited_neighbors else direct_neighbors

        #find the best move - select highest neighbour
        best_move = max(chosen_neighbors, key=chosen_neighbors.get)
        best_height = chosen_neighbors[best_move]
        delta_height = best_height - terrain.get_height(*self.current_location)
        acceptance_probability = min(1, np.exp(delta_height / self.temperature))
        # acceptance probability uses simulated annealing to avoid getting stuck at local minima by always selecting highest slope
        #allows agent to choose moves which are not always to the highest neighbour
        if np.random.random() < acceptance_probability:
            return best_move
        else:
            #if not taking the best move, chooses a  random move from avaliable tiles
            valid_moves = list(chosen_neighbors.keys())
            chosen_move = valid_moves[np.random.choice(len(valid_moves))]
            return chosen_move

    def make_move(self, move, terrain):
        #updates agent location based on chosen move
        self.current_location = ((self.current_location[0] + move[0]) % terrain.size,
                                 (self.current_location[1] + move[1]) % terrain.size)
        #modulo operator (%) means it will smoothly transition over the edge and relocate in the corresponding position on the other side, continuing its path as if it was a spherical surface to traverse
        #allows an endless loop around boundaries
        self.trail.append(self.current_location)
        self.visited.add(self.current_location)
        self.decrease_temperature()

    def decrease_temperature(self): #gradually reduce temp
        if self.temperature > 0:
            self.temperature -= 1


class Simulation(ShowBase):
    def __init__(self, terrain_size=1000, start_temperature=100):
        ShowBase.__init__(self)
        self.terrain = Terrain(terrain_size)
        start_location = (np.random.randint(0, terrain_size), np.random.randint(0, terrain_size))
        self.agent = Agent(start_location, start_temperature)

        self.time_accumulator = 0  #initialise time_accumulator

        self.build_environment()
        self.build_agent()

        self.taskMgr.add(self.update_task, "update_task")

    def build_environment(self):
        #format to store vertex data
        format = GeomVertexFormat.get_v3n3()
        vdata = GeomVertexData('vertices', format, Geom.UH_static)
        #set up writers
        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        #loop through grid of terrain points
        for i in range(self.terrain.size-1):
            for j in range(self.terrain.size-1):
                z = self.terrain.get_height(i, j) 
                # get height of current point
                z_right = self.terrain.get_height(i+1, j) 
                #get height of point to the right
                z_down = self.terrain.get_height(i, j+1)
                #get height of point below
                z_diag = self.terrain.get_height(i+1, j+1)
                #get height of diagonal


                #get normal vector for first triangle for lighting
                #basically to label which direction the part of terrain is pointing
                normal.addData3f(0, 0, 1)

                vertex.addData3f(i, j, z)
                #add first vertex of triangle (top left corner)
                vertex.addData3f(i+1, j, z_right)
                #add second vertex of triangle (top right)
                #forms top edge of the triangle with the first vertex
                vertex.addData3f(i, j+1, z_down)
                #add third vertex of triangle (bottom left)
                #completes triangle with first 2 vertices

                # add normal vector for second triangle
                normal.addData3f(0, 0, 1)
                #add vertex positions
                vertex.addData3f(i+1, j, z_right)
                vertex.addData3f(i, j+1, z_down)
                vertex.addData3f(i+1, j+1, z_diag)

        tris = GeomTriangles(Geom.UH_static)
        #tells panda3d the geometry is static to allow optimisations
        #loop through each triangle using 3 vertices, connect vertices in the order they are added
        for i in range(0, vdata.get_array(0).get_num_rows(), 3):
            tris.addVertices(i, i+1, i+2)

        geom = Geom(vdata) #geom object to hold vertex data
        geom.add_primitive(tris) #connects dots in each triangle to make surface

        node = GeomNode('terrain') #geomnode to hold geometryobject
        node.add_geom(geom) #attatch the geom (vertex data/triangle) to the geomnode
        #attaches geomnode to render tree
        terrain_np = self.render.attach_new_node(node) 
        terrain_np.set_color(0, 1, 0, 1) # make terrain green



        # LIGHTING - WIP
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.5, 1))
        dlnp = self.render.attach_new_node(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)

        alight = AmbientLight('alight')
        alight.setColor((0.2, 0.2, 0.2, 1))
        alnp = self.render.attach_new_node(alight)
        self.render.setLight(alnp)

    def build_agent(self):
        self.agent_np = self.loader.loadModel("models/smiley") #set avatar
        self.agent_np.reparentTo(self.render) #render avatar
        self.agent_np.setScale(0.5, 0.5, 0.5)
        self.trail_np = self.loader.loadModel("models/smiley")
        self.trail_np.reparentTo(self.render)
        self.trail_np.setScale(0.1, 0.1, 0.1)

    def update_task(self, task):
        #accumulate time
        self.time_accumulator += globalClock.getDt()
        #check if time to update
        if self.time_accumulator >= 0.5:
            self.time_accumulator -= 0.5

            if self.agent.current_location != self.terrain.target_location: #if not at the goal
                move = self.agent.decide_next_move(self.terrain) #decide the next move
                self.agent.make_move(move, self.terrain) #make the move
                #update agent position
                agent_pos = Point3(self.agent.current_location[0], self.agent.current_location[1], 
                                   self.terrain.get_height(*self.agent.current_location))
                self.agent_np.setPos(agent_pos)
                
                #add trail to log movement
                trail_np = self.loader.loadModel("models/smiley") #build trail 
                trail_np.reparentTo(self.render)  # attatch to scene to render
                trail_np.setScale(0.05, 0.05, 0.05) # scale down to amke smaller
                trail_np.setColor(1, 0, 0, 1) #make trail logged by small red smileys
                #position trail at agents current location
                trail_pos = Point3(self.agent.current_location[0], self.agent.current_location[1], 
                                   self.terrain.get_height(*self.agent.current_location))
                trail_np.setPos(trail_pos)

            self.agent.temperature -= 0.001 # reduce agent willingness to make non-optimal moves over time.

        return Task.cont



def main():
    sim = Simulation(terrain_size=100, start_temperature=100)
    sim.run()


if __name__ == "__main__":
    main()