import numpy as np
import math
import time
import matplotlib.pyplot as plt

import mesa.time
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from matplotlib.colors import ListedColormap

def distance(p, q, type="euclid"):
    if type=="euclid":
        return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
    elif type=="moore":
        return max(abs(p[0] - q[0]), abs(p[1] - q[1]))
    elif type=="neumann":
        return abs(p[0] - q[0]) + abs(p[1] - q[1])


'''
Mobility:
{"model" :"classic"}
{"model" :"gravity", "metric":"relevance", "alpha"=1, "beta"=2}         #relevance" OR "population"
{"model" :"radiation", "metric":"relevance"}                            #relevance" OR "population"

'''

class SchellingModel(Model):
    '''
    Model class for the Schelling segregation model.
    '''
    def __init__(self, side=10, density=0.7, population = 0.3, homophily=0.3,  mobility={"model" :"classic"},policy="random",  torus = False, n_towns=None):
        super().__init__()

        self.side = side
        self.density = density

        if type(population) != list:
            self.population = [population, 1-population]
        else:
            self.population = population

        self.homophily = homophily
        self.mobility = mobility
        self.policy = policy
        self.schedule = RandomActivation(self)

        self.radius_dict = {((0,0),0):[(0,0)]}
        self.grid = SingleGrid(side, side, torus= torus)

        self.cell_relevance = {(x, y): 1 for x in range(self.side) for y in range(self.side)}
        self.cell_neighpop = {(x, y): 1 for x in range(self.side) for y in range(self.side)}

        self.total_happy = 0
        self.total_segregation = None

     

        if n_towns != None:
            towns = {(self.random.randint(0, side), self.random.randint(0, side)):(self.random.random()*10
                +5, self.random.random()*0.5+0.1) for i in range(n_towns)}
            print(towns)
            relevance = lambda pos: towns[pos][0] if pos in towns else sum([towns[t][0]/distance(pos,t)**towns[t][1] for t in towns.keys()])
            self.cell_relevance = {(x,y): relevance((x,y)) for x in range(self.side) for y in range(self.side)}
            for t in towns.keys():
                self.cell_relevance[t] = max([self.cell_relevance[p] for p in self.grid.iter_neighborhood(t, "moore")])




        self.datacollector = DataCollector(
            {"total_happy": lambda m: m.total_happy,
             "total_segregation": lambda m: m.total_segregation},
            {"name": lambda a: a.hex_id,
             "type": lambda a: a.type,
             "pos": lambda a: (a.pos[0] ,a.pos[1]),
             "happy": lambda a: a.happy,
             "segreg": lambda a: a.segreg
             })

        self.running = True

        agent_id = 1
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]

            if self.random.random() < self.density:
                atype = self.random.choices(range(1,len(self.population)+1), self.population)[0]

                agent = SchellingAgent(agent_id, (x, y), self, atype, homophily, mobility= self.mobility, policy= self.policy)
                agent_id += 1
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        for a in self.schedule.agents:
            a.agent_stats()

        self.total_segregation = np.mean([a.segreg for a in self.schedule.agents])
        self.total_happy = sum([a.happy for a in self.schedule.agents])
        self.datacollector.collect(self)

    def step(self):

        self.schedule.step()

        for a in self.schedule.agents:
            a.agent_stats()
        self.total_segregation = np.mean([a.segreg for a in self.schedule.agents])
        self.total_happy = sum([a.happy for a in self.schedule.agents])

        self.datacollector.collect(self)
        if self.total_happy == self.schedule.get_agent_count():
            self.running = False


    def show(self,traces=[], labels=False, places= True, agents=True, savepdf=None):

        plt.figure(figsize=(5, 5), dpi=80)
        city = [[(agent.type, agent.hex_id) if agent != None else (0,"0") for agent in self.grid[k]] for k in range(self.grid.height)]


        city_id = [[city[i][k][1] for i in range(self.grid.height)] for k in range(self.grid.height)]

        plt.xticks(range(self.side))
        plt.yticks(range(self.side))

        plt.tick_params(axis='x', colors=(0, 0, 0, 0))
        plt.tick_params(axis='y', colors=(0, 0, 0, 0))

        col_dict = {0:"#aeaeae",
                    1: "#ff3300",
                    2: "#00ff33",
                    3: "#3300ff",
                    4: "#ccff00",
                    5:"#cc00ff" }

        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])


        if places:
            city_type = [[self.cell_relevance[(i, k)] for i in range(self.grid.height)] for k in range(self.grid.height)]
            plt.imshow(city_type, extent=(0, self.side, self.side, 0), cmap= "Greys_r")
        else:
            city_type = [[0 for i in range(self.grid.height)] for k in range(self.grid.height)]
            plt.imshow(city_type, extent=(0, self.side, self.side, 0), cmap="Greys_r")



        if agents:
            X = [a.pos[0] + 0.5 for a in self.schedule.agents]
            Y = [a.pos[1] + 0.5 for a in self.schedule.agents]
            clist = [col_dict[a.type] for a in self.schedule.agents]
            plt.scatter(X,Y, s=[15000/(self.side*self.side)]*len(X), marker='o', c=clist, edgecolors="white", linewidth=1)


        if labels:
            for k in range(self.grid.height):
                for i in range(self.grid.width):
                    if city_id[i][k] != "0":
                        plt.text(k + 0.5, i +0.5, city_id[i][k], va='center', ha='center')
                    else:
                        plt.text(k + 0.5, i + 0.5, (k,i), va='center', ha='center')




        plt.grid(color='w', linewidth= 10/self.side)

        for name_agent in traces:
            history = self.datacollector.get_agent_vars_dataframe().reset_index()
            type = history[history["name"] == name_agent]["type"].values[0]
            history_agent = history[history["name"] == name_agent]["pos"]
            history_agent = list(history_agent)
            for kpos in range(1,len(history_agent)):
                o = history_agent[kpos-1]
                d = history_agent[kpos]
                plt.quiver(o[0]+0.5, o[1]+0.5, d[0]-o[0], -d[1]+o[1],color=col_dict[type],
                           units='xy', scale=1, headwidth=5, headlength =7)
        if savepdf != None:

            plt.savefig('img/'+savepdf+'.pdf')

        plt.show()


class SchellingAgent(Agent):
    def __init__(self, unique_id, pos, model, atype, homophily, mobility , policy):

        super().__init__(unique_id, model)

        self.hex_id = hex(unique_id)[2:]
        self.pos = pos
        self.type = atype
        self.homophily = homophily

        self.mobility = mobility

        self.mobility_model = mobility["model"]

        if self.mobility_model != "classic":
            self.mobility_metric = mobility["metric"]
        else:
            self.mobility_metric = None

        self.policy = policy
        self.next_pos = (0,0)
        self.happy, self.segreg  = None, None


    def agent_stats(self):
        tot_neigh, sim_neigh = 0, 0
        neighbors = self.model.grid.iter_neighbors(self.pos, moore=True)
        for neighbor in neighbors:
            tot_neigh += 1
            if neighbor.type == self.type:
                sim_neigh += 1
        self.segreg = sim_neigh/tot_neigh if tot_neigh!=0 else 0
        self.happy = (tot_neigh==0) or (self.segreg >= self.homophily)

    def step(self):
        self.agent_stats()
        if not self.happy:
            self.next_pos = self.next_pos_func()
            self.model.grid.move_agent(self, self.next_pos)



    def next_pos_func(self):

        empties = list(self.model.grid.empties)
        empties_good = []

        for e in empties:
            tot_neigh, sim_neigh = 0, 0
            neighbors = self.model.grid.iter_neighbors(e, moore=True)
            for neighbor in neighbors:
                tot_neigh += 1
                if neighbor.type == self.type:
                    sim_neigh += 1
            if (tot_neigh == 0) or (sim_neigh/tot_neigh >= self.homophily):
                empties_good.append(e)

        if self.policy == "good":
            empties = empties_good

        self.model.cell_neighdens = {p: len([x for x in self.model.grid.iter_neighbors(p, moore=True)]) for p in empties}


        if self.mobility_metric == "relevance":
            score_dict = self.model.cell_relevance
        else:
            score_dict = self.model.cell_neighdens


        if self.mobility_model == "classic":
            empties_prob = [1/len(empties) for e in empties]

        elif self.mobility_model == "gravity":

            alpha = self.mobility["alpha"]
            beta = self.mobility["beta"]


            score_list = [score_dict[e]**alpha/(distance(e, self.pos, "moore")**beta) for e in empties]
            empties_prob = [score / sum(score_list) for score in score_list]


        elif self.mobility_model == "radiation":

            score_list = []

            if self.mobility_metric == "relevance":
                val_source = score_dict[self.pos]
            else:
                val_source = len([x for x in self.model.grid.iter_neighbors(self.pos, moore=True)])

            for e in empties:
                val_dest = score_dict[e]
                radius = distance(self.pos, e, "moore")
                if (self.pos, radius) not in self.model.radius_dict:
                    S = self.model.grid.iter_neighborhood(self.pos,
                                                        moore = True,
                                                        include_center= False,
                                                        radius=radius )
                    self.model.radius_dict[(self.pos, radius)] = S
                s_r = sum([score_dict[c] for c in self.model.radius_dict[(self.pos, radius)] if self.model.grid.get_cell_list_contents(c) == []])
                #print(self.pos, e, val_source, val_dest,radius, s_r)
                score_list.append(val_dest / ((val_source + s_r) * (val_source + val_dest + s_r)))
            empties_prob = [score / sum(score_list) for score in score_list]


        index_next_pos =  self.model.random.choices(range(len(empties)), empties_prob)[0]
        return empties[index_next_pos]

