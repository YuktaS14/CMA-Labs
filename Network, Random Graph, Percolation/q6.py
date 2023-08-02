import matplotlib.pyplot as plt
import numpy  as np
import random
import networkx as nx
import queue

class Lattice():
    def __init__(self,n):
        # creating a n x n 2d grid graph 
        self.n = n;
        self.L = nx.grid_2d_graph(n,n);
        # the edges of graph are stored in the form of adjacency list.
        self.adjacencyList = {}
        # co ordinates of grid are set up using pos
        self.pos = {(x,y):(y,-x) for x,y in self.L.nodes()}
        # initializing adjacencyList for each node (x,y)
        for x,y in self.L.nodes():
            self.adjacencyList[(x,y)] = []   

    """
        Method that returns the list of all edges got from the adjacency list of all nodes
    """
    def getList(self):
        edgeList = [];
        # for each node in the nodes of graph, we find all the edges associated with that node and add it to edgeList
        for n1 in self.adjacencyList.keys():
            ls = self.adjacencyList[n1];
            for n2 in ls:
                # if the edge already exist we dont add it again
                if((n1,n2) not in edgeList and (n2,n1) not in edgeList):
                    edgeList.append((n1,n2));
        return edgeList;
                
    """
        Method to draw the nodes and the edges for the graph based on the number of nodes, pos and edge list given as arguments to draw
    """
    def show(self,plot = True):
        nx.draw_networkx_nodes(self.L,pos=self.pos,node_size=3,linewidths=0.2,node_color='r');
        nx.draw_networkx_edges(self.L,pos=self.pos,edgelist = self.getList(),edge_color="r");
        # after drawing, to plot the graph we use matplotlib library
        plt.show()
        
    """
        Method to visualize bond percolation, given the probability with which edge can exist
    """
    def percolate(self,p):

        # removing all existing edges in the graph and percolating newly with probability p
        for key in self.adjacencyList.keys():
            self.adjacencyList[key] = [];

        # while adding edge we only add to the right and the down node of the curr node, starting from (0,0) node. 
        # This way we wont add or consider the same edge 2 times (with top, with left- as that edge would already be considered before)

        def EdgeDown(x,y):
            # if random value is less than p then edge exist.
            if random.random()<p:
                # update the adjacency lists of both the nodes
                self.adjacencyList[(x,y)].append((x+1,y))
                self.adjacencyList[(x+1,y)].append((x,y))

        def EdgeRight(x,y):
            # if random value is less than p then edge exist.
            if random.random()<p:
                # update the adjacency lists of both the nodes
                self.adjacencyList[(x,y)].append((x,y+1))
                self.adjacencyList[(x,y+1)].append((x,y))

        # for each node (x,y) we check:  
        for x in range(self.n):
            for y in range(self.n):
                # if it is not boundary node edge can be added to the right as well to the node below it
                if x<self.n-1 and y<self.n-1:
                    EdgeDown(x,y)
                    EdgeRight(x,y)
                # if the curent node (x,y) is in the last column then we can add edge to only the node below it and not to the right
                elif x< self.n-1:
                    EdgeDown(x,y)
                # if the curent node (x,y) is in the last row then we can add edge to only the node right of it and not to the node below it
                elif y<self.n -1:
                    EdgeRight(x,y)
    
    """
        Given the farthest node and the dictionary of parents we find the shortest path from source to the farthest node by backtracking
    """
    def getPath(self,parent,farthestNode):
        path = []
        curr=farthestNode;
        # parent node of source is -1, so we iterate till we reach the source
        while(parent[curr]!= -1):
            path.append((curr,parent[curr]));
            curr=parent[curr];
        #return the path found (the list of edges in the path)
        return path


    
    def bfs(self, source,flag = 0):
        # implementing bfs using queue
        # flag has values: 0 -> bfs only checks if a path exist from topToDown, and 1 -> bfs returns all possible paths from start to farthest node possible in the lattice 
      
        q = queue.Queue()
        # adding the start node with dist = 0 into the queue
        q.put((source,0));
        # visited set is used to keep track of whether the node is already visited or not
        visited = set();
        #parent dictionary stores the parent for each node visited, parent of source node is kept -1
        parent={};
        visited.add(source);
        parent[source]=-1;

        #initializing few flags which would be updated later
        # farthest node and maxDist are used to get the farthest node in the shortest possible path
        maxDist=0;
        farthestNode=source;
        isPercolating = False;

        while not q.empty():
            # pop from the queue
            node,dist = q.get();
            # if the node popped is farthest among all those found till now, we update the farthestNode and the distance accordingly
            if dist > maxDist:
                farthestNode = node
                maxDist=dist;
            
            # for each neighbour of current node, if it is not already visited, we add it to the set
            for nbr in self.adjacencyList[node]:
                if nbr not in visited:
                    visited.add(nbr);
                    # Add the neighbour node to the queue if not already visited, with distance = 1+ distance of current node (dist)
                    q.put((nbr,dist+1));
                    # add the neighbour and its parent to the dictionary
                    parent[nbr]=node;

                    # if we reach at the last row, the latice percolates, we update the flag
                    if nbr[0] == self.n-1:
                        isPercolating=True;

                        #if we only had to check whether path exists or not, we return true here
                        if(flag == 0):
                            return True;

                        # if flag == 1 and isPercolating is true, we have found the the shortest path from top to down, so we break from the loop
                        farthestNode = nbr;
                        break;
            # if for the current node topToDown path exists then we stop iterating over other nodes of queue too, since we have found the shortes path connecting
            if(isPercolating):
                break;
        # if flag is 0, here it means that isPercolating is still false, even after complete bfs, then we return false
        if(flag == 0):
            return False
        #if flag is 1, we have to return the path from start to the farthest node possible
        return self.getPath(parent,farthestNode);

        
        
    def existsTopDownPath(self):
        # starting from every node in the top layer we call bfs and if toptoDown path exist we return true
        for c in range(self.n):
            if(self.bfs((0,c))):
                return True;
        # if none of the bfs returns true, no path exists, hence return false
        return False;

    """
        Method that implements:
          For every node in top-most layer:
          – If there is no path from u to nodes in the bottom-most layer, display the largest shortest path that originates at u.
          – Otherwise, display the shortest path from u to the bottom-most layer.
    """
    def showPaths(self):
        for i in range(self.n):
            # for each node in the topmost layer the edge list of the path from the (0,i) node to the farthest possible node is retrieved
            path = self.bfs((0,i),1);    
            # the path is then given as parameter inorder to draw those edges in our lattice   
            nx.draw_networkx_edges(self.L,pos=self.pos,edgelist = path,edge_color="g")
        # plotting the lattice
        self.show()
            

"""
    Function to verify the statement:
        “A path exists (almost surely) from the top-most layer to the bottom-most layer of a 100 × 100
grid graph only if the bond percolation probability exceeds 0.5”
"""

def verifypathExist():
    # for a 100 x 100 grid graph
    l = Lattice(100);
    # taking 50 points(probabilities) from 0 to 1
    x = list(np.linspace(0,1,50));
    # total number of runs = 50
    runs = 50;

    # corresponding to each probability the fraction of graphs among 50 runs in which the topToDownPath exist is store in list y
    y = []

    # for each probability in x we percolate on the graph 50 times and get the average number of times the path from top to down exists
    for p in x:
            count = 0;
            for j in range(runs):
                l.percolate(p)
                # if top to down path exist increment the count
                if(l.existsTopDownPath()):
                    count+=1;
            # corresponding to each probability we store the fraction of runs for which the path exist
            y.append(count/runs);
    
    # plotting the graph to verify the statement
    plt.title("Critical cut-off in 2-D bond percolation")
    plt.xlabel("p");
    plt.ylabel("Fraction of runs top-to-down-path exists")
    plt.plot(x,y,'b')
    plt.show()

if __name__ == "__main__":
    
    verifypathExist()
