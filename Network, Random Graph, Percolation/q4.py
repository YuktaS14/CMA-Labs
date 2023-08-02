import matplotlib.pyplot as plt
import math
import numpy  as np
import random


"""
    Using function decorators to handle exceptions.
"""
def handleException(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            print(type(e))
            print(e)
            return

            # Uncomment following line if program should exit after an exception occurs
            # sys.exit(1);
    return wrapper

class UndirectedGraph():
    def __init__(self,vertices=None):
        #creating dictionary for adjacency list of nodes
        self.adjacencyList = {}
        self.edges=0;
        # if vertices are not defined while creating graph(object of this class),
        # the graph is considered as free graph, hence we maintain a flag for that
        if(vertices == None):
            self.isFree=1;
            self.vertices=0;
        else:
            #if number of vertices is specified, we pecify the vertices variable and correspondingly create adjacent list for vertices
            self.vertices=vertices;
            self.isFree=0;
            for i in range(1,self.vertices+1):
                self.adjacencyList[i]=[];  

    def addNode(self,v):
        #if graph is not a free graph, then we cannot add nodes with value greater than total number of vertices in it
        if(self.isFree == 0 and v > self.vertices):
            raise Exception("Node index cannot exceed number of nodes");
        if(not isinstance(v,int)):
            raise Exception("Invalid type of node given");
        if(v<=0):
            raise Exception("Node can have only positive values");
        
        # else if graph is free or node to be added is less than graph's vertices
        # then we add it to corresponding adjacency list dictionary, and initialize empty adjList for that node
        if(v not in self.adjacencyList.keys()):
            self.adjacencyList[v]=[];
            self.vertices+=1;
    
    def addEdge(self,node1,node2):
        # if node1 not present in graph, we add it
        if(node1 not in self.adjacencyList.keys()):
            self.addNode(node1);
        # if node2 not present in graph, we add it
        if(node2 not in self.adjacencyList.keys()):
            self.addNode(node2);
        # if edge already exist -> return
        if(node2 in self.adjacencyList[node1]):
            return
        # finally if both nodes are not same and edge between them do not already exist, we add it to graph
        if(node1 != node2):
            self.adjacencyList[node1].append(node2);
            self.adjacencyList[node2].append(node1);
            #incrementing count of edges
            self.edges+=1;
                
    #overloading '+' operator to add nodes and edges to graph.
    def __add__(self,other):
        #if integer is added with object, consider it as addNode of value equal to that integer into the graph
        if isinstance(other,int):
            self.addNode(other)
        #else if tuple is given, it is considered as adding edge to the given graph object
        else:
            v1=other[0];v2=other[1];
            self.addEdge(v1,v2);
        return self;
    
    #overloading print to print the object of this class
    def __str__(self):
        st = "Graph with {} nodes and {} edges. Neighbours of the nodes are belows:\n".format(self.vertices,self.edges);
        for key in self.adjacencyList.keys():
            # printing adjacency list for each node of the graph
            ls=self.adjacencyList[key];
            st += "Node {}: {}\n".format(key,ls);
        return st;

    # method for plotting degree distribution of the graph
    def plotDegDist(self):
        # counting the number of neighbours (edges) each node has
        # maintaing a dictionary to store number of nodes with specific degree (degree is the key and number of nodes with corresponding degree is the value for it)
        nodeFraction={};
        for value in self.adjacencyList.values():
            # len of adjacency list for a node will give the degree of that node
            fract= len(value);
            # incrementing the count of particular degree in dictionary if already exist, else adding the degree to dictionary
            if fract in nodeFraction.keys():
                nodeFraction[fract]+=1;
            else:
                nodeFraction[fract]=1;
        
        # method to plot the degree distribution
        self.plotGraph(nodeFraction);

    def plotGraph(self,nodeFraction):
        #
        x = np.linspace(0,self.vertices-1,self.vertices);
        y = [0]*self.vertices; 
        for key in nodeFraction.keys():
            # taking fraction of nodes having degree 'key'
            y[key]=nodeFraction[key]/self.vertices;
        # xAvg is the average node degree
        xAvg = [float((self.edges*2)/self.vertices)];

        #plotting the graph for node degree distribution
        plt.plot(x,y,'bo');
        plt.axvline(xAvg,color='r',label = "Avg. node degree");
        plt.grid()
        plt.xlabel('Node degree'); plt.ylabel('Fraction of nodes');
        plt.title('Node Degree Distribution');
        plt.legend(['Actual degree distribution','Avg. node degree'])
        plt.show()

    def isConnected(self):
        # visited array to keep track if the node is visited earlier
        self.visited = [0] * (self.vertices+1);

        # we start bfs from node 1
        self.bfs(1);

        # if any vertex is not visted even after coming out of bfs, it means that the corresponding vertex forms another connected component
        for node in range (2,self.vertices+1):
            #hence if any such vertex exist we return False else True
            if not self.visited[node]:
                return False;
        return True;


    def bfs(self,source):
        # implementing queue for bfs
        queue = [];
        queue.append(source);
        self.visited[source] =1
        while len(queue) != 0:
            # pop the queue to get the next vertex which is visited.
            source = queue.pop(0);

            #incrementing the count for number of vertices in the current connected component
            self.count+=1;

            # for each neighbour of source, if it is not yet visited we add it to the queue
            for nbr in self.adjacencyList[source]:
                if not self.visited[nbr]:
                    queue.append(nbr);
                    # update visited value of neighbour to 1
                    self.visited[nbr] = 1;

    
    def oneTwoComponentSizes(self):
        # visited array to keep track if the node is visited earlier
        self.visited = [0] * (self.vertices+1);
        # storing the sizes of the components found in the graph while doing bfs
        compSizes=[];
        # we start bfs from node 1
        # then iterate over all vertices to find if it is alrady visited.
        # if it is not, we start bfs again from that vertex, store the earlier count and update the count to 0 for the new connected component
        for node in range (1,self.vertices+1):
            if not self.visited[node]:
                self.count = 0;
                self.bfs(node);
                # once we come out of bfs, we have visited over all nodes of one connected component, so we add the count value(size of the component) in the list compSizes
                compSizes.append(self.count);
        
        # sort the list in desc order to get sizes of largest 2 components
        compSizes.sort(reverse=True);
        if len(compSizes)>=2:
            largest2 = [compSizes[0],compSizes[1]];
        else:
            #if only one component exist, size of 2nd largest is kept 0;
            largest2 = [compSizes[0],0];
        # returning the sizes of largest 2 components in the graph
        return largest2;
        

        

class ERRandomGraph(UndirectedGraph):
    # function to create a sample of a random graph in which the probability with which an edge exist is given by 'prob'
    def sample(self,prob):
        # in order to create new sample random graph we remove all the edges the earlier graph had, and start over again
        for k in self.adjacencyList.keys():
            self.adjacencyList[k] = [];
        
        # if randomly generated value is less than prob it means that edge exist between i and j th node
        # so we add corresponding edge, if not we take next pair of nodes and continue;
        for i in range(1,self.vertices+1):
            for j in range(i+1,self.vertices+1):
                isEdge = random.random();
                if(isEdge<prob):
                    self.addEdge(i,j);
    


def verifyCompSize():
    print("\n If p < 0.001, the Erd ̋os-R ́enyi random graph G(1000, p) will almost surely have only small connected components.\n On the other hand, if p > 0.001, almost surely, there will be a single giant component containing a positive fraction of the vertices. \n As can be visualized in graph:\n ");
    # the statement is to verify for the graph G(1000,p)
    g = ERRandomGraph(1000)

    # 50 probabilities are taken and is stored in x
    x = list(np.linspace(0,0.01,50));

    # probability threshold for connected component sizes and number
    cc_threshold = 0.001;

    # probability threshold for graph connectedness
    connectedness_threshold =  math.log(1000)/1000;

    noOfRuns = 50;
    yL  = [];
    y2L = [];

    # for a probability in x, we construct a sample random graph noOfRuns(=50) times
    # Then find the fraction of nodes in largest and 2nd largest component each time

    for i in x:
        # countL stores fraction of nodes in largest component
        # count2L stores fraction of nodes in 2nd largest component
        countL = 0;
        count2L = 0;
        for j in range(1,noOfRuns+1):
            g.sample(i);
            sizes = g.oneTwoComponentSizes();
            # print(sizes)
            countL += float(sizes[0]/1000);
            count2L += float(sizes[1]/1000);

        # appending average of fraction of nodes in largest as well as 2nd largest component after 50 runs 
        yL.append(countL/noOfRuns);
        y2L.append(count2L/noOfRuns);
    
    # plotting the graph for verifying the statement
    plt.plot(x,yL,'g', label="Largest Connected Component");
    plt.plot(x,y2L,'b', label="2nd largest Connected Component");
    # plotting threshold values as vertical lines
    plt.axvline(connectedness_threshold,color='y',label = "Connectedness threshold");
    plt.axvline(cc_threshold,color='r',label = "Largest CC size threshold");
    
    plt.xlim(0,0.01);
    plt.grid();
    plt.title("Fraction of nodes in the largest and second-largest \nconnected components (CC) of G(1000, p) as function of p");
    plt.xlabel("p");
    plt.ylabel("fraction of nodes");
    plt.legend();
    plt.show();




if __name__ == "__main__":
    # test case 1
    print("For test case 1: ")
    g = UndirectedGraph(6)
    g = g + (1, 2)
    g = g + (3, 4)
    g = g + (6, 4)
    print(g.oneTwoComponentSizes())

    # test case 2
    print("\nFor test case 2: ")
    g = ERRandomGraph(100)
    g.sample(0.01)
    print(g.oneTwoComponentSizes())

    # test Case 3
    print("\nTest Case 3: ")
    verifyCompSize();