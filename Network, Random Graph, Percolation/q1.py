import matplotlib.pyplot as plt
import math
import numpy  as np
import sys

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

    @handleException
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
        elif isinstance(other,tuple):
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

if __name__ == "__main__":

    # test case 1
    print("Test case1: ")
    g = UndirectedGraph(10)
    g.addNode(4)

    # test case 2
    print("\nTest case2: ")
    g = UndirectedGraph(10)
    g.addNode(11)

    # test case 3
    print("\nTest case3: ")
    g = UndirectedGraph(5)
    g = g + (1, 2)
    g = g + (3, 4)
    g = g + (1, 4)
    print(g)

    # test case 4
    print("\nTest case4: ")
    g = UndirectedGraph(5)
    g = g + (1, 2)
    g = g + (3, 4)
    g = g + (1, 4)
    g.plotDegDist()

    # test case 5
    print("\nTest case5: ")
    g = UndirectedGraph()
    g = g + 100
    g = g + (1, 2)
    g = g + (1, 100)
    g = g + (100, 3)
    g = g + 20
    g.plotDegDist()