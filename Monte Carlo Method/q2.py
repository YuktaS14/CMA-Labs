import matplotlib.pyplot as plt
import math
import numpy  as np
import random


class Dice():
    def __init__(self,numSides=6):
        try:

            # if numOf sides of dice is less than 4 or decimal exception is raised
            if(numSides<4 or (type(numSides) != int) ):
                raise Exception;
            else:

                # if specified number of sides of the dice, by default equal probabilities 
                # is given to each face of the dice, which can be modified later
                self.numSides = numSides;
                self.probs = tuple(np.full(self.numSides, float(1/self.numSides)));

                # ls stores the partition of the interval (0,1) according to the probability of each face
                self.ls = [];
        except Exception:
            print("Cannot construct the dice");
            
            
    
    def setProb(self,prob=None):

        # if probilities are specified (by default every face has equal probabilty)
        # we check if sum of all probabilities is 1 and whether probability of each face is given
        if(prob != None):
            sumi = sum(prob);
            try:

                # here round function is used to get precision till 4 in oreder to prevent the inaccuracy resulting from 
                # operations between floating point numbers, since these are stored and handled differently in computer.
                if(round(sumi,4)!= 1.0 or (len(prob) != self.numSides)):
                    print(len(prob)," ",sumi)
                    raise Exception;
                else:

                    #self.probs is updated with new probabilities
                    self.probs = prob;
                    print(self.probs)

            except Exception:
                print('Invalid probability distribution', self.probs);
                return;
    
    #overloading print statement
    def __str__(self):
        return "Dice with {} faces and probability distribution {}".format(self.numSides,self.probs);
    
    #function to get the partition(and thus the face of the dice) corresponding to a randomly chosen value between 0 and 1;
    def findPart(self,val):
        for i in range (0,len(self.ls)):
            if self.ls[i] >= val:
                return i+1;
    

    def roll(self,num):
        # expected times a face turns up would be just equal to number of times dice is thrown * probability of that face to be turned up
        expected = list(map(lambda x: x*num,self.probs));
        self.ls = [self.probs[0]];
        for i in range(1,len(self.probs)):
            (self.ls).append((self.ls[i-1] + self.probs[i]));
        
        print(self.ls);
        # numTimes ith index stores the number of times ith vertex turn up when num times dixe is thrown (randomly)
        self.numTimes =[0]*len(self.ls);

        # we throw the dice num times, each time a random value in the interval(0,1) is chosen.
        # probability that it lies in any partition depends on the width of the partition( since we conider  Uniform distribution of random variables)
        # hence we find the partition and thus corresponding face value and increment number of occurences of that face value.
        
        for i in range (1,num+1):
            val = random.random();
            partition = self.findPart(val);
            self.numTimes[partition-1]= self.numTimes[partition-1]+1;
    
        # plotting barchart for expected and actual number of occurrences of each face when the dice is thrown n times.
        barWidth = 0.25;
        br1 = np.linspace(1,self.numSides,self.numSides);
        br2 = [x+barWidth for x in br1];
 
        plt.bar(br1,self.numTimes,color='b',width=barWidth,edgecolor='grey',label='Actual');
        plt.bar(br2,expected,color='r',width=barWidth,edgecolor='grey',label='Expected');
        plt.xticks([r+barWidth/2 for r in range(1,len(br1)+1)],br1);

        plt.title("Outcome of {} throws of a {}-faced dice".format(num,self.numSides));
        plt.xlabel('Sides');
        plt.ylabel('Occurences');
        
        plt.legend();
        plt.show();


if __name__ == "__main__":
    # making object of the class and plotting the actual-expected occurences roll graph for it
    d = Dice(5);
    d.setProb((0.2,0.1,0.4,0.2,0.1));
    print(d);
    d.roll(1000);