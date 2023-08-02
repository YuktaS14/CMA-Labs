import random

class TextGenerator():

    # function to parse the given text file, it creates a dictionary which has 2 word tuple as a key and list of words 
    # which immediately follow this tuple in the text as its value

    def parseText(self,text):
        #dict is the associated dictionary
        self.dict ={};
        i=0;

        # splitting text into words, ls contains all words in the text
        self.ls = text.split();
        while (i+2) < (len(self.ls)):
            key = self.ls[i] + " "+ self.ls[i+1];

            # if key is already present in dictionary append the word to its value list
            if(key in self.dict.keys()):
                self.dict[key].append(self.ls[i+2]);

            #else create a list and add the word found
            else:
                self.dict[key]=[(self.ls[i+2])];
            i=i+1;
        # return self.dict;

    # function which reads the file and parses it through parseText method
    def assimilateText(self,file):
        with open('sherlock.txt') as file:
            file_contents = file.read();
            # removing '\n' characters from text file
            file_contents.split('\n');
            file_contents.join("");
            self.parseText(file_contents);

    # function to generate text from given file
    def generateText(self,num,word=None):

        #if word is specified, it would be the starting word of generated text (genText)
        start=word;

        #if not specified, we randomly chose a word from the text file
        if(word==None):
            wNum=random.randint(0,len(self.ls));
            word=self.ls[wNum];
            # print(word)

        try:
            # flag = 0;
            # creating a list of keypairs starting with given start word and then randomly chosing one pair out of it.
            keyPairs=[]
            for key in self.dict.keys():
                k=key.split();
                if word in k[0]:
                    # flag = 1;
                    keyPairs.append(key);

            # start now has randomly chosen key pair (2 word tuple) starting with 1stword we got ( the specified or randomly chosen one)
            start = keyPairs[random.randint(0,len(keyPairs)-1)];

            # if no pair with the given word as starting word is found, exception is raised.           
            if(len(keyPairs) == 0):
                raise Exception;
        except Exception:
            print("Unable to produce text with the specified start word.");
            return;

        # 2 word tuple is the start of the text
        genText= start;       
        i=2;
        
        # w is the 2nd component of the tuple for which consequent word is to be found
        key = start;
        
        while (i<num):

            # if the key (2 word tuple) is present in dictionary, next word is randomly chosen from its value list.
            if key in self.dict.keys():
                val = self.dict[key];
                vNum = random.randint(0,len(val)-1);
                next = val[vNum];
                # next tuple is formed and it would our key now
                key = key.split()[1]+" "+ next;
            else:
                #randomly next word is chosen from the text
                #flag is used to keep note if we are able to form next key-word
                flag=0;
                while(not flag):
                    next = self.ls[random.randint(0,len(self.ls)-1)];

                    # a key pair starting with that word(next) is found from the dictionary;
                    # keyPairs store all the keys starting with word next
                    keyPairs=[];
                    # finding if any pair starts with word next;
                    for key in self.dict.keys():
                        k = key.split();
                        if next in k[0]:
                            keyPairs.append(key);

                    # if a pairs are found, we select a random pair from it, and flag is updated
                    if (len(keyPairs) != 0):
                        key = keyPairs[random.randint(0,len(keyPairs)-1)];
                        flag =1;       

                    # if no key is found we go back to loop and find another random word from text            
           
            # add the next word to our genText and update the count
            genText += " " + next;
            i=i+1;
        print(genText);

if __name__ == "__main__":
    # creating object of TextGenerator class to generate text from 'Sherlock.txt' file 
    t1=TextGenerator();
    t1.assimilateText('sherlock.txt');
    t1.generateText(100,'London');