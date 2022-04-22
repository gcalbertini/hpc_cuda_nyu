class Adder:
    # Constructor
    def __init__(self):
        self.sum = 0

    # Add an element to the Adder
    def add(self,x):
        print("adding via python ", x)
        self.sum += x


    # Print the total sum
    def printSum(self):
        print("the sum via the Python class is ",self.sum)
