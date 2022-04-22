class Adder:
    # Constructor
    def __init__(self):
        self.sum = 0

    # Add up 2 element to the Adder
    def add2(self,x,y):
        print("x*y adding via python")
        self.sum += x*y


    # Print the total sum
    def printSum(self):
        print("the sum via the Python class is ",self.sum)
