import numpy

def nearest_neighbor(k):
    for i in range(k):
        return 0

def distance(x1, x2, y1, y2):
    return numpy.sqrt((x2-x1)^2+(y2-y1)^2)

def menu():
    print("Welcome to Dylan Vuong's Feature Selection Algorithm.")
    # inputFile = input("Type in the name of the file to test: ")
    inputFile = "CS170_Small_Data__20.txt"
    data = numpy.loadtxt(inputFile)
    print(data[:, 0]) # prints first column
menu()