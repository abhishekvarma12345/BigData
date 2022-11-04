import time
import sys
import math
import numpy as np

def readVectorsSeq(filename):
    with open(filename) as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result
    
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)

def ComputeObjective(inputPoints,solution,z):
    n = len(inputPoints)

    min_dist = []
    for x in inputPoints:
        min = sys.maxsize
        for s in solution:
            dxs = euclidean(x,s)
            if dxs < min:
                min = dxs
        min_dist.append(min)
    min_dist.sort()

    return max(min_dist[:len(min_dist)-z])

        


def SeqWeightedOutliers(inputPoints,weights,k,z,alpha):
    n = len(inputPoints)
    
    print("input size n = ",n)
    print("Number of centers k = ",k)
    print("Number of outliers z = ",z)
    r = sys.maxsize

    distances = np.zeros((n,n), dtype="float64")
    # computing distance between one point to every other point
    for i in range(n):
        for j in range(n):
            if i<j:
                distances[i,j] = euclidean(inputPoints[i],inputPoints[j])
            elif j<i:
                distances[i,j] = distances[j,i]

 
    for i in range(k+z+1):
        for j in range(k+z+1):
            if i<j:
                if distances[i,j] < r:
                    r = distances[i,j]

    r /=2
    print("Initial guess = ", r)
    
    guess = 1
    while(True):
        Z = inputPoints.copy()
        S = list()
        W = sum(weights)
        first_radius = (1+2*alpha)*r
        second_radius = (3+4*alpha)*r
        while(len(S) < k and W > 0):
            max = 0
            for x_idx in range(n):
                ball_weight = sum([weights[j] for j in range(len(Z)) if Z[j] != -1 and distances[x_idx,j] <= first_radius])
 
                if ball_weight > max:
                    max = ball_weight
                    new_center = x_idx
            
            S.append(inputPoints[new_center])
            
            for y in range(len(Z)): # look for points in Z
                if Z[y] != -1 and distances[new_center,y] <= second_radius:
                    W -= weights[y]
                    Z[y] = -1
            
        if W <= z:
            print("Final guess = ",r)
            print("Number of guesses = ",guess)
            return S
        else:
            r = 2*r
            guess += 1
    




if __name__ == "__main__":
    assert len(sys.argv) == 4

    Filename = sys.argv[1]

    assert sys.argv[2].isdigit() 
    k = int(sys.argv[2])
    assert sys.argv[3].isdigit()
    z = int(sys.argv[3])

    inputPoints = readVectorsSeq(Filename)
    weights = np.ones(len(inputPoints))
    start = time.time()
    solution = SeqWeightedOutliers(inputPoints,weights,k,z,0)
    end = time.time()
    objective = ComputeObjective(inputPoints,solution,z)

    print("Objective function = ",objective)
    print("Time of SeqWeightedOutliers = ",(end-start)*1000)



