import math

#import numpy as np
class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 8
    self.outputSize = 1
    self.hiddenSize = 8

    #weights (get them from keras model)
    self.W1 = [[ 2.0315645 ,  1.8409909 , -0.33502752, -0.18639019, -0.96386826,
        -0.11131245, -0.5236887 , -0.30002964],
       [-0.49199048, -0.24202986, -0.14692506, -0.2556005 ,  0.29572394,
        -0.4300866 ,  0.11291618,  0.58880913],
       [-0.1155514 , -0.38233158, -0.42540902,  0.46698067,  0.31963027,
        -0.10048038,  0.20791264, -0.08006934],
       [-0.08894206,  0.2367364 , -0.02009247,  0.05370482,  0.00876237,
         0.19531894,  0.41058493,  0.40323824],
       [-0.40059203,  0.19815329, -0.16533062, -0.09545176, -0.5597245 ,
        -0.13986266, -0.66129607, -0.17688236],
       [ 0.15824443, -0.1294241 , -0.1975826 ,  0.17408755,  0.2516969 ,
        -0.52434146,  0.15168232,  0.4617063 ],
       [ 0.85206074,  0.28743362, -0.23727593, -0.9338444 , -0.26853472,
         0.11744624,  0.16852175, -0.16940284],
       [-0.04717655,  0.55974305, -0.23455885, -0.25749975, -0.11437127,
        -0.22862965, -0.5393369 ,  0.16032581]]
    self.W2 = [[ 2.576184  ],
       [ 0.7918516 ],
       [ 0.5252627 ],
       [-1.9406364 ],
       [-0.63452315],
       [ 0.07101798],
       [ 1.9215938 ],
       [-0.50750875]]
    self.B1 = [-0.8985953 , -1.2108401 , -0.09709592,  1.6016883 ,  1.3206937 ,
        0 , -0.8971717 ,  0.04050966]
    self.B2 = [-0.08436321]
    
    
    
  def forward(self, X):
    #forward propagation through our network
    self.z = self.MM(X, self.W1)
    self.z = self.sum(self.z , self.B1)
    self.z2 = list(map(self.modif_sigmoid, self.z))
    self.z3 = self.MM(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    self.z3 = self.sum(self.z3 , self.B2)
    o = list(map(self.modif_sigmoid, self.z3)) # final activation function

    return o

  def sigmoid(self, s):
    
    # activation function
    return 1/(1+math.exp(-s))
  def modif_sigmoid(self, s):
    result = []
    for i in s:
      result.append(1/(1+math.exp(-i)))
    return result
      
  def MM(self , a,b):
    c = []
    for i in range(0,len(a)):
        temp=[]
        for j in range(0,len(b[0])):
            s = 0
            for k in range(0,len(a[0])):
                s += a[i][k]*b[k][j]
            temp.append(s)
        c.append(temp)
    return c
  def sum( self, matrix_a, matrix_b ) :
    res =[]
    print(len(matrix_a))
    print(len(matrix_b))
    for i in matrix_a:
      comp = []
      for j in range(len(i)):
        app_var = i[j] + matrix_b[j]
        comp.append(app_var)
      res.append(comp)
    return res
NN = Neural_Network()
