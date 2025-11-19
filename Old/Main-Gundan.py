import pandas as pd
import numpy
from numpy import random as rd
extractor = pd.read_csv('mnist_train.csv')
Grid_Length_A = 28
Grid_Length_B = 28

def Squishification_Function_Sigmoid(z_L_j):
    a_L_j = 1/(1+numpy.exp(-z_L_j))
    return a_L_j

def Assigning_Input_Values(turn):
    a_L_j_k = []
    copy = extractor.iloc[turn]
    for i in range(len(copy),1):
        a_L_j_k.append(copy[i])
    label = copy[0]
    return_value = [label, a_L_j_k]
    return return_value

#for finding the least initial cost only.
def Weitghs_Generaion():
    w_L_j_k = [[],[],[]]
    for L in range(3):
        if L == 0:
          w_first_layer = []
          for i in range(784*16):
             w_first_layer.append(rd.randint(-10,10)*rd.rand())
          w_L_j_k[L] = w_first_layer
        elif L == 1:
          w_second_layer = []
          for i in range(16*16):
               w_second_layer.append(rd.randint(-10,10)*rd.rand())
          w_L_j_k[L] = w_second_layer
        elif L == 2:
          w_third_layer = []
          for i in range(16*10):
               w_third_layer.append(rd.randint(0,10)*rd.rand())
          w_L_j_k[L] = w_third_layer
    return w_L_j_k

#the most genuis stuff I've ever come up with:
def Importing_Parameters():
    w_L_j_k = [[],[],[]]
    with open("Parameters.txt", "r") as f:
        read = f.readlines()
        Range_Integer_Old = 0
        for L in range(3):
            Range = (784*16)/pow((784*16)**(L),4) + (L%2)*256 + (L//2)*160
            Range_Integer = int(Range)
            for i in range(Range_Integer):
                w_L_j_k[L].append(float(read[i + Range_Integer_Old].replace('\n','')))
            Range_Integer_Old = Range_Integer + Range_Integer_Old
    return w_L_j_k

def First_Bias_Assignment():
    bias_L_j = [[],[],[]]
    for L in range(3):
        BIAS_append = []
        for j in range(16 - (L//2)*6):
            BIAS_append.append(rd.randint(-10,10)*rd.rand())
        bias_L_j[L] = BIAS_append
    return bias_L_j

def Importing_Bias():
    bias_L_j = [[],[],[]]
    Bias_range_old = 0
    with open("bias.txt", "r") as f:
        read = f.readlines()
        for L in range(3):
            Bias_range = 16 - (L//2)*6
            for i in range(Bias_range):
                bias_L_j[L].append(float(read[i + Bias_range_old].replace("\n", '')))
            Bias_range_old = Bias_range_old + Bias_range
    return bias_L_j

def a_Values_and_cost(a_L_j_k, w_L_j_k, bias_L_j, label):
    returning_list = [[],[]]
    a_L_j_k_FOR_DERIVATIVE = [[],[],[],[]]
    for L in range(3):
        y_L = 16 - (L//2)*6
        a_L_j_k_new = []
        for j in range(y_L):
            sum = 0
            for k in range(len(a_L_j_k)):
                sum = sum + w_L_j_k[L][k+j*len(a_L_j_k)]*a_L_j_k[k]
            a_L_j_k_new.append(Squishification_Function_Sigmoid(sum + bias_L_j[L][j]))
        a_L_j_k_FOR_DERIVATIVE[L] = a_L_j_k
        a_L_j_k = a_L_j_k_new
    a_L_j_k_FOR_DERIVATIVE[3] = a_L_j_k
    true_output_values = []
    sum_cost_individ = 0
    for i in range(len(a_L_j_k)):
        if i == label:
            true_output_values.append(1)
        else:
            true_output_values.append(0)
        sum_cost_individ = sum_cost_individ + (a_L_j_k[i] - true_output_values[i])**2
    cost = sum_cost_individ/len(true_output_values)
    with open("cost.txt","w") as extractor2:
        extractor2.write(str(cost))
    for i in range(len(a_L_j_k)):
        if a_L_j_k[i] == max(a_L_j_k):
            print(i)
    print(a_L_j_k)
    weight_gradient = [[],[],[]]
    for L in range(3): #backpropogating for partial derivatives
        y_L = 16 - (L//2)*6
        for j in range(y_L):
            for k in range(len(a_L_j_k_FOR_DERIVATIVE[L])):
                
    returning_list[0] = w_L_j_k
    returning_list[1] = bias_L_j
    return(returning_list)

final_result = a_Values_and_cost(Assigning_Input_Values(0)[1], Importing_Parameters(), Importing_Bias(), Assigning_Input_Values(0)[0])

with open("Parameters.txt", 'w') as Write_Parameters:
    for i in final_result[0]:
        for j in range(len(i)):
            Write_Parameters.write(str(i[j]) + "\n")
Write_Parameters.close()

with open("bias.txt", "w") as Write_Bias:
    for i in final_result[1]:
        for j in range(len(i)):
            Write_Bias.write(str(i[j]) + "\n")
Write_Bias.close()