f = open("Parameters.txt", "r")
read = f.readlines()
print(read[0])

def Importing_Parameters():
    w_L_j_k = [[],[],[]]
    with open("Parameters.txt", "r") as f:
        read = f.readlines()
        Range_Integer_Old = 0
        for L in range(3):
            Range = (784*16)/pow((784*16)**(L),4) + (L%2)*256 + (L//2)*160
            Range_Integer = int(Range)
            for i in range(Range_Integer):
                w_L_j_k[L].append(read[i + Range_Integer_Old*L].replace('\n',''))
            Range_Integer_Old = Range_Integer
    return w_L_j_k

def Importing_Bias():
    with open("bias.txt", "r") as f:
        read = f.read()

def aboba():
    a = []