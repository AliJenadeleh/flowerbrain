import numpy as np
import math

def Sigmoid(b):
    _b = b * -1
    return 1 / (1 + math.exp(_b))

def ShowColor(b):
    if b >= .5:
        print("Red")
    else:
        print("Blue")

class brain:
    def __init__(self):
        self.bi = np.random.randn()
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.TrainLoop = 50000
        self.alpha = 0.2
        self.length = 0
        self.unknow = (1,4.5)
        self.data = [(1.5,3,1),(1,2,0),(1.5,4,1),(1,3,0),(0.5,3.5,1),(0.5,2,0),(1,5.5,1),(1,1,0)]
        self.length = len(self.data)
    
    def training(self):
        print("Training ....")
        for i in range(self.TrainLoop):
            inx = np.random.choice(range(self.length))
            item = self.data[inx]
            
            z = (item[0] * self.w1) + (item[1] * self.w2) + self.bi
            pred = Sigmoid(z)
            cost = (pred - item[2]) ** 2
            dcost_pred = 2 * (pred - item[2])
            dpred_dz = Sigmoid(z) * (1 - Sigmoid(z))
            dz_dw1 = item[0]
            dz_dw2 = item[1]
            dz_dbi = 1
            dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
            dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
            dcost_dbi = dcost_pred * dpred_dz * dz_dbi
            self.w1 -= self.alpha * dcost_dw1
            self.w2 -= self.alpha * dcost_dw2
            self.bi -= self.alpha * dcost_dbi
        print("Training .... End")

    def Suggest(self,width,length):
        z = (self.w1 * width) + (self.w2 * length) + self.bi
        pred = Sigmoid(z)
        print("Width",width,"Length",length)
        ShowColor(pred)

    def SuggestUnknow(self):
        z = (self.w1 * self.unknow[0]) + (self.w2 * self.unknow[1]) + self.bi
        pred = Sigmoid(z)
        print("Width",self.unknow[0],"Length",self.unknow[1])
        ShowColor(pred)

    def ShowDataSuggestion(self):
        for i in self.data:
            z = (self.w1 * i[0]) + (self.w2 * i[1]) + self.bi
            pred = Sigmoid(z)
            print("Width",i[0],"Length",i[1])
            ShowColor(pred)
            print("in real")
            ShowColor(i[2])

