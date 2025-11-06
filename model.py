from torch_geometric.nn import Linear,GCNConv,SGConv,GATConv,GATv2Conv,SuperGATConv,FAConv,EGConv,GCN2Conv,SSGConv,GENConv,GeneralConv
import torch_geometric
import torch
import torch.nn as nn
from torch_geometric.nn.models import GAT,GIN,LightGCN,GCN

def mynorm(tensor):
    # 计算每一行的最小值和最大值
    min_values = tensor.min(dim=1, keepdim=True)[0]
    max_values = tensor.max(dim=1, keepdim=True)[0]
    # 进行最小-最大归一化
#     normalized_tensor = (tensor - min_values) / (max_values - min_values + 1e-8)               ### 0-1
    normalized_tensor = 2 * (tensor - min_values) / (max_values - min_values + 1e-8) - 1       ### -1-1
    return normalized_tensor



class SDGCN4(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv1 = GCNConv(32,32)
        self.conv2 = GCNConv(32,32)
        self.conv3 = GCNConv(32,32)
        self.conv4 = nn.Linear(32*4,out_channels)
    
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()    
    
    
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0= self.fc1(x).relu()
        x0=mynorm(x0)
        
        x1 = self.conv1(x0, edge_index)         ## 1hop       inf
        xx1= x1
        x2 = self.conv2(x1, edge_index)         ## 0,2hop       inf
        xx2=mynorm(x2)- mynorm(x0)
        x3 = self.conv3(x2, edge_index)          ## 1,3hop     inf
        xx3=mynorm(x3)-mynorm(x1)
        
        xx4=torch.cat((x0,xx1,xx2,xx3),dim=1)
        x4 = self.conv4(xx4) 
        return x4  


class SDGCN8(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv1 = GCNConv(32,32)
        self.conv2 = GCNConv(32,32)
        self.conv3 = GCNConv(32,32)
        self.conv4 = GCNConv(32,32)
        self.conv5 = GCNConv(32,32)
        self.conv6 = GCNConv(32,32)
        self.conv7 = GCNConv(32,32)
        self.conv8 = nn.Linear(32*8,out_channels)
    
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()    
    
    
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0= self.fc1(x).relu()
        x0=mynorm(x0)
        
        x1 = self.conv1(x0, edge_index)         ## 1hop       inf
        xx1= x1
        x2 = self.conv2(x1, edge_index)         ## 0,2hop       inf
        xx2=mynorm(x2)- mynorm(x0)
        x3 = self.conv3(x2, edge_index)          ## 1,3hop     inf
        xx3=mynorm(x3)-mynorm(x1)
        
        x4 = self.conv4(x3, edge_index)         ## 0,2,4hop   inf
        xx4=mynorm(x4)-mynorm(x2) 
        
        x5 = self.conv5(x4, edge_index)          ## 1,3,5hop   inf
        xx5=mynorm(x5)-mynorm(x3)
        
        x6 = self.conv6(x5, edge_index)         ## 0,2,4,6hop inf
        xx6=mynorm(x6)-mynorm(x4)
        
        x7 = self.conv7(x6, edge_index)          ## 1,3,5,7 hop   inf
        xx7=mynorm(x7)-mynorm(x5)
        
        xx8=torch.cat((x0,xx1,xx2,xx3,xx4,xx5,xx6,xx7),dim=1)
        x8 = self.conv8(xx8) 
        return x8  


class SDGCN16(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv1 = GCNConv(32,32)
        self.conv2 = GCNConv(32,32)
        self.conv3 = GCNConv(32,32)
        self.conv4 = GCNConv(32,32)
        self.conv5 = GCNConv(32,32)
        self.conv6 = GCNConv(32,32)
        self.conv7 = GCNConv(32,32)
        self.conv8 = GCNConv(32,32)
        
        self.conv9 = GCNConv(32,32)
        self.conv10 = GCNConv(32,32)
        self.conv11 = GCNConv(32,32)
        self.conv12 = GCNConv(32,32)
        self.conv13 = GCNConv(32,32)
        self.conv14 = GCNConv(32,32)
        self.conv15 = GCNConv(32,32)        
#         self.conv7 = GCNConv(16*6,out_channels)
        self.conv16 = nn.Linear(32*16,out_channels)
    
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()    
    
    
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0= self.fc1(x).relu()
        x0=mynorm(x0)
        
        x1 = self.conv1(x0, edge_index)         ## 1hop       inf
        xx1= x1
        x2 = self.conv2(x1, edge_index)         ## 0,2hop       inf
        xx2=mynorm(x2)- mynorm(x0)
        x3 = self.conv3(x2, edge_index)          ## 1,3hop     inf
        xx3=mynorm(x3)-mynorm(x1)
        
        x4 = self.conv4(x3, edge_index)         ## 0,2,4hop   inf
        xx4=mynorm(x4)-mynorm(x2) 
        
        x5 = self.conv5(x4, edge_index)          ## 1,3,5hop   inf
        xx5=mynorm(x5)-mynorm(x3)
        
        x6 = self.conv6(x5, edge_index)         ## 0,2,4,6hop inf
        xx6=mynorm(x6)-mynorm(x4)
        
        x7 = self.conv7(x6, edge_index)          ## 1,3,5,7 hop   inf
        xx7=mynorm(x7)-mynorm(x5)
        
        x8 = self.conv8(x7, edge_index)         ## 0,2,4,6,8 hop inf
        xx8=mynorm(x8)-mynorm(x6)
        
        x9 = self.conv9(x8, edge_index)         ## 0,2,4,6,8 hop inf
        xx9=mynorm(x9)-mynorm(x7)
        
        x10 = self.conv10(x9, edge_index)         ## 0,2,4,6,8 hop inf
        xx10=mynorm(x10)-mynorm(x8)
        
        x11 = self.conv11(x10, edge_index)         ## 0,2,4,6,8 hop inf
        xx11=mynorm(x11)-mynorm(x9)
        
        x12 = self.conv12(x11, edge_index)         ## 0,2,4,6,8 hop inf
        xx12=mynorm(x12)-mynorm(x10)
        
        x13 = self.conv13(x12, edge_index)         ## 0,2,4,6,8 hop inf
        xx13=mynorm(x13)-mynorm(x11)
        
        x14 = self.conv14(x13, edge_index)         ## 0,2,4,6,8 hop inf
        xx14=mynorm(x14)-mynorm(x12)
        
        x15 = self.conv15(x14, edge_index)         ## 0,2,4,6,8 hop inf
        xx15=mynorm(x15)-mynorm(x13)

        xx16=torch.cat((x0,xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,xx9,xx10,xx11,xx12,xx13,xx14,xx15),dim=1)
        x16 = self.conv16(xx16) 
        return x16       



class SDGCN32(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv1 = GCNConv(32,32)
        self.conv2 = GCNConv(32,32)
        self.conv3 = GCNConv(32,32)
        self.conv4 = GCNConv(32,32)
        self.conv5 = GCNConv(32,32)
        self.conv6 = GCNConv(32,32)
        self.conv7 = GCNConv(32,32)
        self.conv8 = GCNConv(32,32)
        self.conv9 = GCNConv(32,32)
        self.conv10 = GCNConv(32,32)
        self.conv11 = GCNConv(32,32)
        self.conv12 = GCNConv(32,32)
        self.conv13 = GCNConv(32,32)
        self.conv14 = GCNConv(32,32)
        self.conv15 = GCNConv(32,32)  
        self.conv16 = GCNConv(32,32)
        self.conv17 = GCNConv(32,32)
        self.conv18 = GCNConv(32,32)
        self.conv19 = GCNConv(32,32)
        self.conv20 = GCNConv(32,32)
        self.conv21 = GCNConv(32,32)
        self.conv22 = GCNConv(32,32)
        self.conv23 = GCNConv(32,32)
        self.conv24 = GCNConv(32,32)
        self.conv25 = GCNConv(32,32)
        self.conv26 = GCNConv(32,32)
        self.conv27 = GCNConv(32,32)
        self.conv28 = GCNConv(32,32)
        self.conv29 = GCNConv(32,32)
        self.conv30 = GCNConv(32,32) 
        self.conv31 = GCNConv(32,32)
        self.conv32 = nn.Linear(32*32,out_channels)
    
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()    
    
    
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0= self.fc1(x).relu()
        x0=mynorm(x0)
        
        x1 = self.conv1(x0, edge_index)         ## 1hop       inf
        xx1= x1
        x2 = self.conv2(x1, edge_index)         ## 0,2hop       inf
        xx2=mynorm(x2)- mynorm(x0)
        x3 = self.conv3(x2, edge_index)          ## 1,3hop     inf
        xx3=mynorm(x3)-mynorm(x1)
        
        x4 = self.conv4(x3, edge_index)         ## 0,2,4hop   inf
        xx4=mynorm(x4)-mynorm(x2) 
        
        x5 = self.conv5(x4, edge_index)          ## 1,3,5hop   inf
        xx5=mynorm(x5)-mynorm(x3)
        
        x6 = self.conv6(x5, edge_index)         ## 0,2,4,6hop inf
        xx6=mynorm(x6)-mynorm(x4)
        
        x7 = self.conv7(x6, edge_index)          ## 1,3,5,7 hop   inf
        xx7=mynorm(x7)-mynorm(x5)
        
        x8 = self.conv8(x7, edge_index)         ## 0,2,4,6,8 hop inf
        xx8=mynorm(x8)-mynorm(x6)
        
        x9 = self.conv9(x8, edge_index)         ## 0,2,4,6,8 hop inf
        xx9=mynorm(x9)-mynorm(x7)
        
        x10 = self.conv10(x9, edge_index)         ## 0,2,4,6,8 hop inf
        xx10=mynorm(x10)-mynorm(x8)
        
        x11 = self.conv11(x10, edge_index)         ## 0,2,4,6,8 hop inf
        xx11=mynorm(x11)-mynorm(x9)
        
        x12 = self.conv12(x11, edge_index)         ## 0,2,4,6,8 hop inf
        xx12=mynorm(x12)-mynorm(x10)
        
        x13 = self.conv13(x12, edge_index)         ## 0,2,4,6,8 hop inf
        xx13=mynorm(x13)-mynorm(x11)
        
        x14 = self.conv14(x13, edge_index)         ## 0,2,4,6,8 hop inf
        xx14=mynorm(x14)-mynorm(x12)
        
        x15 = self.conv15(x14, edge_index)         ## 0,2,4,6,8 hop inf
        xx15=mynorm(x15)-mynorm(x13)

        x16 = self.conv16(x15, edge_index)         ## 0,2,4,6,8 hop inf
        xx16=mynorm(x16)-mynorm(x14)

        x17 = self.conv17(x16, edge_index)         ## 0,2,4,6,8 hop inf
        xx17=mynorm(x17)-mynorm(x15)

        x18 = self.conv18(x17, edge_index)         ## 0,2,4,6,8 hop inf
        xx18=mynorm(x18)-mynorm(x16)

        x19 = self.conv19(x18, edge_index)         ## 0,2,4,6,8 hop inf
        xx19=mynorm(x19)-mynorm(x17)

        x20 = self.conv20(x19, edge_index)         ## 0,2,4,6,8 hop inf
        xx20=mynorm(x20)-mynorm(x18)

        x21 = self.conv21(x20, edge_index)         ## 0,2,4,6,8 hop inf
        xx21=mynorm(x21)-mynorm(x19)

        x22 = self.conv22(x21, edge_index)         ## 0,2,4,6,8 hop inf
        xx22=mynorm(x22)-mynorm(x20)

        x23 = self.conv23(x22, edge_index)         ## 0,2,4,6,8 hop inf
        xx23=mynorm(x23)-mynorm(x21)

        x24 = self.conv24(x23, edge_index)         ## 0,2,4,6,8 hop inf
        xx24=mynorm(x24)-mynorm(x22)

        x25 = self.conv25(x24, edge_index)         ## 0,2,4,6,8 hop inf
        xx25=mynorm(x25)-mynorm(x23)

        x26 = self.conv26(x25, edge_index)         ## 0,2,4,6,8 hop inf
        xx26=mynorm(x26)-mynorm(x24)

        x27 = self.conv27(x26, edge_index)         ## 0,2,4,6,8 hop inf
        xx27=mynorm(x27)-mynorm(x25)

        x28 = self.conv28(x27, edge_index)         ## 0,2,4,6,8 hop inf
        xx28=mynorm(x28)-mynorm(x26)

        x29 = self.conv29(x28, edge_index)         ## 0,2,4,6,8 hop inf
        xx29=mynorm(x29)-mynorm(x27)

        x30 = self.conv30(x29, edge_index)         ## 0,2,4,6,8 hop inf
        xx30=mynorm(x30)-mynorm(x28)

        x31 = self.conv31(x30, edge_index)         ## 0,2,4,6,8 hop inf
        xx31=mynorm(x31)-mynorm(x29)


        xx32=torch.cat((x0,xx1,xx2,xx3,xx4,xx5,xx6,xx7,xx8,xx9,xx10,xx11,xx12,xx13,xx14,xx15,x16,xx17,xx18,xx19,xx20,xx21,xx22,xx23,xx24,xx25,xx26,xx27,xx28,xx29,xx30,xx31),dim=1)
        x32 = self.conv32(xx32) 
        return x32     
    
class GEN4(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
#         self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv0 = Linear(in_channels,32)
        self.conv1 = GENConv(32,32)
        self.conv2 = GENConv(32,32)
        self.conv3 = GENConv(32,32)
        self.conv4 = Linear(32,out_channels)
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()   
            
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0 = self.conv0(x)   ## 1hop       inf
        x1 = self.conv1(x0,edge_index).relu()
        x2 = self.conv2(x1,edge_index).relu()
        x3 = self.conv3(x2,edge_index).relu()
        x4 = self.conv4(x3)   ## 1hop       inf
        return x4  


class GEN8(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
#         self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv0 = Linear(in_channels,32)
        self.conv1 = GENConv(32,32)
        self.conv2 = GENConv(32,32)
        self.conv3 = GENConv(32,32)
        self.conv4 = GENConv(32,32)
        self.conv5 = GENConv(32,32)
        self.conv6 = GENConv(32,32)
        self.conv7 = GENConv(32,32)
        self.conv8 = Linear(32,out_channels)
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()   
            
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0 = self.conv0(x)   ## 1hop       inf
        x1 = self.conv1(x0,edge_index).relu()
        x2 = self.conv2(x1,edge_index).relu()
        x3 = self.conv3(x2,edge_index).relu()
        x4 = self.conv4(x3,edge_index).relu()
        x5 = self.conv5(x4,edge_index).relu()
        x6 = self.conv6(x5,edge_index).relu()
        x7 = self.conv7(x6,edge_index).relu()
        x8 = self.conv8(x7)   ## 1hop       inf
        return x8  


class GEN16(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
#         self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv0 = Linear(in_channels,32)
        self.conv1 = GENConv(32,32)
        self.conv2 = GENConv(32,32)
        self.conv3 = GENConv(32,32)
        self.conv4 = GENConv(32,32)
        self.conv5 = GENConv(32,32)
        self.conv6 = GENConv(32,32)
        self.conv7 = GENConv(32,32)
        self.conv8 = GENConv(32,32)
        self.conv9 = GENConv(32,32)
        self.conv10 = GENConv(32,32)
        self.conv11 = GENConv(32,32)
        self.conv12 = GENConv(32,32)
        self.conv13 = GENConv(32,32)
        self.conv14 = GENConv(32,32)
        self.conv15 = GENConv(32,32)
        self.conv16 = Linear(32,out_channels)
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()   
            
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0 = self.conv0(x)   ## 1hop       inf
        x1 = self.conv1(x0,edge_index).relu()
        x2 = self.conv2(x1,edge_index).relu()
        x3 = self.conv3(x2,edge_index).relu()
        x4 = self.conv4(x3,edge_index).relu()
        x5 = self.conv5(x4,edge_index).relu()
        x6 = self.conv6(x5,edge_index).relu()
        x7 = self.conv7(x6,edge_index).relu()
        x8 = self.conv8(x7,edge_index).relu()
        x9 = self.conv9(x8,edge_index).relu()
        x10 = self.conv10(x9,edge_index).relu()
        x11 = self.conv11(x10,edge_index).relu()
        x12 = self.conv12(x11,edge_index).relu()
        x13 = self.conv13(x12,edge_index).relu()
        x14 = self.conv14(x13,edge_index).relu()
        x15 = self.conv15(x14,edge_index).relu()
        x16 = self.conv16(x15)   ## 1hop       inf
 
        return x16  


class GEN32(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
#         self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv0 = Linear(in_channels,32)
        self.conv1 = GENConv(32,32)
        self.conv2 = GENConv(32,32)
        self.conv3 = GENConv(32,32)
        self.conv4 = GENConv(32,32)
        self.conv5 = GENConv(32,32)
        self.conv6 = GENConv(32,32)
        self.conv7 = GENConv(32,32)
        self.conv8 = GENConv(32,32)
        self.conv9 = GENConv(32,32)
        self.conv10 = GENConv(32,32)
        self.conv11 = GENConv(32,32)
        self.conv12 = GENConv(32,32)
        self.conv13 = GENConv(32,32)
        self.conv14 = GENConv(32,32)
        self.conv15 = GENConv(32,32)
        self.conv16 = GENConv(32,32)
        self.conv17 = GENConv(32,32)
        self.conv18 = GENConv(32,32)
        self.conv19 = GENConv(32,32)
        self.conv20 = GENConv(32,32)
        self.conv21 = GENConv(32,32)
        self.conv22 = GENConv(32,32)
        self.conv23 = GENConv(32,32)
        self.conv24 = GENConv(32,32)
        self.conv25 = GENConv(32,32)
        self.conv26 = GENConv(32,32)
        self.conv27 = GENConv(32,32)
        self.conv28 = GENConv(32,32)
        self.conv29 = GENConv(32,32)
        self.conv30 = GENConv(32,32)
        self.conv31 = GENConv(32,32)
        self.conv32 = Linear(32,out_channels)
        
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()   
            
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0 = self.conv0(x)   ## 1hop       inf
        x1 = self.conv1(x0,edge_index).relu()
        x2 = self.conv2(x1,edge_index).relu()
        x3 = self.conv3(x2,edge_index).relu()
        x4 = self.conv4(x3,edge_index).relu()
        x5 = self.conv5(x4,edge_index).relu()
        x6 = self.conv6(x5,edge_index).relu()
        x7 = self.conv7(x6,edge_index).relu()
        x8 = self.conv8(x7,edge_index).relu()
        x9 = self.conv9(x8,edge_index).relu()
        x10 = self.conv10(x9,edge_index).relu()
        x11 = self.conv11(x10,edge_index).relu()
        x12 = self.conv12(x11,edge_index).relu()
        x13 = self.conv13(x12,edge_index).relu()
        x14 = self.conv14(x13,edge_index).relu()
        x15 = self.conv15(x14,edge_index).relu()
        x16 = self.conv16(x15,edge_index).relu()
        x17 = self.conv17(x16,edge_index).relu()
        x18 = self.conv18(x17,edge_index).relu()
        x19 = self.conv19(x18,edge_index).relu()
        x20 = self.conv20(x19,edge_index).relu()
        x21 = self.conv21(x20,edge_index).relu()
        x22 = self.conv22(x21,edge_index).relu()
        x23 = self.conv23(x22,edge_index).relu()
        x24 = self.conv24(x23,edge_index).relu()
        x25 = self.conv25(x24,edge_index).relu()
        x26 = self.conv26(x25,edge_index).relu()
        x27 = self.conv27(x26,edge_index).relu()
        x28 = self.conv28(x27,edge_index).relu()
        x29 = self.conv29(x28,edge_index).relu()
        x30 = self.conv30(x29,edge_index).relu()
        x31 = self.conv31(x30,edge_index).relu()
        x32 = self.conv32(x31)   ## 1hop       inf
        return x32 
    
    
    
###################### SG ####################
    
class SG4(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
#         self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv0 = Linear(in_channels,32)
        self.conv1 = SGConv(32,32)
        self.conv2 = SGConv(32,32)
        self.conv3 = SGConv(32,32)
        self.conv4 = Linear(32,out_channels)
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()   
            
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0 = self.conv0(x)   ## 1hop       inf
        x1 = self.conv1(x0,edge_index).relu()
        x2 = self.conv2(x1,edge_index).relu()
        x3 = self.conv3(x2,edge_index).relu()
        x4 = self.conv4(x3)   ## 1hop       inf
        return x4  


class SG8(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
#         self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv0 = Linear(in_channels,32)
        self.conv1 = SGConv(32,32)
        self.conv2 = SGConv(32,32)
        self.conv3 = SGConv(32,32)
        self.conv4 = SGConv(32,32)
        self.conv5 = SGConv(32,32)
        self.conv6 = SGConv(32,32)
        self.conv7 = SGConv(32,32)
        self.conv8 = Linear(32,out_channels)
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()   
            
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0 = self.conv0(x)   ## 1hop       inf
        x1 = self.conv1(x0,edge_index).relu()
        x2 = self.conv2(x1,edge_index).relu()
        x3 = self.conv3(x2,edge_index).relu()
        x4 = self.conv4(x3,edge_index).relu()
        x5 = self.conv5(x4,edge_index).relu()
        x6 = self.conv6(x5,edge_index).relu()
        x7 = self.conv7(x6,edge_index).relu()
        x8 = self.conv8(x7)   ## 1hop       inf
        return x8  


class SG16(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
#         self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv0 = Linear(in_channels,32)
        self.conv1 = SGConv(32,32)
        self.conv2 = SGConv(32,32)
        self.conv3 = SGConv(32,32)
        self.conv4 = SGConv(32,32)
        self.conv5 = SGConv(32,32)
        self.conv6 = SGConv(32,32)
        self.conv7 = SGConv(32,32)
        self.conv8 = SGConv(32,32)
        self.conv9 = SGConv(32,32)
        self.conv10 = SGConv(32,32)
        self.conv11 = SGConv(32,32)
        self.conv12 = SGConv(32,32)
        self.conv13 = SGConv(32,32)
        self.conv14 = SGConv(32,32)
        self.conv15 = SGConv(32,32)
        self.conv16 = Linear(32,out_channels)
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()   
            
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0 = self.conv0(x)   ## 1hop       inf
        x1 = self.conv1(x0,edge_index).relu()
        x2 = self.conv2(x1,edge_index).relu()
        x3 = self.conv3(x2,edge_index).relu()
        x4 = self.conv4(x3,edge_index).relu()
        x5 = self.conv5(x4,edge_index).relu()
        x6 = self.conv6(x5,edge_index).relu()
        x7 = self.conv7(x6,edge_index).relu()
        x8 = self.conv8(x7,edge_index).relu()
        x9 = self.conv9(x8,edge_index).relu()
        x10 = self.conv10(x9,edge_index).relu()
        x11 = self.conv11(x10,edge_index).relu()
        x12 = self.conv12(x11,edge_index).relu()
        x13 = self.conv13(x12,edge_index).relu()
        x14 = self.conv14(x13,edge_index).relu()
        x15 = self.conv15(x14,edge_index).relu()
        x16 = self.conv16(x15)   ## 1hop       inf
 
        return x16  


class SG32(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
#         self.fc1 = nn.Linear(in_channels,32)
#         self.fc2=nn.Linear(64,16)
        self.conv0 = Linear(in_channels,32)
        self.conv1 = SGConv(32,32)
        self.conv2 = SGConv(32,32)
        self.conv3 = SGConv(32,32)
        self.conv4 = SGConv(32,32)
        self.conv5 = SGConv(32,32)
        self.conv6 = SGConv(32,32)
        self.conv7 = SGConv(32,32)
        self.conv8 = SGConv(32,32)
        self.conv9 = SGConv(32,32)
        self.conv10 = SGConv(32,32)
        self.conv11 = SGConv(32,32)
        self.conv12 = SGConv(32,32)
        self.conv13 = SGConv(32,32)
        self.conv14 = SGConv(32,32)
        self.conv15 = SGConv(32,32)
        self.conv16 = SGConv(32,32)
        self.conv17 = SGConv(32,32)
        self.conv18 = SGConv(32,32)
        self.conv19 = SGConv(32,32)
        self.conv20 = SGConv(32,32)
        self.conv21 = SGConv(32,32)
        self.conv22 = SGConv(32,32)
        self.conv23 = SGConv(32,32)
        self.conv24 = SGConv(32,32)
        self.conv25 = SGConv(32,32)
        self.conv26 = SGConv(32,32)
        self.conv27 = SGConv(32,32)
        self.conv28 = SGConv(32,32)
        self.conv29 = SGConv(32,32)
        self.conv30 = SGConv(32,32)
        self.conv31 = SGConv(32,32)
        self.conv32 = Linear(32,out_channels)
        
    def reset_parameters(self):
        for layer in self.children():
            layer.reseEt_parameters()   
            
    def forward(self, x, edge_index):
#         print("conv00:{}".format(x))
        x0 = self.conv0(x)   ## 1hop       inf
        x1 = self.conv1(x0,edge_index).relu()
        x2 = self.conv2(x1,edge_index).relu()
        x3 = self.conv3(x2,edge_index).relu()
        x4 = self.conv4(x3,edge_index).relu()
        x5 = self.conv5(x4,edge_index).relu()
        x6 = self.conv6(x5,edge_index).relu()
        x7 = self.conv7(x6,edge_index).relu()
        x8 = self.conv8(x7,edge_index).relu()
        x9 = self.conv9(x8,edge_index).relu()
        x10 = self.conv10(x9,edge_index).relu()
        x11 = self.conv11(x10,edge_index).relu()
        x12 = self.conv12(x11,edge_index).relu()
        x13 = self.conv13(x12,edge_index).relu()
        x14 = self.conv14(x13,edge_index).relu()
        x15 = self.conv15(x14,edge_index).relu()
        x16 = self.conv16(x15,edge_index).relu()
        x17 = self.conv17(x16,edge_index).relu()
        x18 = self.conv18(x17,edge_index).relu()
        x19 = self.conv19(x18,edge_index).relu()
        x20 = self.conv20(x19,edge_index).relu()
        x21 = self.conv21(x20,edge_index).relu()
        x22 = self.conv22(x21,edge_index).relu()
        x23 = self.conv23(x22,edge_index).relu()
        x24 = self.conv24(x23,edge_index).relu()
        x25 = self.conv25(x24,edge_index).relu()
        x26 = self.conv26(x25,edge_index).relu()
        x27 = self.conv27(x26,edge_index).relu()
        x28 = self.conv28(x27,edge_index).relu()
        x29 = self.conv29(x28,edge_index).relu()
        x30 = self.conv30(x29,edge_index).relu()
        x31 = self.conv31(x30,edge_index).relu()
        x32 = self.conv32(x31)   ## 1hop       inf
        return x32 
    
