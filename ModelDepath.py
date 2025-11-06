import torch
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import argparse
import torch.nn.functional as F
from torch_geometric.datasets import Coauthor,CitationFull,Amazon,AttributedGraphDataset
import torch_geometric.transforms as T
import torch
import model as M
import model2 as M2


def train(model,train_set_ind,train_label_ind,optimizer,data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_set_ind], data.y[train_label_ind])
    loss.backward()
    optimizer.step()

    return float(loss)

@torch.no_grad()
def test(model,train_set_ind,test_set_ind,optimizer,data):
    model.eval()
    ret = model(data.x, data.edge_index)
    pred=ret.argmax(dim=-1)
    mask = train_set_ind
    trainacc=int((pred[mask] == data.y[mask]).sum()) / int(len(mask))
    mask = test_set_ind
    testacc=int((pred[mask] == data.y[mask]).sum()) / int(len(mask))
        
    return trainacc,testacc


def compute(setname,version,modelname):
    
    if setname == "Cite":
        nm = CitationFull(root="../datasets",name="citeseer") 
    elif setname == "Cora": 
        nm = CitationFull(root="../datasets",name="cora_ml")
    elif setname == "Pub":     
        nm = CitationFull(root="../datasets",name="pubmed")
    elif setname == "Computer":
        nm = Amazon(root="../datasets",name="Computers")
    elif setname == "Photo":
        nm = Amazon(root="../datasets",name="Photo")
    elif setname == "Blog": 
        nm = AttributedGraphDataset(root="../datasets",name="BlogCatalog")
    elif setname == "Wiki":     
        nm = AttributedGraphDataset(root="../datasets",name="WIKI")
    elif setname == "CS":
        nm = Coauthor(root="../datasets2",name="CS")  
    elif setname == "Physics":
        nm = Coauthor(root="../datasets2",name="Physics")
    else:
        print("datasets error!\n")
        return
        
    print(f'Dataset: {nm}:')
    print(f'Number of graphs: {len(nm)}')
    print(f'Number of features: {nm.num_features}')
    print(f'Number of classes: {nm.num_classes}')
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = nm
    data = dataset[0].to(device)


    framedata=[]
    for percent in range(1,10):
        for num in range(10):
            index=[i for i in range(dataset.x.shape[0])]
            train_set_ind,test_set_ind,train_label_ind,test_label_ind=train_test_split(index,index,test_size=0.1*percent,random_state=42,stratify=dataset.y)
            value=[0.1*percent,0,0,0,0]

            if version == 32:
                # model = GEN8(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "GCN":
                    model = M.GCN(in_channels=dataset.num_node_features,hidden_channels=32,num_layers=32,out_channels=dataset.num_classes).to(device)
                if modelname == "GAT":
                    model = M.GAT(in_channels=dataset.num_node_features,hidden_channels=32,num_layers=32,out_channels=dataset.num_classes).to(device)
                if modelname == "SDGCN":
                    model = M.SDGCN32(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "GEN":
                    model = M.GEN32(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "SG":
                    model = M.SG32(dataset.num_features, dataset.num_classes).to(device)  
                if modelname == "supGAT":
                    model = M2.SGAT32(dataset.num_features, dataset.num_classes).to(device) 
                    
                if modelname == "SDSG":
                    model = M2.SDSG32(dataset.num_features, dataset.num_classes).to(device)     
                    

            if version == 16:
                # model = GEN8(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "GCN":
                    model = M.GCN(in_channels=dataset.num_node_features,hidden_channels=32,num_layers=16,out_channels=dataset.num_classes).to(device)
                if modelname == "GAT":
                    model = M.GAT(in_channels=dataset.num_node_features,hidden_channels=32,num_layers=16,out_channels=dataset.num_classes).to(device)
                if modelname == "SDGCN":
                    model = M.SDGCN16(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "GEN":
                    model = M.GEN16(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "SG":
                    model = M.SG16(dataset.num_features, dataset.num_classes).to(device) 
                if modelname == "supGAT":
                    model = M2.SGAT16(dataset.num_features, dataset.num_classes).to(device)  
                if modelname == "SDSG":
                    model = M2.SDSG16(dataset.num_features, dataset.num_classes).to(device)     

            if version == 8:
                # model = GEN8(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "GCN":
                    model = M.GCN(in_channels=dataset.num_node_features,hidden_channels=32,num_layers=8,out_channels=dataset.num_classes).to(device)
                if modelname == "GAT":
                    model = M.GAT(in_channels=dataset.num_node_features,hidden_channels=32,num_layers=8,out_channels=dataset.num_classes).to(device)
                if modelname == "SDGCN":
                    model = M.SDGCN8(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "GEN":
                    model = M.GEN8(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "SG":
                    model = M.SG8(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "supGAT":
                    model = M2.SGAT8(dataset.num_features, dataset.num_classes).to(device)   
                    
                if modelname == "SDSG":
                    model = M2.SDSG8(dataset.num_features, dataset.num_classes).to(device)     
                
            if version == 9:
                if modelname == "SDSG":
                    model = M2.SDSG9(dataset.num_features, dataset.num_classes).to(device)  
            
            if version == 7:
                if modelname == "SDSG":
                    model = M2.SDSG7(dataset.num_features, dataset.num_classes).to(device)     
               
            if version == 5:
                if modelname == "SDSG":
                    model = M2.SDSG5(dataset.num_features, dataset.num_classes).to(device)     
                
                
                
            if version == 4:
                # model = GEN4(dataset.num_features, dataset.num_classes).to(device) 
                if modelname == "GCN":
                    model = M.GCN(in_channels=dataset.num_node_features,hidden_channels=32,num_layers=4,out_channels=dataset.num_classes).to(device)
                if modelname == "GAT":
                    model = M.GAT(in_channels=dataset.num_node_features,hidden_channels=32,num_layers=4,out_channels=dataset.num_classes).to(device) 
                if modelname == "SDGCN":
                    model = M.SDGCN4(dataset.num_features, dataset.num_classes).to(device)
                if modelname == "GEN":
                    model = M.GEN4(dataset.num_features, dataset.num_classes).to(device)                
                if modelname == "SG":
                    model = M.SG4(dataset.num_features, dataset.num_classes).to(device)   
                if modelname == "supGAT":
                    model = M2.SGAT4(dataset.num_features, dataset.num_classes).to(device) 
                if modelname == "SDSG":
                    model = M2.SDSG4(dataset.num_features, dataset.num_classes).to(device) 
                
#            model = torch_geometric.compile(model)                   # Compile the model into an optimized version:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
            
            for epoch in range(0, 400):
                loss = train(model,train_set_ind,train_label_ind,optimizer,data)
                    # 记录当前权重
                train_acc,test_acc = test(model,train_set_ind,test_set_ind,optimizer,data)
                print(f'Epoch: {epoch}, Loss: {loss:.4f}, Train: {train_acc:.4f},'f'Test: {test_acc:.4f}')
                if test_acc > value[4]:
                    value[4] = test_acc 
                    value[1] = epoch
                    value[2] = loss
                    value[3] = train_acc
            print(value)
            framedata.append(value)
            del model
            del optimizer
            torch.cuda.empty_cache()
            time.sleep(0.5)
    pd.DataFrame(framedata).to_csv("./{0}/{0}_{1}_{2}L.csv".format(modelname,setname,version))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name", default="cora", type=str, help="name of dataset")
    parser.add_argument("-v","--version", default=16, type=int, help="version of 4 or 8 or 16 or 32L")
    parser.add_argument("-m","--model", default="GCN", type=str, help="GCN or GAT or GEN")
    args = parser.parse_args()
    print(args.name)
    print(args.version)
    compute(args.name,args.version,args.model)

    
    

    
