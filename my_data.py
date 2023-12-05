import torch
import numpy as np
from torch.utils.data import Dataset
chars = 'abcdefghijklmnopqrs'
coordinates = {k:v for v,k in enumerate(chars)}
chartonumbers = {k:v for k,v in enumerate(chars)}


def prepare_input(moves):
    x = np.zeros((5, 19, 19))
    
    for move in moves:
        color = move[0] #move的顏色為B或W
        column = coordinates[move[2]] #為move的col
        row = coordinates[move[3]] #為move的row
        if color == 'B':
            x[0,row,column] = 1 #如果為黑子設定其值為1
            x[2,row,column] = 1 #有下黑子，設定empty之值為1(與定義相反但後面會反轉)
            x[4, :, :] = 1 #代表下的是黑方
        if color == 'W':
            x[1,row,column] = 1 #如果為白子設定其值為1
            x[2,row,column] = 1 #有下白子，設定empty之值為1(與定義相反但後面會反轉)
            x[4, :, :] = 0 #代表下的是白方
        #x[5, :, :] = idx
    if moves:
        # last_move_column = coordinates[moves[-1][2]]
        # last_move_row = coordinates[moves[-1][3]]
        x[3,row,column] = 1 #做完上面的for迴圈後，col和row儲存的為最後一手
    x[2,:,:] = np.where(x[2,:,:] == 0, 1, 0) #將0的位置改為1=>empty，1的位置改為0=>occupied
    return torch.tensor(torch.from_numpy(x), dtype=torch.float32)

def prepare_label(move):
    column = coordinates[move[2]] #為move的col
    row = coordinates[move[3]] #為move的row
    return torch.tensor(column*19+row)

# 用於訓練play style dateset
class GamesDataset(Dataset):
    def __init__(self, games):
        self.games = games

    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, idx):
        return self.games[idx]
    
# 用於訓練dan和kyu dateset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    