import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class my_CNN(nn.Module):
    def __init__(self, input_dim=4, output_dim=1):
        super(my_CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, 32, 7, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 7, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding="same"), 
            nn.Flatten(), 
        )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, k_size=3):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn,ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ResNet(nn.Module):
    def __init__(self,input_dim, ResidualBlock, num_classes=361):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2)
        self.layer2 = self.make_layer(ResidualBlock, 64, 2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2)        
        self.layer4 = self.make_layer(ResidualBlock, 64, 2)    
        self.layer5 = self.make_layer(ResidualBlock, 64, 2)
        self.layer6 = self.make_layer(ResidualBlock, 64, 2)
        self.layer7 = self.make_layer(ResidualBlock, 64, 2)
        self.layer8 = self.make_layer(ResidualBlock, 64, 2)
        self.layer9 = self.make_layer(ResidualBlock, 64, 2)
        self.layer10 = self.make_layer(ResidualBlock, 64, 2)
        self.layer11 = self.make_layer(ResidualBlock, 64, 2)        
        self.layer12 = self.make_layer(ResidualBlock, 64, 2)    

        self.fc = nn.Linear(64, num_classes)
        self.convert = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding="same"),
            nn.BatchNorm2d(1)
        )
   
        
    def make_layer(self, block, channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.inchannel, channels))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.convert(out)
        out = out.view(out.size(0),-1)
        return out
    
class ResViT_PS(nn.Module):
    def __init__(self,input_dim, ResidualBlock, num_classes=3, dim=361, depth=2, heads=16, dim_head=64, mlp_dim=128):
        super(ResViT_PS, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, 7)
        
        self.layer2 = self.make_layer(ResidualBlock, 64, 2, 5)
        
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, 3)
        
        self.layer4 = self.make_layer(ResidualBlock, 64, 2, 3)
  
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c h w -> b c (h w)"), # 256, 64, 19, 19 => 256, 64, 361
            nn.LayerNorm(361),
            nn.Linear(361, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)  
        self.convert = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding="same"),
            nn.BatchNorm2d(1)
        )
        self.fc = nn.Linear(361, num_classes)
        
    def make_layer(self, block, channels, num_blocks, k_size):
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.inchannel, channels, k_size))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.to_patch_embedding(out)
        out = self.transformer(out)
        out = out.view(out.size(0), 64, 19, 19)
        out = self.convert(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out
    
class ResViT(nn.Module):
    #For Dan
    def __init__(self,input_dim, ResidualBlock, num_classes=361, dim=361, depth=2, heads=4, dim_head=256, mlp_dim=512):
        super(ResViT, self).__init__()
        self.inchannel = 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.layer1 = self.make_layer(ResidualBlock, 256, 2)
        self.layer2 = self.make_layer(ResidualBlock, 256, 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 2)
  
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c h w -> b c (h w)"), # 256, 256, 19, 19 => 256, 256, 361
            nn.LayerNorm(361),
            nn.Linear(361, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)  
        self.convert = nn.Sequential(
            nn.Conv2d(256, 1, 3, padding="same"),
            nn.BatchNorm2d(1)
        )
   
        
    def make_layer(self, block, channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.inchannel, channels))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  
        out = self.to_patch_embedding(out)
        out = self.transformer(out)
        out = out.view(out.size(0), 256, 19, 19)
        out = self.convert(out)
        out = out.view(out.size(0),-1)
        return out

class ResViT_kyu(nn.Module):
    
    def __init__(self,input_dim, ResidualBlock, num_classes=361, dim=361, depth=2, heads=4, dim_head=128, mlp_dim=256):
        super(ResViT_kyu, self).__init__()
        self.inchannel = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.layer1 = self.make_layer(ResidualBlock, 128, 2)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2)
        self.layer3 = self.make_layer(ResidualBlock, 128, 2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2)
  
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c h w -> b c (h w)"), # 256, 128, 19, 19 => 256, 128, 361
            nn.LayerNorm(361),
            nn.Linear(361, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)  
        self.convert = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding="same"),
            nn.BatchNorm2d(1)
        )
   
        
    def make_layer(self, block, channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.inchannel, channels))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)    
        out = self.to_patch_embedding(out)
        out = self.transformer(out)
        out = out.view(out.size(0), 128, 19, 19)
        out = self.convert(out)
        out = out.view(out.size(0),-1)
        return out
