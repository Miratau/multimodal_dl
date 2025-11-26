import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.modality_dim = num_classes
        
        self.tabnet_branch = nn.Sequential(
            nn.Linear(self.modality_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        self.vit_branch = nn.Sequential(
            nn.Linear(self.modality_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        combined_feature_dim = 32 + 64  # tabnet_branch output + vit_branch output
        self.attention = nn.Sequential(
            nn.Linear(combined_feature_dim, 64),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # 2 attention weights (one per modality)
            nn.Softmax(dim=1)
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        tabnet_proba = x[:, :self.num_classes]
        vit_proba = x[:, self.num_classes:]
        
        tabnet_features = self.tabnet_branch(tabnet_proba)
        vit_features = self.vit_branch(vit_proba) 
        
        combined_features = torch.cat([tabnet_features, vit_features], dim=1)
    
        attn_weights = self.attention(combined_features)
        
        tabnet_attn = attn_weights[:, 0:1].expand_as(tabnet_features)
        vit_attn = attn_weights[:, 1:2].expand_as(vit_features)
        
        weighted_tabnet = tabnet_features * tabnet_attn
        weighted_vit = vit_features * vit_attn
        
        fused_features = torch.cat([weighted_tabnet, weighted_vit], dim=1)
        
        logits = self.fusion_layer(fused_features)
        
        return logits
    
    def get_attention_weights(self, x):
        with torch.no_grad():
            tabnet_proba = x[:, :self.num_classes]
            vit_proba = x[:, self.num_classes:]
            
            tabnet_features = self.tabnet_branch(tabnet_proba)
            vit_features = self.vit_branch(vit_proba)
            
            combined_features = torch.cat([tabnet_features, vit_features], dim=1)
            attn_weights = self.attention(combined_features)
            
        return attn_weights

