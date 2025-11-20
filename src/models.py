"""
models.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionPooling(nn.Module):
    """
    Attention Pooling layer that compresses Frame Sequence into a single vector
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1) # Generate probability value according to frame axis (dim = 1)
        )

    def forward(self, x):
        # x: [Batch, Frames, Dim]
        weights = self.attention(x) # [Batch, Frames, 1]
        
        # Weighted Sum
        out = torch.sum(x * weights, dim=1) # [Batch, Dim]
        return out

class HybridTemporalModel(nn.Module):
    """
    [Structure]
    Input (SLIP Embeddings) -> Positional Encoding -> 
    Transformer Encoder (1-layer) -> Temporal Attention Pooling -> Output
    """
    def __init__(self, input_dim=512, hidden_dim=512, n_heads=8, dropout=0.1):
        super().__init__()
        
        # 1. Transformer Encoder (1 layer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True # Keep the input structure as [Batch, Seq, Feature]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 2. Temporal Attention Pooling
        self.pooling = TemporalAttentionPooling(input_dim)
        
        # (Option) Location Encoding can be done here
        
    def forward(self, x):
        # x: [Batch, Frames, Input_Dim] (ex: Feature extraction per frames from SLIP output)
        
        # Transformer to learn temporal semantics
        x = self.transformer(x) # [Batch, Frames, Input_Dim]
        
        # Sequence compression by Pooling (Create Video Embedding)
        video_emb = self.pooling(x) # [Batch, Input_Dim]
        
        return video_emb

class ProtoNetClassifier(nn.Module):
    """
    Prototypical Network Head for Zero-Shot/Few-shot
    """
    def __init__(self):
        super().__init__()

    def compute_prototypes(self, support_embeddings, support_labels, n_classes):
        """
        Create Protype vector by averaging the Support set by classes
        """
        # support_embeddings: [Total_Support_Samples, Dim]
        # support_labels: [Total_Support_Samples]
        
        dim = support_embeddings.size(1)
        prototypes = torch.zeros(n_classes, dim).to(support_embeddings.device)

        for c in range(n_classes):
            # Extract samples from the given class (c)
            class_samples = support_embeddings[support_labels == c]
            if class_samples.size(0) > 0:
                prototypes[c] = class_samples.mean(dim=0)
                
        return prototypes # [N_Classes, Dim]

    def euclidean_distance(self, x, y):
        """
        x: Query Embeddings [N_Query, Dim]
        y: Prototypes [N_Classes, Dim]
        Returns: [N_Query, N_Classes] (거리 행렬)
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def forward(self, support_emb, support_labels, query_emb, n_classes):
        """
        End-to-End Forward
        """
        # 1. Create Prototype
        prototypes = self.compute_prototypes(support_emb, support_labels, n_classes)
        
        # 2. Compute Euclidean Distance
        # ProtoNet is better with smaller distance -> use negative (-) when using Logits
        dists = self.euclidean_distance(query_emb, prototypes)
        logits = -dists 
        
        return logits