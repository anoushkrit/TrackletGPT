import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

# ===== BATCHED GRAPH CONSTRUCTION =====
def construct_knn_graph_batch(batch_point_embeddings, k=8):
    """
    Constructs a batched k-NN graph from a batch of variable-sized PointNet embeddings.

    Args:
        batch_point_embeddings (list of torch.Tensor): List of tensors of shape (num_patches, feature_dim) for each sample.
        k (int): Number of neighbors for k-NN graph.

    Returns:
        torch_geometric.data.Batch: Batched graph data object.
    """
    graph_list = []

    for i, point_embeddings in enumerate(batch_point_embeddings):
        num_patches = point_embeddings.shape[0]

        # If num_patches < k, reduce k to avoid errors
        k_neighbors = min(k, num_patches - 1) if num_patches > 1 else 1  

        # Compute k-NN graph for this sample
        edge_index = knn_graph(point_embeddings, k=k_neighbors, loop=False)

        # Create PyG Data object
        graph_data = Data(x=point_embeddings, edge_index=edge_index)
        graph_list.append(graph_data)

    # Combine variable-sized graphs into a batch
    batched_graph = Batch.from_data_list(graph_list)

    return batched_graph

# ===== GAT FOR BATCHED GRAPHS =====
class GAT(torch.nn.Module):
    def __init__(self, config):
        """
        Graph Attention Network (GAT) for batched graphs.

        Args:
            in_channels (int): Input feature dimension.
            hidden_channels (int): Hidden feature dimension.
            out_channels (int): Output feature dimension.
            num_heads (int): Number of attention heads.
        """
        super(GAT, self).__init__()
        self.config = config
        in_channels = config.encoder_dim
        hidden_channels = config.hidden_dim
        out_channels = config.out_dim
        num_heads = config.num_heads
        self.k = config.k
        
        # GATConv layers
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False)

    def construct_knn_graph_batch(self, batch_point_embeddings, k=8):
        """
        Constructs a batched k-NN graph from a batch of variable-sized PointNet embeddings.

        Args:
            batch_point_embeddings (list of torch.Tensor): List of tensors of shape (num_patches, feature_dim) for each sample.
            k (int): Number of neighbors for k-NN graph.

        Returns:
            torch_geometric.data.Batch: Batched graph data object.
        """
        graph_list = []

        for i, point_embeddings in enumerate(batch_point_embeddings):
            num_patches = point_embeddings.shape[0]

            # If num_patches < k, reduce k to avoid errors
            k_neighbors = min(k, num_patches - 1) if num_patches > 1 else 1  

            # Compute k-NN graph for this sample
            edge_index = knn_graph(point_embeddings, k=k_neighbors, loop=False)

            # Create PyG Data object
            graph_data = Data(x=point_embeddings, edge_index=edge_index)
            graph_list.append(graph_data)

        # Combine variable-sized graphs into a batch
        batched_graph = Batch.from_data_list(graph_list)

        return batched_graph

    def forward(self, batch_point_embeddings):
        batched_data = self.construct_knn_graph_batch(batch_point_embeddings, k=self.k)

        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

        x = F.elu(self.gat1(x, edge_index))  # First GAT layer
        x = self.gat2(x, edge_index)  # Second GAT layer

        return x  # Processed node embeddings for the batch
    
class LSTMencoder(nn.Module):
    def __init__(self, config):
        super(LSTMencoder, self).__init__()

        input_dim = config.encoder_dim
        hidden_dim = config.hidden_dim
        num_layers = config.num_layers
        bidirectional = config.bidirectional

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
    
    def forward(self, x):
        # x shape: (B, N, D) -> (batch, seq_length, input_dim)
        output, (h_n, c_n) = self.lstm(x)  
        return output, (h_n, c_n)  # h_n can be used as a summary representation


# ===== TESTING THE IMPLEMENTATION =====
if __name__ == "__main__":
    input_dim = 128  # Output dim from PointNet
    hidden_dim = 256
    num_layers = 2
    lstm_encoder = LSTMencoder(input_dim, hidden_dim, num_layers)

    pointnet_output = torch.randn(16, 64, 128)  # (batch_size=16, patches=64, feature_dim=128)
    lstm_output, _ = lstm_encoder(pointnet_output)
    print(lstm_output.shape)  # (16, 64, 256)
