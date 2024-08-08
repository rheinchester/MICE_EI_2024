import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

# Assuming data preparation and model definition from previous example

class GNNModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# Load your data
data = pd.read_csv('your_data.csv')

# Assuming each row represents an edge with two nodes and features
# Example: node_a, node_b = data['column1'], data['column2']
# Example: features = data[['ATB', 'EI', 'ESS', 'OR', 'PBC', 'SSN']]

# Create a graph
# For simplicity, let's create a random graph with NetworkX
G_nx = nx.gnp_random_graph(10, 0.2)
g = dgl.from_networkx(G_nx)

# Example node features (randomly generated for illustration)
features = torch.rand((10, 6))  # 10 nodes with 6 features each
g.ndata['feat'] = features

# Define the model
model = GNNModel(in_feats=6, h_feats=16, num_classes=2)
model.eval()

# Predict links
with torch.no_grad():
    logits = model(g, g.ndata['feat'])
    pred = torch.argmax(logits, dim=1)

# Convert DGL graph back to NetworkX for visualization
G_nx = g.to_networkx().to_undirected()

# Visualize the original graph
pos = nx.spring_layout(G_nx)
plt.figure(figsize=(12, 8))

nx.draw(G_nx, pos, with_labels=True, node_color='skyblue', edge_color='k', node_size=500, alpha=0.7)

# Highlight predicted links
predicted_edges = [(i, j) for i in range(len(pred)) for j in range(i+1, len(pred)) if pred[i] == pred[j]]
nx.draw_networkx_edges(G_nx, pos, edgelist=predicted_edges, edge_color='r', width=2)

plt.title('Graph with Predicted Links Highlighted')
plt.show()
