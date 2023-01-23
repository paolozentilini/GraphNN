import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.profile import get_model_size
import torch.nn.functional as F
from torch.nn import Softmax, Linear
from torch_geometric.nn import global_max_pool, global_mean_pool, GCNConv, GraphConv, GATConv, SAGEConv
from torch_geometric.loader import DataListLoader, DataLoader
import sys


dataset = MNISTSuperpixels(root="mnist")
data_size = 57600
dataset = dataset[:data_size]
batch_size = 256
#learning_rate = 5.838e-05
learning_rate = 9.961656018667525e-05
hidden_channels = 512
n_epochs = 80
data = dataset[0]


print('Dataset:\t\t', dataset)
print("number of graphs:\t\t", len(dataset))
print("number of classes:\t\t\t", dataset.num_classes)
print("number of node features:\t", dataset.num_node_features)
print("number of edge features:\t", dataset.num_edge_features)


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


train_loader = DataLoader(
    dataset=dataset[:int(data_size * 0.8)],
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    dataset=dataset[int(data_size * 0.8):],
    batch_size=256,
    shuffle=True
)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.initial_conv = GraphConv(dataset.num_features, hidden_channels)
        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(2*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch_index):
        x = self.initial_conv(x, edge_index)
        x = x.tanh()
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)
        x = x.tanh()
        x = self.conv4(x, edge_index)
        x = x.tanh()

        x = torch.cat([global_max_pool(x, batch_index), global_mean_pool(x, batch_index)], dim=1)
        x = F.dropout(x, p=0.5, training=self.training)    # 3)
        x = self.lin1(x)
        x = x.tanh()
        x = self.lin2(x)
        x = x.tanh()
        x = self.lin3(x)
        output = F.log_softmax(x, dim=1)

        return output


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.in_head = 1
        self.out_head = 1

        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(hidden_channels*self.in_head, hidden_channels, concat=False, heads=self.in_head, dropout=0.6)
        self.conv3 = GATConv(hidden_channels*self.in_head, hidden_channels, concat=False, heads=self.out_head, dropout=0.6)
        self.conv4 = GATConv(hidden_channels*self.out_head, hidden_channels, concat=False, heads=self.out_head, dropout=0.6)
        self.lin1 = Linear(2*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index,  batch_index):
        # Dropout before the GAT layer helps avoid overfitting
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = x.tanh()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)
        x = x.tanh()
        x = self.conv4(x, edge_index)
        x = x.tanh()

        x = torch.cat([global_max_pool(x, batch_index), global_mean_pool(x, batch_index)], dim=1)
        x = F.dropout(x, p=0.6, training=self.training)    # 3)
        x = self.lin1(x)
        x = x.tanh()
        x = self.lin2(x)
        output = F.log_softmax(x, dim=1)

        return output


class GraphCNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GraphCNN, self).__init__()
        torch.manual_seed(12345)
        self.initial_conv = SAGEConv(in_channels=dataset.num_features,  out_channels=hidden_channels)
        self.conv1 = SAGEConv(in_channels=hidden_channels,  out_channels=hidden_channels)
        self.conv2 = SAGEConv(in_channels=hidden_channels,  out_channels=hidden_channels)
        self.conv3 = SAGEConv(in_channels=hidden_channels,  out_channels=hidden_channels)
        self.conv4 = SAGEConv(in_channels=hidden_channels,  out_channels=hidden_channels)
        self.lin1 = Linear(2*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch_index):
        x = self.initial_conv(x, edge_index)
        x = x.tanh()
        x = self.conv1(x, edge_index)
        x = x.tanh()
        x = self.conv2(x, edge_index)
        x = x.tanh()
        x = self.conv3(x, edge_index)
        x = x.tanh()
        x = self.conv4(x, edge_index)
        x = x.tanh()

        x = torch.cat([global_max_pool(x, batch_index), global_mean_pool(x, batch_index)], dim=1)
        x = F.dropout(x, p=0.5, training=self.training)    # 3)
        x = self.lin1(x)
        x = x.tanh()
        x = self.lin2(x)
        x = x.tanh()
        x = self.lin3(x)
        output = F.log_softmax(x, dim=1)

        return output


model = GraphCNN(hidden_channels=hidden_channels)
print(model)
print('Model size: ', get_model_size(model), ' bytes')
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

# Cross Entropy Loss
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train(model):
    acc = 0.
    counter = 0
    # Enumerate over the data
    for batch in train_loader:
        counter += 1
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(batch.x.float(), batch.edge_index, batch.batch)
        # Calculating the loss and gradients
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        # Update using the gradients
        optimizer.step()
        # Train accuracy
        acc += accuracy(pred.argmax(dim=1), batch.y)
    return loss, 100*acc/counter


def test(model):
    with torch.no_grad():
        """Evaluate the model on test set"""
        acc = 0.
        counter = 0
        for batch in test_loader:
            batch.to(device)
            model.eval()
            counter += 1
            out = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = torch.sqrt(loss_fn(out, batch.y))
            acc += accuracy(out.argmax(dim=1), batch.y)
        return loss, 100*acc/counter


print('Hyper parameters: ')
print('Learning_rate: ', learning_rate)
print('N epochs: ', n_epochs)
print('Batch size:', batch_size)
print('Device: ', device)
# Train the graph-neural-network:
print('Starting training...')

epochs = np.arange(0, n_epochs+1, 1)
train_losses = []
test_losses = []
train_accuracy_list = []
test_accuracy_list = []

for epoch in epochs:
    train_loss, train_acc = train(model)
    train_losses.append(train_loss.to("cpu"))
    test_loss, acc = test(model)
    test_losses.append(test_loss.to("cpu"))
    train_accuracy_list.append(train_acc)
    test_accuracy_list.append(acc)
    if epoch % 1 == 0:
        print(f"Epoch {epoch} \t| Train Loss {train_loss} \t| Train Accuracy {round(train_acc, 3)} "
              f"% \t| Test Loss {test_loss} \t| Test Accuracy {round(acc, 3)} %")

with torch.no_grad():
    plt.figure(figsize=(10, 5))
    plt.title('Loss', fontsize=20)
    plt.plot(epochs, train_losses, color='#88CCEE', label='Train loss', linewidth=3)
    plt.plot(epochs, test_losses, color='#332288', label='Test loss', linewidth=3)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.savefig('MnistLoss.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.title('Accuracy', fontsize=20)
    plt.plot(epochs, test_accuracy_list, color='#332288', label='Test accuracy', linewidth=3)
    plt.plot(epochs, train_accuracy_list, color='#88CCEE', label='Train accuracy', linewidth=3)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy [%]', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.savefig('MnistAccuracy.png')
    plt.show()


def image_example(data, y_pred, k):
    labels = [data.x.numpy()[i][0] for i in range(75)]
    data_nx = to_networkx(data, node_attrs=['x'])

    plt.figure(figsize=(10, 10))
    plt.title('Number in the picture: {} \n Number predicted by the GNN: {}'.format(data.y.numpy()[0], y_pred), fontsize=16)
    nx.draw_networkx(data_nx, with_labels=True, node_color=labels, cmap='binary')
    plt.savefig(f"mnist{k}.png")
    plt.show()


with torch.no_grad():

    dataset1 = MNISTSuperpixels(root="mnist", train=False)[0:4]
    loader = DataLoader(dataset=dataset1, batch_size=4)
    for batch in loader:
        batch.to(device)
        y_predicted = model(batch.x.float(), batch.edge_index, batch.batch).argmax(dim=1)
    y_predicted.to("cpu")

    for i in range(4):
        data = dataset1[i]
        y_pred = y_predicted[i]
        image_example(data, y_pred, i)
