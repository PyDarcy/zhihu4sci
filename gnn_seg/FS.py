from time import sleep
from pathlib import Path
from itertools import tee
from functools import lru_cache

import trimesh
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.transforms import BaseTransform, Compose, FaceToEdge
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.loader import DataLoader
import random
from absl import app
from absl import flags
flags.DEFINE_string("root", '.', help="path to model")
flags.DEFINE_enum("device", "cuda", ["cuda", "cpu"], help="device")

FLAGS = flags.FLAGS

# data processing

class NormalizeUnitSphere(BaseTransform):
    """Center and normalize node-level features to unit length."""

    @staticmethod
    def _re_center(x):
        """Recenter node-level features onto feature centroid."""
        centroid = torch.mean(x, dim=0)
        return x - centroid

    @staticmethod
    def _re_scale_to_unit_length(x):
        """Rescale node-level features to unit-length."""
        max_dist = torch.max(torch.norm(x, dim=1))
        return x / max_dist

    def __call__(self, data: Data):
        if data.pos is not None:
            data.pos = self._re_scale_to_unit_length(self._re_center(data.pos))

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
    
def load_mesh(mesh_filename: Path):
    mesh = trimesh.load_mesh(mesh_filename, process=False)
    vertices = torch.from_numpy(mesh.vertices).to(torch.float)
    faces = torch.from_numpy(mesh.faces)
    faces = faces.t().to(torch.long).contiguous()
    return vertices, faces

class SegmentationFaust(InMemoryDataset):
    map_seg_label_to_id = dict(
        head=0,
        torso=1,
        left_arm=2,
        left_hand=3,
        right_arm=4,
        right_hand=5,
        left_upper_leg=6,
        left_lower_leg=7,
        left_foot=8,
        right_upper_leg=9,
        right_lower_leg=10,
        right_foot=11,
    )

    def __init__(self, root, train: bool = True):
        """
        Parameters
        ----------
        root: PathLike
            Root directory where the dataset should be saved.
        train: bool
            Whether to load training data or test data.

        """
        super().__init__(root,  Compose([FaceToEdge(remove_faces=False), NormalizeUnitSphere()]))
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self) -> list:
        return ["training.pt", "test.pt"]

    @property
    @lru_cache(maxsize=32)
    def _segmentation_labels(self):
        """Extract segmentation labels."""
        path_to_labels = Path(self.root) / "MPI-FAUST"/ "segmentations.npz"
        seg_labels = np.load(str(path_to_labels))["segmentation_labels"]
        return torch.from_numpy(seg_labels).type(torch.int64)

    def _mesh_filenames(self):
        """Extract all mesh filenames."""
        path_to_meshes = Path(self.root)/ "MPI-FAUST" / "meshes"
        return path_to_meshes.glob("*.ply")

    def _unzip_dataset(self):
        """Extract dataset from zip."""
        path_to_zip = Path(self.root) / "MPI-FAUST.zip"
        extract_zip(str(path_to_zip), self.root, log=False)

    def process(self):
        """Process the raw meshes files and their corresponding class labels."""
        self._unzip_dataset()

        data_list = []
        for mesh_filename in sorted(self._mesh_filenames()):
            vertices, faces = load_mesh(mesh_filename)
            data = Data(pos=vertices, face=faces)
            data.segmentation_labels = self._segmentation_labels
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list[:80]), self.processed_paths[0])
        torch.save(self.collate(data_list[80:]), self.processed_paths[1])
        
# construct layers
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_conv_layers(channels: list, conv: MessagePassing, conv_params: dict):
    conv_layers = [
        conv(in_ch, out_ch, **conv_params) for in_ch, out_ch in pairwise(channels)
    ]
    return conv_layers

def get_mlp_layers(channels: list, activation, output_activation=nn.Identity):
    layers = []
    *intermediate_layer_definitions, final_layer_definition = pairwise(channels)

    for in_ch, out_ch in intermediate_layer_definitions:
        intermediate_layer = nn.Linear(in_ch, out_ch)
        layers += [intermediate_layer, activation()]

    layers += [nn.Linear(*final_layer_definition), output_activation()]
    return nn.Sequential(*layers)

class FeatureSteeredConvolution(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        with_self_loops: bool = True,
        num_heads: int=12,
    ):
        super().__init__(aggr="mean")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.with_self_loops = with_self_loops

        self.linear = torch.nn.Linear(
            in_features=in_channels,
            out_features=out_channels * num_heads,
            bias=False,
        )
        self.u = torch.nn.Linear(
            in_features=in_channels,
            out_features=num_heads,
            bias=False,
        )
        self.c = torch.nn.Parameter(torch.Tensor(num_heads))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialization of tuneable network parameters."""
        torch.nn.init.uniform_(self.linear.weight)
        torch.nn.init.uniform_(self.u.weight)
        torch.nn.init.normal_(self.c, mean=0.0, std=0.1)
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0.0, std=0.1)

    def forward(self, x, edge_index):
        if self.with_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=x.shape[0])

        out = self.propagate(edge_index, x=x)
        return out if self.bias is None else out + self.bias

    def _compute_attention_weights(self, x_i, x_j):
        assert x_j.shape[-1] == self.in_channels
        attention_logits = self.u(x_i - x_j) + self.c
        return F.softmax(attention_logits, dim=1)

    def message(self, x_i, x_j):
        attention_weights = self._compute_attention_weights(x_i, x_j)
        x_j = self.linear(x_j).view(-1, self.num_heads, self.out_channels)
        return (attention_weights.view(-1, self.num_heads, 1) * x_j).sum(dim=1)
    
class GraphFeatureEncoder(torch.nn.Module):
    """Graph neural network consisting of stacked graph convolutions."""
    def __init__(
        self,
        in_features,
        conv_channels,
        apply_batch_norm: int = True,
    ):
        super().__init__()
        self.apply_batch_norm = apply_batch_norm

        *first_conv_channels, final_conv_channel = conv_channels
        conv_layers = [FeatureSteeredConvolution(in_ch, out_ch) for in_ch, out_ch in pairwise([in_features] + conv_channels)]
        self.conv_layers = nn.ModuleList(conv_layers)

        self.batch_layers = [None for _ in first_conv_channels]
        if apply_batch_norm:
            self.batch_layers = nn.ModuleList(
                [nn.BatchNorm1d(channel) for channel in first_conv_channels]
            )

    def forward(self, x, edge_index):
        *first_conv_layers, final_conv_layer = self.conv_layers
        for conv_layer, batch_layer in zip(first_conv_layers, self.batch_layers):
            x = conv_layer(x, edge_index)
            x = F.relu(x)
            if batch_layer is not None:
                x = batch_layer(x)
        return final_conv_layer(x, edge_index)
    
class MeshSeg(torch.nn.Module):
    """Mesh segmentation network."""
    def __init__(
        self,
        in_features,
        encoder_features,
        conv_channels,
        encoder_channels,
        decoder_channels,
        num_classes,
        apply_batch_norm=True,
    ):
        super().__init__()
        self.input_encoder = get_mlp_layers(
            channels=[in_features] + encoder_channels,
            activation=nn.ReLU,
        )
        self.gnn = GraphFeatureEncoder(
            in_features=encoder_features,
            conv_channels=conv_channels,
            apply_batch_norm=apply_batch_norm,
        )
        *_, final_conv_channel = conv_channels

        self.final_projection = get_mlp_layers(
            [final_conv_channel] + decoder_channels + [num_classes],
            activation=nn.ReLU,
        )

    def forward(self, data):
        x, edge_index = data.pos, data.edge_index
        x = self.input_encoder(x)
        x = self.gnn(x, edge_index)
        return self.final_projection(x)
    
# misc

def train(net, train_data, optimizer, loss_fn, device):
    """Train network on training dataset."""
    net.train()
    cumulative_loss = 0.0
    for data in train_data:
        data = data.to(device)
        optimizer.zero_grad()
        out = net(data)
        loss = loss_fn(out, data.segmentation_labels.squeeze())
        loss.backward()
        cumulative_loss += loss.item()
        optimizer.step()
    return cumulative_loss / len(train_data)

def accuracy(predictions, gt_seg_labels):
    predicted_seg_labels = predictions.argmax(dim=-1, keepdim=True)
    if predicted_seg_labels.shape != gt_seg_labels.shape:
        raise ValueError("Expected Shapes to be equivalent")
    correct_assignments = (predicted_seg_labels == gt_seg_labels).sum()
    num_assignemnts = predicted_seg_labels.shape[0]
    return float(correct_assignments / num_assignemnts)

def evaluate_performance(dataset, net, device):
    prediction_accuracies = []
    for data in dataset:
        data = data.to(device)
        predictions = net(data)
        prediction_accuracies.append(accuracy(predictions, data.segmentation_labels))
    return sum(prediction_accuracies) / len(prediction_accuracies)

@torch.no_grad()
def test(net, train_data, test_data, device):
    net.eval()
    train_acc = evaluate_performance(train_data, net, device)
    test_acc = evaluate_performance(test_data, net, device)
    return train_acc, test_acc

model_params = dict(
    in_features=3,
    encoder_features=16,
    conv_channels=[32, 64, 128, 64],
    encoder_channels=[16],
    decoder_channels=[32],
    num_classes=12,
    apply_batch_norm=True,
)

def get_net():
    net = MeshSeg(**model_params).to(FLAGS.device)
    return net

def main_train(_):
    train_data = SegmentationFaust(
        root=FLAGS.root,
    )
    test_data = SegmentationFaust(
        root=FLAGS.root,
        train=False,
    )
    train_loader = DataLoader(train_data,  shuffle=True)
    test_loader = DataLoader(test_data, shuffle=False)
    lr = 0.001
    num_epochs = 100
    net = get_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_acc_table = []
    test_acc_table= []
    best_test_acc=0.0
    with tqdm(range(num_epochs), unit="Epoch") as tepochs:
        for epoch in tepochs:
            train_loss = train(net, train_loader, optimizer, loss_fn, FLAGS.device)
            train_acc, test_acc = test(net, train_loader, test_loader, FLAGS.device)
            
            tepochs.set_postfix(
                train_loss=train_loss,
                train_accuracy=100 * train_acc,
                test_accuracy=100 * test_acc,
            )
            sleep(0.1)
            train_acc_table.append(train_acc)
            test_acc_table.append(test_acc)
            np.savez('attention.npz', 
                        train_acc=np.array(train_acc_table), 
                        test_acc=np.array(test_acc_table))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(net.state_dict(), "checkpoint_attention.pth")
if __name__=="__main__":
    seed=123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    app.run(main_train)


