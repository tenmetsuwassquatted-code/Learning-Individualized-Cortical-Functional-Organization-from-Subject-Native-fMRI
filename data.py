import utils
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp
import os.path
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pkl

"""
https://github.com/THUDM/GRAND-plus/blob/main/utils/make_dataset.py
"""
def col_normalize(mx):
    """Column-normalize sparse matrix"""
    scaler = StandardScaler()

    mx = scaler.fit_transform(mx)

    return mx

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

"""
https://github.com/THUDM/GRAND-plus/blob/main/utils/make_dataset.py
"""
def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    print(len(set(train_indices)), len(train_indices))
    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

class BrainSubjectDataset(torch.utils.data.Dataset):
    """
    Subject-level dataset for cross-subject brain graph learning.

    Each subject is one graph sample.

    Each .pt file contains:
        [
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test
        ]

    Important:
        Different subjects are allowed to have different numbers of nodes.
        Therefore, samples are NOT stacked into a single tensor.
    """

    def __init__(
        self,
        root,
        pattern="sub-*.pt",
        preload=False,
        force_all_nodes_train=True,
    ):
        """
        Parameters
        ----------
        root : str or Path
            Directory containing subject .pt files.

        pattern : str
            File matching pattern.

        preload : bool
            If False, load one subject only when __getitem__ is called.
            Recommended for large brain graphs.

        force_all_nodes_train : bool
            For self-supervised learning, use all nodes as training nodes.
        """
        super().__init__()

        self.root = os.path.abspath(os.path.expanduser(str(root)))
        self.pattern = pattern
        self.preload = preload
        self.force_all_nodes_train = force_all_nodes_train

        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"Brain dataset directory does not exist:\n{self.root}"
            )

        self.subject_files = sorted(
            [
                os.path.join(self.root, f)
                for f in os.listdir(self.root)
                if f.endswith(".pt")
                and f.startswith("sub-")
            ]
        )

        if len(self.subject_files) == 0:
            raise FileNotFoundError(
                f"No subject .pt files found in:\n{self.root}\n"
                f"Expected files such as sub-001.pt, sub-002.pt, ..."
            )

        # Subject IDs
        self.subject_ids = [
            os.path.splitext(os.path.basename(path))[0]
            for path in self.subject_files
        ]

        # Optional memory cache
        self._cache = {} if preload else None

        print("=" * 80)
        print("BrainSubjectDataset")
        print("=" * 80)
        print("Root:", self.root)
        print("Subjects:", len(self.subject_files))

        if len(self.subject_ids) > 0:
            print(
                "Subject range:",
                self.subject_ids[0],
                "->",
                self.subject_ids[-1]
            )

    def __len__(self):
        return len(self.subject_files)

    def _load_subject(self, idx):
        """
        Load one subject from disk.
        """

        if idx < 0:
            idx += len(self.subject_files)

        if idx < 0 or idx >= len(self.subject_files):
            raise IndexError(
                f"Subject index {idx} out of range "
                f"[0, {len(self.subject_files) - 1}]"
            )

        if self.preload and idx in self._cache:
            return self._cache[idx]

        path = self.subject_files[idx]

        data_list = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(data_list, (list, tuple)):
            raise TypeError(
                f"Expected list/tuple in {path}, "
                f"got {type(data_list)}"
            )

        if len(data_list) != 6:
            raise ValueError(
                f"Expected six elements in {path}, "
                f"got {len(data_list)}"
            )

        (
            adj,
            features,
            labels,
            idx_train,
            idx_val,
            idx_test,
        ) = data_list

        # --------------------------------------------------------
        # Basic validation
        # --------------------------------------------------------

        if not torch.is_tensor(features):
            features = torch.tensor(
                features,
                dtype=torch.float32
            )
        else:
            features = features.float()

        if features.ndim != 2:
            raise ValueError(
                f"{path}: features must be 2D, "
                f"got shape {features.shape}"
            )

        n_nodes = features.shape[0]
        feature_dim = features.shape[1]

        # Sparse adjacency
        if not adj.is_sparse:
            raise TypeError(
                f"{path}: adjacency must be a torch sparse tensor."
            )

        if tuple(adj.shape) != (n_nodes, n_nodes):
            raise ValueError(
                f"{path}: adjacency shape {tuple(adj.shape)} "
                f"does not match number of nodes {n_nodes}."
            )

        adj = adj.coalesce()

        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------

        if labels is None:
            labels = torch.full(
                (n_nodes,),
                -1,
                dtype=torch.long
            )
        else:
            labels = torch.as_tensor(labels)

        # Current brain dataset is self-supervised.
        # Therefore use all nodes as training nodes.
        if self.force_all_nodes_train:
            idx_train = torch.arange(
                n_nodes,
                dtype=torch.long
            )
        else:
            idx_train = torch.as_tensor(
                idx_train,
                dtype=torch.long
            )

        idx_val = torch.as_tensor(
            idx_val,
            dtype=torch.long
        )

        idx_test = torch.as_tensor(
            idx_test,
            dtype=torch.long
        )

        subject_id = self.subject_ids[idx]

        sample = {
            "subject": subject_id,
            "path": path,

            "adj": adj,
            "features": features,
            "labels": labels,

            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test,

            "num_nodes": n_nodes,
            "feature_dim": feature_dim,
        }

        if self.preload:
            self._cache[idx] = sample

        return sample

    def __getitem__(self, idx):
        return self._load_subject(idx)

    def get_subject_id(self, idx):
        return self.subject_ids[idx]

    def get_subject_path(self, idx):
        return self.subject_files[idx]

    def summary(self):
        """
        Print basic information for all subjects without
        permanently loading all graphs into memory.
        """

        print("\n" + "=" * 80)
        print("Brain dataset summary")
        print("=" * 80)

        for i, path in enumerate(self.subject_files):

            data_list = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            adj, features, labels, \
                idx_train, idx_val, idx_test = data_list

            print(
                f"{i:4d} | "
                f"{self.subject_ids[i]:12s} | "
                f"N={features.shape[0]:8d} | "
                f"F={features.shape[1]:4d} | "
                f"edges={adj._nnz():9d}"
            )

def get_dataset(dataset, pe_dim, split_seed=0):
    file_dir = 'dataset/'
    
    if dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics", "cora", "citeseer"}:
        # 直接从DGL数据集创建，避免依赖.pt文件
        if dataset == "pubmed":
            dgl_dataset = PubmedGraphDataset()
        elif dataset == "corafull":
            dgl_dataset = CoraFullDataset()
        elif dataset == "computer":
            dgl_dataset = AmazonCoBuyComputerDataset()
        elif dataset == "photo":
            dgl_dataset = AmazonCoBuyPhotoDataset()
        elif dataset == "cs":
            dgl_dataset = CoauthorCSDataset()
        elif dataset == "physics":
            dgl_dataset = CoauthorPhysicsDataset()
        elif dataset == "cora":
            dgl_dataset = CoraGraphDataset()
        elif dataset == "citeseer":
            dgl_dataset = CiteseerGraphDataset()
        
        graph = dgl_dataset[0]
        
        # 提取特征和标签
        features = graph.ndata['feat']
        labels = graph.ndata['label']
        
        # 获取邻接矩阵（DGL 2.4.0+兼容方式）
        # 使用edges方法获取边，然后构建邻接矩阵
        src, dst = graph.edges()
        num_nodes = graph.number_of_nodes()
        adj_array = np.zeros((num_nodes, num_nodes))
        adj_array[src.numpy(), dst.numpy()] = 1
        adj_sparse = sp.coo_matrix(adj_array)
        
        # 转换为与原始代码兼容的稀疏张量格式
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj_sparse)
        
        # 获取或创建训练/验证/测试划分
        if 'train_mask' in graph.ndata:
            idx_train = torch.where(graph.ndata['train_mask'])[0]
            idx_val = torch.where(graph.ndata['val_mask'])[0]
            idx_test = torch.where(graph.ndata['test_mask'])[0]
        else:
            # 如果没有预定义划分，使用随机划分
            num_nodes = graph.number_of_nodes()
            indices = np.random.RandomState(split_seed).permutation(num_nodes)
            
            train_size = int(0.6 * num_nodes)
            val_size = int(0.2 * num_nodes)
            
            idx_train = torch.tensor(indices[:train_size], dtype=torch.long)
            idx_val = torch.tensor(indices[train_size:train_size+val_size], dtype=torch.long)
            idx_test = torch.tensor(indices[train_size+val_size:], dtype=torch.long)
        
        # 确保图是双向的（与原代码一致）
        graph = dgl.to_bidirected(graph)
        
        # 计算拉普拉斯位置编码（使用utils中的函数）
        lpe = utils.laplacian_positional_encoding(graph, pe_dim)
        
        # 拼接特征和位置编码
        features = torch.cat((features, lpe), dim=1)

    elif dataset in {"aminer", "reddit", "Amazon2M"}:
        # 保留原代码逻辑处理大型数据集
        file_path = file_dir + dataset + '.pt'
        if os.path.exists(file_path):
            data_list = torch.load(file_path)

            #adj, features, labels, idx_train, idx_val, idx_test
            adj = data_list[0]
            features = data_list[1]
            labels = data_list[2]
            idx_train = data_list[3]
            idx_val = data_list[4]
            idx_test = data_list[5]
        else:
            # 从原始文件加载
            if dataset == 'aminer':
                adj = pkl.load(open(os.path.join(file_dir, "{}.adj.sp.pkl".format(dataset)), "rb"))
                features = pkl.load(open(os.path.join(file_dir, "{}.features.pkl".format(dataset)), "rb"))
                labels = pkl.load(open(os.path.join(file_dir, "{}.labels.pkl".format(dataset)), "rb"))
                random_state = np.random.RandomState(split_seed)
                idx_train, idx_val, idx_test = get_train_val_test_split(
                    random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
                idx_unlabel = np.concatenate((idx_val, idx_test))
                features = col_normalize(features)
            elif dataset in ['reddit']:
                adj = sp.load_npz(os.path.join(file_dir, '{}_adj.npz'.format(dataset)))
                features = np.load(os.path.join(file_dir, '{}_feat.npy'.format(dataset)))
                labels = np.load(os.path.join(file_dir, '{}_labels.npy'.format(dataset))) 
                print(labels.shape, list(np.sum(labels, axis=0)))
                random_state = np.random.RandomState(split_seed)
                idx_train, idx_val, idx_test = get_train_val_test_split(
                    random_state, labels, train_examples_per_class=20, val_examples_per_class=30)    
                idx_unlabel = np.concatenate((idx_val, idx_test))
                print(dataset, features.shape)
            
            elif dataset in ['Amazon2M']:
                adj = sp.load_npz(os.path.join(file_dir, '{}_adj.npz'.format(dataset)))
                features = np.load(os.path.join(file_dir, '{}_feat.npy'.format(dataset)))
                labels = np.load(os.path.join(file_dir, '{}_labels.npy'.format(dataset)))
                print(labels.shape, list(np.sum(labels, axis=0)))
                random_state = np.random.RandomState(split_seed)
                class_num = labels.shape[1]
                idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20*class_num, val_size=30 * class_num)
                idx_unlabel = np.concatenate((idx_val, idx_test))

            # 对邻接矩阵进行归一化处理
            adj = adj + sp.eye(adj.shape[0])
            D1 = np.array(adj.sum(axis=1))**(-0.5)
            D2 = np.array(adj.sum(axis=0))**(-0.5)
            D1 = sp.diags(D1[:, 0], format='csr')
            D2 = sp.diags(D2[0, :], format='csr')

            A = adj.dot(D1)
            A = D2.dot(A)
            adj = A

        # 转换为PyTorch张量
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels)
        idx_train = torch.tensor(idx_train)
        idx_val = torch.tensor(idx_val)
        idx_test = torch.tensor(idx_test)

        # 创建图并计算位置编码
        graph = dgl.from_scipy(adj)
        lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
    
        # 拼接特征和位置编码
        features = torch.cat((features, lpe), dim=1)

        # 将邻接矩阵转换为稀疏张量
        adj = utils.sparse_mx_to_torch_sparse_tensor(adj)

        # 如果是多分类问题，取最大索引作为标签
        if len(labels.shape) > 1:
            labels = torch.argmax(labels, -1)
    elif dataset == "brain":

        brain_dir = os.path.join(file_dir, "brain")

        brain_dataset = BrainSubjectDataset(
            root=brain_dir,
            preload=False,
            force_all_nodes_train=True,
        )

        return brain_dataset
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

    return adj, features, labels, idx_train, idx_val, idx_test


if __name__ == "__main__":

    brain_dataset = get_dataset(
        "brain",
        pe_dim=16
    )

    print("\nDataset length:")
    print(len(brain_dataset))

    print("\nSubject IDs:")
    print(brain_dataset.subject_ids)

    print("\nLoading first subject...")

    sample = brain_dataset[0]

    print("Subject:", sample["subject"])
    print("Path:", sample["path"])
    print("Adjacency:", sample["adj"])
    print("Features:", sample["features"].shape)
    print("Labels:", sample["labels"].shape)
    print("idx_train:", sample["idx_train"].shape)
    print("idx_val:", sample["idx_val"].shape)
    print("idx_test:", sample["idx_test"].shape)
    print("Num nodes:", sample["num_nodes"])
    print("Feature dim:", sample["feature_dim"])

    for i in range(len(brain_dataset)):

        sample = brain_dataset[i]

        print(
            sample["subject"],
            "N =", sample["num_nodes"],
            "F =", sample["feature_dim"],
            "adj =", tuple(sample["adj"].shape)
        )