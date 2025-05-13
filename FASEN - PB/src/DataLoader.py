from torch.utils.data import DataLoader
import torch


def get_dataloader():



    train_loader = DataLoader(train_set,
                              batch_size=model_config['batch_size'],
                              num_workers=model_config['num_workers'],
                              shuffle=False)
    test_loader = DataLoader(test_set, batch_size=model_config['batch_size'], shuffle=False)
    return train_loader, test_loader


class CsvDataset(Dataset):
    def __init__(self, dataset_name: str, data_dim: int, data_dir: str, mode: str = 'train'):
        super(CsvDataset, self).__init__()
        x = []
        labels = []
        path = os.path.join(data_dir, dataset_name + '.csv')
        with (open(path, 'r')) as data_from:
            csv_reader = csv.reader(data_from)
            for i in csv_reader:
                x.append(i[0:data_dim])  # 最后一列当标签
                labels.append(i[data_dim])

        for i in range(len(x)):  # 将特征和标签的字符串数据显式转换为浮点数。
            for j in range(data_dim):
                x[i][j] = float(x[i][j])
        for i in range(len(labels)):
            labels[i] = float(labels[i])

        data = np.array(x)  # 将特征和标签从列表形式转换为 numpy 数组，方便后续操作。
        target = np.array(labels)
        inlier_indices = np.where(target == 0)[0]  # 使用 np.where 获取标签为 0（内点）和 1（外点）样本的索引。
        outlier_inices = np.where(target == 1)[0]
        train_data, train_label, test_data, test_label = train_test_split(data[inlier_indices], data[outlier_inices])
        if mode == 'train':
            self.data = torch.Tensor(train_data)
            self.targets = torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)
        print(len(self.data))

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)


class MatDataset(Dataset):
    def __init__(self, dataset_name: str, data_dim: int, data_dir: str, mode: str = 'train'):
        super(MatDataset, self).__init__()
        path = os.path.join(data_dir, dataset_name + '.mat')
        data = io.loadmat(path)
        samples = data['X']
        labels = ((data['y']).astype(np.int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)
        if mode == 'train':
            self.data = torch.Tensor(train_data)
            self.targets = torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)


class NpzDataset(Dataset):
    def __init__(self, dataset_name: str, data_dim: int, data_dir: str, mode: str = 'train'):
        super(NpzDataset, self).__init__()
        path = os.path.join(data_dir, dataset_name + '.npz')
        data = np.load(path)
        samples = data['X']
        labels = ((data['y']).astype(np.int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        train_data, train_label, test_data, test_label = train_test_split(inliers, outliers)
        if mode == 'train':
            self.data = torch.Tensor(train_data)
            self.targets = torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)