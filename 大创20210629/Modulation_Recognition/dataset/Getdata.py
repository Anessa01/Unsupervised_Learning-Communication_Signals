import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np

testsz = 100
modulationlabels = {0: "QPSK", 1: "PAM4", 2: "AM-DSB", 3: "GFSK", 4: "QAM64", 5: "AM-SSB", 6: "8PSK", 7: "QAM16", 8: "WBFM", 9: "CPFSK", 10: "BPSK"}


def pkl2numpy(filename):
    f = open(filename, 'rb')
    f.seek(0)
    data = pickle.load(f, encoding='latin1')
    return data


def get_data():
    fname = "dataset/RML2016.10a_dict.pkl"
    return pkl2numpy(fname)

def get_smallset():
    fname = "dataset/RML2016.10a_dict.pkl"
    Data = pkl2numpy(fname)
    dataset = {"data": np.empty([0, 2, 128]), "label": np.empty([0])}
    for i in range(10, 20, 2):
        for j in range(11):
            size = 100
            dataset["data"] = np.append(dataset["data"], Data[modulationlabels[j], i][0: size, :, :], axis=0)
            dataset["label"] = np.append(dataset["label"], np.ones([size]) * j)
    return dataset

def get_dataset():
    fname = "dataset/RML2016.10a_dict.pkl"
    Data = pkl2numpy(fname)
    dataset = {"data": np.empty([0, 2, 128]), "label": np.empty([0])}
    for i in range(10, 20, 2):
        for j in range(11):
            size = Data[modulationlabels[j], i].shape[0] - testsz
            dataset["data"] = np.append(dataset["data"], Data[modulationlabels[j], i][0:size, :, :], axis=0)
            dataset["label"] = np.append(dataset["label"], np.ones([size]) * j)
    return dataset

def get_testset():
    fname = "dataset/RML2016.10a_dict.pkl"
    Data = pkl2numpy(fname)
    dataset = {"data": np.empty([0, 2, 128]), "label": np.empty([0])}
    for i in range(-20, 20, 2):
        for j in range(11):
            size = Data[modulationlabels[j], i].shape[0] - testsz
            dataset["data"] = np.append(dataset["data"], Data[modulationlabels[j], i][size:testsz+size, :, :], axis = 0)
            dataset["label"] = np.append(dataset["label"], np.ones([testsz]) * j)
    return dataset


class RMLdataset(Dataset):
    def __init__(self):
        self.dataset = get_dataset()

    def __len__(self):
        return self.dataset["data"].shape[0]

    def __getitem__(self, idx):
        sample = {"data": self.dataset["data"][idx], "label": self.dataset["label"][idx]}
        return sample

class RMLtestset(Dataset):
    def __init__(self):
        self.dataset = get_testset()

    def __len__(self):
        return self.dataset["data"].shape[0]

    def __getitem__(self, idx):
        sample = {"data": self.dataset["data"][idx], "label": self.dataset["label"][idx]}
        return sample

class RMLsmallset(Dataset):
    def __init__(self):
        self.dataset = get_smallset()

    def __len__(self):
        return self.dataset["data"].shape[0]

    def __getitem__(self, idx):
        sample = {"data": self.dataset["data"][idx], "label": self.dataset["label"][idx]}
        return sample

