import pickle


def pkl2numpy(filename):
    f = open(filename, 'rb')
    f.seek(0)
    data = pickle.load(f, encoding='latin1')
    return data


def get_dataset():
    fname = "dataset/RML2016.10a_dict.pkl"
    return pkl2numpy(fname)
