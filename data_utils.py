import dask.dataframe as dd
import os.path


class DataSet:
    train_data = []
    test_data = []
    ntrain = 0
    ntest = 0
    sample_rate = 1.0

    def copy(self):
        c = DataSet()
        c.train_data = self.train_data.copy()
        c.test_data = self.test_data.copy()
        c.ntrain = self.ntrain
        c.ntest = self.ntest
        c.sample_rate = self.sample_rate
        return c

def load_from(train_file, test_file):
    data = DataSet()
    data.train_data = dd.read_csv(train_file)
    data.test_data = dd.read_csv(test_file)

    if train_file == 'data/all_train.csv':
        data.ntrain = 7000000
    else:
        data.ntrain = len(data.train_data)

    if test_file == 'data/all_test.csv':
        data.ntest = 3500000
    else:
        data.ntest = len(data.test_data)

    return data

def generate_sample_set(r=0.001):
    data = dd.read_csv('data/all_train.csv')
    data = data.sample(r)
    with open('data/all_sample.csv', 'w') as file:
        data.compute().to_csv(file, index=False)

def load_sample_data(sample_file):
    sample_rate = 0.001

    if not os.path.isfile(sample_file):
        generate_sample_set(sample_rate)

    data = DataSet()
    data.train_data = dd.read_csv('data/all_sample.csv')
    data.test_data = data.train_data
    data.ntrain = int(7000000 * sample_rate)
    data.ntest = data.ntrain
    data.sample_rate = sample_rate

    return data

def subsample_data(data, r, output_data):
    output_data.train_data = data.train_data.sample(r)
    output_data.test_data = data.test_data.sample(r)
    output_data.ntrain = int(data.ntrain * r)
    output_data.ntest = int(data.ntest * r)
    output_data.sample_rate = r
