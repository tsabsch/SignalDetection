import dask.dataframe as dd

def generate_sample_set(r=0.001):
    data = dd.read_csv('data/all_train.csv')
    data = data.sample(r)
    file = open('data/all_sample.csv', 'w')
    data.compute().to_csv(file, index=False)
