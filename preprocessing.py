import sklearn.decomposition as skdecomp

def perform_pca(data, n):
    pca = skdecomp.PCA(n_components=n)
    features = data.drop('# label', axis=1).compute()
    pca.fit(features)
    print('PCA with {} principal components computed.'.format(n))
    return pca
