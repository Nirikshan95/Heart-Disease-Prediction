import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def apply_pca(features):
    ss=StandardScaler()
    scaled_features=ss.fit_transform(features)
    pca=PCA(n_components=4)
    principal_components=pca.fit_transform(scaled_features)
    return principal_components,pca