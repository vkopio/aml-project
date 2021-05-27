import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from common import train_data, train_target


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(train_data)

principalDf = pd.DataFrame(
    data=principalComponents,
    columns=['pc1', 'pc2'],
)

principalDf['target'] = train_target

sns.scatterplot(data=principalDf, x='pc1', y='pc2', hue='target', alpha=0.3)
plt.show()

pca_full = PCA()
pca_full.fit_transform(train_data)

plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
