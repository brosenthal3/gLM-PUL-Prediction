import polars
import numpy as np
import matplotlib.pyplot as plt
import umap

embeddings = (
    polars.read_parquet("src/data/results/genecat/fold_data/fold_0_data.parquet")
    .select("embeddings", "label")
)
embedding_matrix = np.stack(embeddings.select("embeddings").to_series().to_list())
labels = embeddings.select("label").to_series().to_list()

reducer = umap.UMAP()
embedding_2d = reducer.fit_transform(embedding_matrix)

plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap="coolwarm", alpha=0.7, s=4)
#plt.colorbar()
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Non-PUL gene'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=5, label='PUL gene')
])
plt.xticks([])
plt.yticks([])
plt.title("UMAP projection of gene embeddings")
plt.savefig("src/data/plots/embedding_umap.png")
plt.clf()
