import polars
import numpy as np
import matplotlib.pyplot as plt
import umap

embeddings = (
    polars.read_parquet("src/data/results/genecat/fold_data/fold_0_data.parquet")
    .select("embeddings", "label")
    .sort("label")
)
embeddings_puls = embeddings.filter(polars.col("label") == True)
embeddings_non_puls = embeddings.filter(polars.col("label") == False)

colors = plt.cm.tab20.colors
for i, embeddings in enumerate([embeddings_non_puls, embeddings_puls]):
    embedding_matrix = np.stack(embeddings.select("embeddings").to_series().to_list())
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform(embedding_matrix)
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.4, s=3, color=colors[i])

plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=5, label='Non-PUL gene'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=5, label='PUL gene')
])
plt.xticks([])
plt.yticks([])
plt.title("UMAP projection of gene embeddings")
plt.savefig("src/data/plots/embedding_umap.png")
plt.clf()
