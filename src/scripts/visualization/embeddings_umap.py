import polars
import numpy as np
import matplotlib.pyplot as plt
import umap

embeddings = (
    polars.read_parquet("src/data/results/genecat/fold_data/fold_0_data.parquet")
    .select("embeddings", "label")
    .sort("label")
)
# convert to matrix
embedding_matrix = np.stack(embeddings["embeddings"].to_list())
# fit umap
reducer = umap.UMAP()
embedding_2d = reducer.fit_transform(embedding_matrix)
x = embedding_2d[:, 0]
y = embedding_2d[:, 1]
reduced_embeddings = polars.DataFrame({
    "x": x,
    "y": y,
    "label": embeddings["label"]
})

colors = plt.cm.tab20.colors
for i, label in enumerate([False, True]):
    embedding_2d = reduced_embeddings.filter(polars.col("label") == label)
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.5, s=1, color=colors[i])

plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=5, label='PUL gene'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=5, label='Non-PUL gene')
])
plt.xticks([])
plt.yticks([])
plt.title("UMAP projection of gene embeddings")
plt.savefig("src/results/plots/genecat/embedding_umap.png")
plt.clf()
