import polars
import numpy as np
import matplotlib.pyplot as plt
import umap

def plot_embeddings_umap(embeddings_path="src/data/results/genecat_zeroshot_cazy/fold_data/fold_0_data.parquet", save_path="results/plots/embedding_umap.png", model_name="genecat_zeroshot_cazy"):
    cryptic_puls = polars.read_csv("src/data/data_collection/cryptic_puls_genes.tsv", separator="\t").with_columns(polars.lit(True).alias("cryptic"))
    embeddings = (
        polars.read_parquet(embeddings_path)
        .select("embedding", "label", "protein_id")
        .join(
            cryptic_puls,
            on="protein_id",
            how="left"
        )
        .sort("label")
    )
    # convert to matrix
    embedding_matrix = np.stack(embeddings["embedding"].to_list())
    # fit umap
    print("Running UMAP...")
    reducer = umap.UMAP(
        metric="cosine",
        n_epochs=150,
        n_neighbors=10,
        verbose=True,
        n_jobs=-1
    )
    embedding_2d = reducer.fit_transform(embedding_matrix.astype("float32"))
    x = embedding_2d[:, 0]
    y = embedding_2d[:, 1]
    reduced_embeddings = polars.DataFrame({
        "x": x,
        "y": y,
        "label": embeddings["label"],
        "cryptic": embeddings["cryptic"]
    })

    print("Plotting...")
    colors = plt.cm.tab10.colors
    for i, label in enumerate([False, True]):
        embedding_2d = reduced_embeddings.filter(polars.col("label") == label)
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.5, s=1, color=colors[i])
        # add cryptic PULs
        if label == True:
            embedding_2d = reduced_embeddings.filter(polars.col("cryptic") == label)
            plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.35, s=1, color=colors[2])


    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=5, label='PUL gene'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=5, label='Cryptic PUL gene'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=5, label='Non-PUL gene')
    ])
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.xticks([])
    plt.yticks([])
    plt.title(f"UMAP projection of {model_name} embeddings")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # # genecat zeroshot
    # plot_embeddings_umap(
    #     save_path="results/plots/umap_genecat_zeroshot_cazy.png",
    #     model_name="GeneCAT Zeroshot (Pfam+CAZy)"
    # )

    # # genecat finetuned
    # plot_embeddings_umap(
    #     embeddings_path="src/data/results/genecat_finetuned_cazy_masked/embeddings/fold_0_data.parquet", 
    #     save_path="results/plots/umap_genecat_finetuned_cazy.png", 
    #     model_name="GeneCAT Finetuned (Pfam+CAZy)"
    # )

    # # bacformer
    # plot_embeddings_umap(
    #     embeddings_path="src/data/results/bacformer/fold_data/fold_0_data.parquet",
    #     save_path="results/plots/umap_bacformer.png",
    #     model_name="Bacformer"
    # )

    # # ESM-C
    # plot_embeddings_umap(
    #     embeddings_path="src/data/results/esmc/fold_data/fold_0_data.parquet",
    #     save_path="results/plots/umap_esmc.png",
    #     model_name="ESM-C"
    # )

    # genecat_untrained
    plot_embeddings_umap(
        embeddings_path="src/data/results/genecat_untrained/embeddings/fold_0_data.parquet",
        save_path="results/plots/umap_genecat_untrained.png",
        model_name="GeneCAT Untrained & Finetuned (Pfam+CAZy)"
    )