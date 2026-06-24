import os
import polars
import numpy as np
import matplotlib.pyplot as plt
import umap
from viz_data import model_names, Bold_10

def plot_embeddings_umap(
    embeddings_path="src/data/results/genecat_zeroshot_cazy/fold_data/fold_0_data.parquet", 
    save_path="results/plots/embedding_umap.png", 
    model_name="genecat_zeroshot_cazy",
    visualize_taxonomy=False
):
    cryptic_puls = polars.read_csv("src/data/data_collection/cryptic_puls_genes.tsv", separator="\t").with_columns(polars.lit(True).alias("cryptic"))
    tax_annotations = (
        polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t", infer_schema_length=600)
        .select("sequence_id", "phylum", "class", "order")
        .unique()
    )
    # check if umap results already saved
    umap_path = f"src/data/results/{model_name}/umap.parquet"
    if os.path.exists(umap_path):
        print("Found existing embeddings", flush=True)
        reduced_embeddings = polars.read_parquet(umap_path)
    else:
        print("Processing embeddings input...", flush=True)
        embeddings = (
            polars.read_parquet(embeddings_path)
            .select("embedding", "label", "protein_id", "sequence_id")
            .join(
                cryptic_puls,
                on="protein_id",
                how="left"
            )
            .sort("label")
            .join(
                tax_annotations,
                on="sequence_id",
                how="left"
            )
            .drop("sequence_id")
        )
        # convert to matrix
        embedding_matrix = np.stack(embeddings["embedding"].to_list())
        # fit umap
        print("Running UMAP...", flush=True)
        reducer = umap.UMAP(
            metric="euclidean",
            n_neighbors=12,
            verbose=True,
        )
        embedding_2d = reducer.fit_transform(embedding_matrix.astype("float32"))
        x = embedding_2d[:, 0]
        y = embedding_2d[:, 1]
        reduced_embeddings = polars.DataFrame({
            "x": x,
            "y": y,
            "label": embeddings["label"],
            "cryptic": embeddings["cryptic"],
            "phylum": embeddings["phylum"],
            "class": embeddings["class"],
            "protein_id": embeddings["protein_id"]
        })
        reduced_embeddings.write_parquet(umap_path)

    print("Plotting...")

    model_name = model_names.get(model_name, model_name)
    if visualize_taxonomy:
        colors = {i: c for i, c in enumerate(Bold_10)}
        for rank in ["phylum", "class"]:
            reduced_embeddings = (
                reduced_embeddings
                .join(
                    reduced_embeddings.group_by(rank).len(),
                    how="left",
                    on=rank
                )
                .with_columns(
                    polars.when(polars.col("len") < 5000)
                    .then(polars.lit("Other (<5000 genes)"))
                    .otherwise(polars.col(rank))
                    .alias(rank)
                )
            )

            handles = []
            labels = reduced_embeddings.select(rank, "len").unique().sort(by="len", descending=True)[rank].unique(maintain_order=True).to_list()
            for i, label in enumerate(labels):
                c = colors.get(i, "#808889")
                if label is None:
                    embedding_2d = reduced_embeddings.filter(polars.col(rank).is_null())
                else:
                    embedding_2d = reduced_embeddings.filter(polars.col(rank) == label)

                plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=0.2, color=c, alpha=0.1, label=label)
    
                if label is not None:
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=5, label=label))
        
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.xticks([])
            plt.yticks([])
            plt.title(f"UMAP of {model_name} embeddings (colored by {rank})")

            plt.legend(handles=handles)
            plt.tight_layout()
            plt.savefig(save_path+"_"+rank+".png", dpi=300)
            plt.close()

    else:
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
        plt.savefig(save_path, dpi=300)
        plt.close()

print("Running umap script...")

# # genecat zeroshot
# plot_embeddings_umap(
#     save_path="results/plots/UMAP/umap_genecat_zeroshot_cazy.png",
#     model_name="genecat_zeroshot_cazy"
# )

# # genecat finetuned
# plot_embeddings_umap(
#     embeddings_path="src/data/results/genecat_finetuned_cazy_masked/embeddings/fold_0_data.parquet", 
#     save_path="results/plots/UMAP/umap_genecat_finetuned_cazy.png", 
#     model_name="genecat_finetuned_cazy_masked"
# )

# # bacformer
# plot_embeddings_umap(
#     embeddings_path="src/data/results/bacformer/fold_data/fold_1_data.parquet",
#     save_path="results/plots/UMAP/umap_bacformer",
#     model_name="bacformer",
# )

# # ESM-C
# plot_embeddings_umap(
#     embeddings_path="src/data/results/esmc/fold_data/fold_0_data.parquet",
#     save_path="results/plots/UMAP/umap_esmc.png",
#     model_name="esmc"
# )

# # genecat_untrained
# plot_embeddings_umap(
#     embeddings_path="src/data/results/genecat_untrained/embeddings/fold_0_data.parquet",
#     save_path="results/plots/UMAP/umap_genecat_untrained.png",
#     model_name="genecat_untrained"
# )

# bacformer with taxonomy
plot_embeddings_umap(
    embeddings_path="src/data/results/bacformer/fold_data/fold_1_data.parquet",
    save_path="results/plots/UMAP/umap_bacformer",
    model_name="bacformer",
    visualize_taxonomy=True
)

# esm-c with taxonomy
plot_embeddings_umap(
    embeddings_path="src/data/results/esmc/fold_data/fold_0_data.parquet",
    save_path="results/plots/UMAP/umap_esmc",
    model_name="esmc",
    visualize_taxonomy=True
)
