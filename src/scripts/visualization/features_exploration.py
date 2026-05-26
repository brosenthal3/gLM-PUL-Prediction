import polars
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
from viz_data import Cork_7
from visualization_utilities import join_gene_and_PUL_table

cols = ["protein_id", "domain", "sequence_id"]
features_pfam = polars.read_parquet("src/data/genecat_output/pfam.features.parquet").select(cols).with_columns(polars.lit("pfam").alias("feature"))
features_cazy = polars.read_parquet("src/data/genecat_output/dbcan.features.parquet").select(cols).with_columns(polars.lit("cazy").alias("feature"))
selected_sequences = (
    polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t", infer_schema_length=600)
    .select("sequence_id")
    .unique()
)
all_puls = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t")
genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
genes_with_puls = join_gene_and_PUL_table(gene_table=genes, cluster_table=all_puls).filter("is_PUL").select("protein_id").unique()

genes_cazy = set(features_cazy.join(selected_sequences, on="sequence_id", how="semi").select("protein_id").to_series())
genes_pfam = set(features_pfam.join(selected_sequences, on="sequence_id", how="semi").select("protein_id").to_series())
only_cazy = polars.DataFrame({"protein_id":list(genes_cazy - genes_pfam)})
cazy_only_domains = features_cazy.join(only_cazy, on="protein_id", how="semi")
#print(cazy_only_domains.select("domain").to_series().value_counts().sort(by="count"))

only_cazy_puls = only_cazy.join(genes_with_puls, on="protein_id", how="inner").select("protein_id").n_unique()
print(f"Number of proteins with only CAZy annotations that are in PULs: {only_cazy_puls}")


fig, ax = plt.subplots(figsize=(5, 5))
# plot overlap
v = venn2(
    [genes_pfam, genes_cazy], 
    set_labels=(f'Pfam\n(n={len(genes_pfam)})', f'CAZy\n(n={len(genes_cazy)})'),
    set_colors=(Cork_7[0], Cork_7[1]),
    ax=ax,
    alpha=0.9
)
for patch in v.patches:
    patch.set_edgecolor("white")
    patch.set_linewidth(0.5)

for t in v.set_labels + v.subset_labels:
    if t:
        t.set_fontsize(10)
        if t.get_text() == "1304":
            t.set_position((t.get_position()[0]+0.065, t.get_position()[1]))
        if "CAZy" in t.get_text():
            t.set_position((t.get_position()[0]+0.1, t.get_position()[1]))

ax.set_title("Overlap in proteins with Pfam and/or CAZy annotations", fontsize=12, pad=2)
fig.tight_layout()
fig.savefig("results/plots/feature_venn.png")