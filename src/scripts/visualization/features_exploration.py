import polars
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

cols = ["protein_id", "domain"]
features_pfam = polars.read_parquet("src/data/genecat_output/pfam.features.parquet").select(cols).with_columns(polars.lit("pfam").alias("feature"))
features_cazy = polars.read_parquet("src/data/genecat_output/dbcan.features.parquet").select(cols).with_columns(polars.lit("cazy").alias("feature"))

genes_cazy = set(features_cazy.select("protein_id").to_series())
genes_pfam = set(features_pfam.select("protein_id").to_series())
only_cazy = polars.DataFrame({"protein_id":list(genes_cazy - genes_pfam)})
cazy_only_domains = features_cazy.join(only_cazy, on="protein_id", how="semi")
print(cazy_only_domains.select("domain").to_series().value_counts().sort(by="count"))

fig, ax = plt.subplots()
# plot overlap
v = venn2(
    [genes_pfam, genes_cazy], 
    set_labels=('Pfam', 'CAZy'), 
    ax=ax
)
for t in v.set_labels + v.subset_labels:
    if t:
        t.set_fontsize(9)

ax.set_title("Overlap in Pfam and CAZy annotations for each gene")
fig.savefig("results/plots/feature_venn.png")