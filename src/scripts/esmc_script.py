from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from Bio.SeqIO.FastaIO import SimpleFastaParser

import os
import polars
from tqdm import tqdm
import torch

faa_path = "src/data/genecat_output/genome.genes.faa"
#sequences = polars.read_csv("src/data/results/cblaster_results.tsv", separator="\t", infer_schema_length=700).select("sequence_id").unique().to_series().to_list()
client = ESMC.from_pretrained("esmc_300m").to("cpu") # or "cpu"

with open(faa_path, "r") as f:
    faa_iter = SimpleFastaParser(f)

    protein_ids = []
    seq_length = []
    embeddings = []

    for gene_id, seq in tqdm(faa_iter, total=772451):
#        if gene_id.split("_")[0] not in sequences:
#            continue

        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)
        # NOTE that the track is sequence - there are other tracks too!
        # NOTE that return_mean_embedding doesnt seem to work. still get None
        logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        protein_ids.append(gene_id)
        seq_length.append(len(seq))
        embeddings.append(logits_output.embeddings.mean(dim=1).squeeze(0).type(torch.float32).cpu().numpy())

emb_series = polars.Series("embeddings", embeddings)
seq_length_series = polars.Series("protein_length", seq_length)
gene_series = polars.Series("protein_id", protein_ids)
df = polars.DataFrame([gene_series, seq_length_series, emb_series])
os.makedirs("src/data/results/esmc", exist_ok=True)
print(df)
print(df.schema)
df.write_parquet("src/data/results/esmc/esmc_embeddings.parquet")
