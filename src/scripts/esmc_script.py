from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.sdk import batch_executor

from Bio.SeqIO.FastaIO import SimpleFastaParser

import os
import polars
from tqdm import tqdm
import torch

faa_path = "src/data/genecat_output/genome.genes.faa"
sequences = polars.read_csv("src/data/results/cblaster_results.tsv", separator="\t", infer_schema_length=700).select("sequence_id").unique()
genes = (
    polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
    .select("protein_id", "sequence_id").unique()
    .join(sequences, on="sequence_id", how="semi")
    .select("protein_id")
    .unique().to_series().to_list()
)

proteins = []
protein_ids = []
with open(faa_path, "r") as f:
    faa_iter = SimpleFastaParser(f)
    for gene_id, seq in tqdm(faa_iter, total=len(genes), desc="Reading FASTA genes"):
        if gene_id not in sequences:
            continue

        proteins.append(seq)
        protein_ids.append(gene_id)

def embed_sequence(client, gene_id, seq):
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    embedding = logits_output.embeddings.mean(dim=1).squeeze(0).type(torch.float32).cpu().numpy()

    return gene_id, len(seq), embedding


client = ESMC.from_pretrained("esmc_300m").to("cuda") # or "cpu"
with batch_executor() as executor:
    outputs = executor.execute_batch(user_func=embed_sequence, client=client, seq=proteins, gene_id=protein_ids)


emb_series = polars.Series("embeddings", [output[2] for output in outputs])
seq_length_series = polars.Series("protein_length", [output[1] for output in outputs])
gene_series = polars.Series("protein_id", [output[0] for output in outputs])
df = polars.DataFrame([gene_series, seq_length_series, emb_series])
os.makedirs("src/data/results/esmc", exist_ok=True)
print(df)
print(df.schema)
df.write_parquet("src/data/results/esmc/esmc_embeddings.parquet")
