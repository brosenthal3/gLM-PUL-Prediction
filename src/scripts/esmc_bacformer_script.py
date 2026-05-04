import multiprocessing as mp
mp.set_start_method("fork", force=True)

from Bio import SeqIO
import os
import polars
from tqdm import tqdm
import torch
from transformers import AutoModel, logging
from bacformer.pp import protein_seqs_to_bacformer_inputs

"""
pip3 install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu126
"""

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.set_verbosity_error()
model = AutoModel.from_pretrained("macwiatrak/bacformer-large-masked-MAG", trust_remote_code=True).to(device).eval().to(torch.bfloat16)

output_path = "src/data/embeddings/esmc_bacformer_embeddings"
os.makedirs(output_path, exist_ok=True)
faa_path = "src/data/genecat_output/genome.genes.faa"
sequences = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t", infer_schema_length=700).select("sequence_id").unique()
genes_df = (
    polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
    .select("protein_id", "sequence_id").unique()
    .join(sequences, on="sequence_id", how="semi")
    .group_by("sequence_id")
    .agg(polars.col("protein_id"))
)
genes_dict = dict(zip(genes_df["sequence_id"], genes_df["protein_id"].to_list()))

def write_genes_fasta():
    os.makedirs("src/data/genecat_output/genes", exist_ok=True)
    all_genes = SeqIO.index(faa_path, "fasta")
    
    for contig in tqdm(genes_dict.keys()):
        genes = genes_dict.get(contig)
        out_file = f"src/data/genecat_output/genes/{contig}.faa"
        if os.path.exists(out_file):
            continue

        genes_faa = []
        for gene in genes:
            gene = all_genes[gene]
            genes_faa.append(gene)

        with open(out_file, "w") as handle:
            SeqIO.write(genes_faa, handle, "fasta")


def sliding_window_bacformer(
    model,
    bacformer_input,
    window_size=3000,
    stride=2000,
):
    """
    Run Bacformer in sliding-window mode to avoid NaNs on long genomes.
    """
    protein_embs = bacformer_input["protein_embeddings"].to(torch.float32)
    attention_mask = bacformer_input["attention_mask"]
    contig_ids = bacformer_input["contig_ids"]

    N = protein_embs.shape[1]

    collected = torch.zeros_like(protein_embs, dtype=torch.float32, device=device)
    counts = torch.zeros((1, N, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        for start in range(0, N, stride):
            end = min(start + window_size, N)

            window_input = {
                "protein_embeddings": protein_embs[:, start:end],
                "attention_mask": attention_mask[:, start:end],
                "contig_ids": contig_ids[:, start:end],
            }

            outputs = model(**window_input, return_dict=True)
            window_out = outputs["last_hidden_state"].to(torch.float32)

            # Skip bad windows (NaNs)
            if torch.isnan(window_out).any():
                print(f"[WARNING] NaNs in window {start}:{end}, skipping")
                continue

            collected[:, start:end] += window_out
            counts[:, start:end] += 1

    # Avoid division by zero
    counts[counts == 0] = 1

    result = (collected / counts).squeeze(0).cpu().numpy()
    return result


# get embeddings per contig
for contig in tqdm(os.listdir("src/data/genecat_output/genes")):
    proteins = []
    protein_ids = []
    genes_file = f"src/data/genecat_output/genes/{contig}"
    contig_genes_index = SeqIO.index(genes_file, "fasta")
    contig_genes_list = genes_dict.get(contig.split(".")[0])
    save_path = f"{output_path}/{contig.replace('faa', 'parquet')}"
    if os.path.exists(save_path):
        existing_embeddings = polars.read_parquet(save_path)
        if not existing_embeddings["embedding_bacformer"][0].is_nan().any():
            continue
        else:
            print(f"Found NaN values in embedding for {contig}, {existing_embeddings['embedding_bacformer'].shape}")

    for gene_id in contig_genes_list:
        seq = str(contig_genes_index[gene_id])
        proteins.append(seq)
        protein_ids.append(gene_id)

    if len(proteins) == 0:
        continue
    print(len(proteins), " proteins to process...")

    # embed the proteins with ESM-C to get average protein embeddings
    bacformer_input = protein_seqs_to_bacformer_inputs(
        proteins,
        device=device,
        batch_size=128,  # the batch size for computing the protein embeddings
        max_n_proteins=9000, # increased to 9000 due to some very long genomes
        bacformer_model_type="large", # Bacformer Large 300M
    )
    print(bacformer_input["protein_embeddings"].shape)
    embs = bacformer_input["protein_embeddings"].squeeze(0).to(torch.float32).detach().cpu().numpy()
    assert len(embs) == len(proteins), "not all proteins got embeddings, something went wrong."
    
    # save ESM-C embeddings for later logistic regression and comparison
    embeddings_df_dict = {"protein_id": [], "embedding_esmc": []}
    for i, embedding in enumerate(embs):
        embeddings_df_dict["protein_id"].append(protein_ids[i])
        embeddings_df_dict["embedding_esmc"].append(embedding)
    esmc_df = polars.DataFrame(embeddings_df_dict).sort("protein_id")

    # compute contextualised protein embeddings with Bacformer
    model = model.to(torch.float32)
    bacformer_embs = sliding_window_bacformer(
        model,
        bacformer_input,
        window_size=2000,
        stride=1500,
    )
    # outputs["last_hidden_state"] will be of shape(batch_size, n, 960), 960 is emb dim.
#    bacformer_embs = outputs["last_hidden_state"].squeeze(0).to(torch.float32).detach().cpu().numpy()

    bacformer_embeddings_df_dict = {"protein_id": [], "embedding_bacformer": []}
    # save bacformer embeddings for comparison
    for i, embedding in enumerate(bacformer_embs):
        bacformer_embeddings_df_dict["protein_id"].append(protein_ids[i])
        bacformer_embeddings_df_dict["embedding_bacformer"].append(embedding)
    bacformer_df = polars.DataFrame(bacformer_embeddings_df_dict).sort("protein_id")

    joined_df = esmc_df.join(bacformer_df, on="protein_id", how="inner")
    print(joined_df)
    joined_df.write_parquet(save_path)
