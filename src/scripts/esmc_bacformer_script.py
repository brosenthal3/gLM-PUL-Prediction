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

device = "cuda:0"
logging.set_verbosity_error()
model = AutoModel.from_pretrained("macwiatrak/bacformer-large-masked-MAG", trust_remote_code=True).to(device).eval().to(torch.bfloat16)

output_path = "src/data/embeddings/esmc_bacformer_embeddings"
os.makedirs(output_path, exist_ok=True)
faa_path = "src/data/genecat_output/genome.genes.faa"
sequences = polars.read_csv("src/data/data_collection/cblaster_results.tsv", separator="\t", infer_schema_length=700).select("sequence_id").unique()
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

# get embeddings per contig
for contig in tqdm(os.listdir("src/data/genecat_output/genes")):
    proteins = []
    protein_ids = []
    genes_file = f"src/data/genecat_output/genes/{contig}"
    contig_genes_index = SeqIO.index(genes_file, "fasta")
    contig_genes_list = genes_dict.get(contig.split(".")[0])
    save_path = f"{output_path}/{contig.replace('faa', 'parquet')}"
    if os.path.exists(save_path):
        print(f"Found embeddings for {contig}, continuing")
        continue

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
    with torch.no_grad():
        outputs = model(**bacformer_input, return_dict=True)

    # outputs["last_hidden_state"] will be of shape(batch_size, n, 960), 960 is emb dim.
    bacformer_embs = outputs["last_hidden_state"].squeeze(0).to(torch.float32).detach().cpu().numpy()

    bacformer_embeddings_df_dict = {"protein_id": [], "embedding_bacformer": []}
    # save bacformer embeddings for comparison
    for i, embedding in enumerate(bacformer_embs):
        bacformer_embeddings_df_dict["protein_id"].append(protein_ids[i])
        bacformer_embeddings_df_dict["embedding_bacformer"].append(embedding)
    bacformer_df = polars.DataFrame(bacformer_embeddings_df_dict).sort("protein_id")

    joined_df = esmc_df.join(bacformer_df, on="protein_id", how="inner")
    print(joined_df)
    joined_df.write_parquet(save_path)
