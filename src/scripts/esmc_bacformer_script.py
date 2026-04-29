import multiprocessing as mp
mp.set_start_method("fork", force=True)

from Bio import SeqIO
import os
import polars
from tqdm import tqdm
import torch
from transformers import AutoModel
from bacformer.pp import protein_seqs_to_bacformer_inputs

"""
pip3 install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu126
"""

device = "cuda:0"
model = AutoModel.from_pretrained("macwiatrak/bacformer-large-masked-MAG", trust_remote_code=True).to(device).eval().to(torch.bfloat16)

output_path = "src/data/results/esmc/esmc_embeddings"
os.makedirs(output_path, exist_ok=True)
faa_path = "src/data/genecat_output/genome.genes.faa"
sequences = polars.read_csv("src/data/results/cblaster_results.tsv", separator="\t", infer_schema_length=700).select("sequence_id").unique()
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
for contig in tqdm(os.listdir("src/data/genecat_output/genes")[:3]):
    proteins = []
    protein_ids = []
    genes_file = f"src/data/genecat_output/genes/{contig}"
    contig_genes_index = SeqIO.index(genes_file, "fasta")
    contig_genes_list = genes_dict.get(contig.split(".")[0])
    save_path = f"{output_path}/{contig.replace('faa', 'parquet')}"
    # if os.path.exists(save_path):
    #     print(f"Found embeddings for {contig}, continuing")
    #     continue

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
        max_n_proteins=6000,  # the maximum number of proteins Bacformer was trained with
        bacformer_model_type="large", # Bacformer Large 300M
    )
    embs = bacformer_input["protein_embeddings"][0].detach().cpu().numpy()
    assert len(embs) == len(proteins), "not all proteins got embeddings, something went wrong."
    
    # save ESM-C embeddings for later logistic regression and comparison
    embeddings_df_dict = {"protein_id": [], "embedding_esmc": [], "embedding_bacformer": []}
    for i, embedding in enumerate(embs):
        embeddings_df_dict["protein_id"].append(protein_ids[i])
        embeddings_df_dict["embedding_esmc"].append(embedding)

    # compute contextualised protein embeddings with Bacformer
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

    # outputs["last_hidden_state"] will be of shape(batch_size, n, 480), 480 is emb dim.
    mask = inputs["special_tokens_mask"] == 4
    bacformer_emb_series = outputs["last_hidden_state"][mask]

    # save bacformer embeddings for comparison
    for i, embedding in enumerate(bacformer_embs):
        embeddings_df_dict["protein_id"].append(protein_ids[i])
        embeddings_df_dict["embedding_bacformer"].append(embedding)

    df = polars.DataFrame(embeddings_df_dict).sort("protein_id")
    print(df)
    #df.write_parquet(save_path)
