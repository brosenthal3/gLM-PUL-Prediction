import os
import polars
import sys
import torch
import numpy as np
from Bio import SeqIO
from transformers import logging
from pathlib import Path
from tqdm import tqdm
from bench.embed_bacformer import BacFormerEmbedder
import multiprocessing as mp
mp.set_start_method("fork", force=True)

"""
Script to generate embeddings from ESM-C and Bacformer for all genes in the dataset
Saves individual files of embeddings for each contig in src/data/embeddings/esmc_bacformer_embeddings/<contig_id>.parquet

pip3 install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu126
export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/glm_bench'
export PYTHONPATH='/exports/lucid-grpzeller-work/brosenthal/glm_bench'

# if issues with gpu nodes:
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" 
python src/scripts/esmc_bacformer_script.py
"""

# write all genes from individual contigs to separate fasta files
def write_genes_fasta(genes_dict):
    os.makedirs("src/data/genecat_output/genes", exist_ok=True)
    faa_path = "src/data/genecat_output/genome.genes.faa"
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


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dir = "/exports/lucid-grpzeller-work/brosenthal/bacformer_cache/cache" # NOTE: should change when working on different dirs
    logging.set_verbosity_error()

    output_path = "src/data/embeddings/esmc_bacformer_embeddings"
    os.makedirs(output_path, exist_ok=True)
    sequences = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t", infer_schema_length=700).select("sequence_id").unique()
    genes_df = (
        polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
        .select("protein_id", "sequence_id").unique()
        .join(sequences, on="sequence_id", how="semi")
        .group_by("sequence_id")
        .agg(polars.col("protein_id"))
    )
    genes_dict = dict(zip(genes_df["sequence_id"], genes_df["protein_id"].to_list()))
    # get fasta sequences of all genes per genome
    if len(os.listdir("src/data/genecat_output/genes")) < len(sequences):
        write_genes_fasta(genes_dict)

    # embedder from glm-bench
    embedder = BacFormerEmbedder(
        model="bacformer-large-masked-MAG",
        cache_dir=Path(model_dir),
        force_redownload=True,
        local_only=False,
    )

    # get embeddings per contig
    for contig in tqdm(os.listdir("src/data/genecat_output/genes")):
        genes_file = f"src/data/genecat_output/genes/{contig}"
        contig_genes_index = SeqIO.index(genes_file, "fasta")
        contig_genes_list = genes_dict.get(contig.split(".")[0])
        save_path = f"{output_path}/{contig.replace('faa', 'parquet')}"

        # check if non-NaN embeddings already and skip
        # if os.path.exists(save_path):
        #     existing_embeddings = polars.read_parquet(save_path)
        #     if not existing_embeddings["embedding_bacformer"][0].is_nan().any():
        #         continue
        #     else:
        #         print(f"Found NaN values in embedding for {contig}, {existing_embeddings['embedding_bacformer'].shape}")

        # make list of protein sequences and ids
        proteins = []
        protein_ids = []
        for gene_id in contig_genes_list:
            seq = str(contig_genes_index[gene_id])
            proteins.append(seq)
            protein_ids.append(gene_id)
        if len(proteins) == 0:
            continue
        print(len(proteins), " proteins to process...")
        
        # embed using glm-bench toolkit
        out_dict = embedder.embed_contig(
            protein_sequences=proteins,
            batch_size=64,
        )
        bacformer_embs = out_dict["bacformer_embeddings"] # <--- a 2d numpy array. shape(proteins, dims)
        esm_embs = out_dict["esm_embeddings"] # <--- a 2d numpy array. shape(proteins, dims)

        # some sanity checks
        if np.isnan(bacformer_embs).any():
            print("[WARNING] NaNs in bacformer!")
            print(sum(np.isnan(bacformer_embs)))
            print(bacformer_embs)
        if len(bacformer_embs) != len(proteins):
            print(f"Found {len(bacformer_embs)} embeddings but expected {len(proteins)}")

        # save ESM-C embeddings
        embeddings_df_dict = {"protein_id": [], "embedding_esmc": []}
        for i, embedding in enumerate(esm_embs):
            embeddings_df_dict["protein_id"].append(protein_ids[i])
            embeddings_df_dict["embedding_esmc"].append(embedding)
        esmc_df = polars.DataFrame(embeddings_df_dict).sort("protein_id")

        # save bacformer embeddings
        bacformer_embeddings_df_dict = {"protein_id": [], "embedding_bacformer": []}
        for i, embedding in enumerate(bacformer_embs):
            bacformer_embeddings_df_dict["protein_id"].append(protein_ids[i])
            bacformer_embeddings_df_dict["embedding_bacformer"].append(embedding)
        bacformer_df = polars.DataFrame(bacformer_embeddings_df_dict).sort("protein_id")

        # combine embeddings to one df, save it
        joined_df = esmc_df.join(bacformer_df, on="protein_id", how="inner")
        joined_df.write_parquet(save_path)
        print(joined_df)


if __name__ == "__main__":
    main()



#-------DEPR CODE (using glm-bench directly now)-------#

# function slightly adapted from https://github.com/RayHackett/glm_bench
# def slide_and_index(data: List, apply_bacformer, window_size: int = 6000, stride: int = 4000,) -> Tuple[np.ndarray, np.ndarray]:
#     N = len(data)
#     model_dim = 960
    
#     # get window start positions
#     starts = list(range(0, N, stride))
#     if starts[-1] + window_size < N:
#         starts[-1] = N - window_size

#     # For each position n, find which window gives it the most context
#     best_window = np.clip(
#         np.round((np.arange(N) - (window_size - 1) / 2.0) / stride).astype(int),
#         0, len(starts) - 1
#     )

#     # Run func on each window, collect embedding results into lists
#     bacformer_window_outputs = []
#     esm_window_outputs = []
#     for start in starts:
#         # slice the data to get only protein sequences in current window
#         end = min(start+window_size, N)
#         window = data[start:end]
#         # apply bacformer to get embeddings
#         bacformer_embs, ems_embs = apply_bacformer(window)  # shape: (window_size, d)
#         bacformer_window_outputs.append(bacformer_embs)
#         esm_window_outputs.append(ems_embs)

#     # Index into the correct window output for each position
#     # best_window[n] -> which window, then local index = n - starts[best_window[n]]
#     bacformer_result = np.empty((N, model_dim), dtype=np.float32)
#     esm_result = np.empty((N, model_dim), dtype=np.float32)
#     for n in range(N):
#         w = best_window[n]
#         local_idx = n - starts[w]
#         bacformer_result[n] = bacformer_window_outputs[w][local_idx, :]
#         esm_result[n] = esm_window_outputs[w][local_idx, :]

#     return bacformer_result, esm_result


# def apply_bacformer(protein_sequences_list: List, device: str, model, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
#     # Workaround from Issue: https://github.com/macwiatrak/Bacformer/issues/33:
#     config = model.config
#     dim = config.hidden_size // config.num_attention_heads
#     inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
#     for layer in model.encoder.layers:
#         layer.attn.rotary.register_buffer("inv_freq", inv_freq.clone(), persistent=False)

#     # embed the proteins with ESM++ to get average protein embeddings
#     inputs = protein_seqs_to_bacformer_inputs(
#         protein_sequences=protein_sequences_list,
#         device=device,
#         batch_size=batch_size,  # the batch size for computing the protein embeddings
#         max_n_proteins=6000, # the maximum number of proteins Bacformer was trained with
#         bacformer_model_type="large",
#     )

#     # compute contextualised protein embeddings with Bacformer
#     with torch.no_grad():
#         outputs = model(**inputs, return_dict=True).last_hidden_state

#     # outputs["last_hidden_state"] will be of shape(batch_size, n, d)
#     # 960 is emb dim for large model
#     bacformer_embs = outputs.squeeze(0).to(dtype=torch.float32).cpu().numpy()
#     esm_embeddings = inputs["protein_embeddings"].squeeze(0).to(dtype=torch.float32).cpu().numpy()

#     return bacformer_embs, esm_embeddings