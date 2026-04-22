# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
# from esm.sdk import batch_executor
from Bio.SeqIO.FastaIO import SimpleFastaParser
import os
import polars
from tqdm import tqdm
import torch
from transformers import AutoModelForMaskedLM

output_path = "src/data/results/esmc/esmc_embeddings"
os.makedirs("src/data/results/esmc", exist_ok=True)
faa_path = "src/data/genecat_output/trunc_genome.genes.faa"
sequences = polars.read_csv("src/data/results/cblaster_results.tsv", separator="\t", infer_schema_length=700).select("sequence_id").unique()
genes_df = (
    polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
    .select("protein_id", "sequence_id").unique()
    .join(sequences, on="sequence_id", how="semi")
    .group_by("sequence_id")
    .agg(polars.col("protein_id"))
)
genes_dict = dict(zip(genes_df["sequence_id"], genes_df["protein_id"].to_list()))

model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True)

# get embeddings per contig
for contig in genes_dict.keys():
    save_path = f"{output_path}_{contig}.pth"
    if os.path.exists(save_path):
        print("Found embeddings, skipping contig: ", {contig})
        continue

    genes = set(genes_dict.get(contig))
    proteins = []
    protein_ids = {}
    with open(faa_path, "r") as f:
        faa_iter = SimpleFastaParser(f)
        for gene_id, seq in tqdm(faa_iter, desc="Reading FASTA genes"):
            if gene_id not in genes:
                continue

            proteins.append(str(seq))
            protein_ids.update({str(seq): gene_id})

    print(len(proteins), " proteins to process...")
    if len(proteins) == 0:
        continue

    embedding_dict = model.embed_dataset(
        sequences=proteins,
        tokenizer=model.tokenizer,
        batch_size=16,
        max_len=1500,
        full_embeddings=False, # pooling is performed
        embed_dtype=torch.float32, # cast to what dtype you want
        pooling_type='mean',
        num_workers=4, # from HUGGINGFACE: if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
        # save=True, # if True, embeddings will be saved as a .pth file
        # save_path=save_path,
    )

    embeddings_df_dict = {"protein_id": [], "embedding": []}
    for sequence, embedding in embedding_dict.items():
        embeddings_df_dict["protein_id"].append(protein_ids.get(sequence))
        embeddings_df_dict["embedding"].append(embedding)

        df = polars.DataFrame(embeddings_df_dict)
        df.write_parquet(save_path.replace(".pth", ".parquet"))

    


#     try:
#         client = ESMC.from_pretrained("esmc_300m").to("cuda") # or "cpu"
#         outputs = []
#         for protein in tqdm(proteins, total=len(proteins), desc="Embedding proteins..."):
#             outputs.append(embed_sequence(protein))    
#         del client
        
#         # save as df
#         emb_series = polars.Series("embeddings", [output[1] for output in outputs])
#         seq_length_series = polars.Series("protein_length", [output[0] for output in outputs])
#         gene_series = polars.Series("protein_id", protein_ids)
#         df = polars.DataFrame([gene_series, seq_length_series, emb_series])
#         print(df)
#         df.write_parquet(save_path)

#     except:
#         failed_contigs.append(contig)
#         continue

# with open(f"{output_path}_failed.txt", w) as f:
#     for failed_contig in failed_contigs:
#         f.write(failed_contig, "\n")