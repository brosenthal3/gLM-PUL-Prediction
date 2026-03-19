import polars
import argparse
from Bio import Align
from Bio.SeqIO import read, FastaIO
from pathlib import Path
import tempfile
import subprocess
from tqdm import tqdm

class orthoANIProcessor:
    def __init__(self, ani_table_path, clusters_table_path):
        self.ani_table = polars.read_csv(ani_table_path, separator='\t', has_header=False, new_columns=["query", "reference", "ani"])
        self.clusters_table = polars.read_csv(clusters_table_path, separator='\t', infer_schema_length=1000)
        self.lengths_table = self.clusters_table.select("sequence_id", "length").unique()
    
    def deduplicate_identical_sequences(self):
        merged_clusters_table = self.clusters_table
        identical_sequences = self.ani_table.filter(polars.col("ani") == 1.0).filter(polars.col("query_length") == polars.col("reference_length")).sort("query")
        identical_sequences_grouped = identical_sequences.group_by("query")

        replaced_sequences = []
        for _, group in identical_sequences_grouped:
            query = group[0, "query"]
            references = group.select("reference").to_series()
            if query in replaced_sequences:
                continue
            
            for ref in references:
                # replace all identical sequences in cluster table with query
                merged_clusters_table = merged_clusters_table.with_columns(
                    polars.when(polars.col("sequence_id").eq(ref))
                    .then(polars.lit(query))
                    .otherwise(polars.col("sequence_id"))
                    .alias("sequence_id")
                )

                # replace reference with query in ani table
                self.ani_table = self.ani_table.with_columns(
                    polars.when(polars.col("query").eq(ref)).then(polars.lit(query)).otherwise(polars.col("query")).alias("query"),
                    polars.when(polars.col("reference").eq(ref)).then(polars.lit(query)).otherwise(polars.col("reference")).alias("reference"),
                )
                replaced_sequences.append(ref)

        print(f"Replaced {len(replaced_sequences)} identical sequences with their cluster representative")
        print(merged_clusters_table.select("sequence_id").unique().shape[0], "unique sequences in merged cluster table")

        self.clusters_table = merged_clusters_table
        self.ani_table = self.ani_table.filter(polars.col("query") != polars.col("reference")).unique()
        return


    def blast_pul(self, pul_sequence, subject_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            pul_path = Path(tmpdir) / "pul.fasta"
            with open(pul_path, 'w') as out_handle:
                fasta_writer = FastaIO.FastaWriter(out_handle, wrap=None)
                fasta_writer.write_record(pul_sequence)

            cmd = f"blastn -query {pul_path} -subject {subject_path} -out {Path(tmpdir) / 'results.txt'} -outfmt '6 qseqid sseqid pident length sstart send evalue'"
            subprocess.run(cmd, shell=True, check=True)
            try:
                blast_results = polars.read_csv(Path(tmpdir) / "results.txt", separator='\t', has_header=False, new_columns=["qseqid", "sseqid", "pident", "length", "sstart", "send", "evalue"])
            except polars.exceptions.NoDataError:
                print(f"No BLAST hits found for PUL {pul_sequence.id} against subject {subject_path.name}")
                return None

        hits = blast_results.sort("pident", descending=True).sort("length", descending=True)
        for hit in hits.iter_rows():
            if hit[2] <= 95 or hit[3] <= 0.9 * len(pul_sequence):
                continue
            else:
                return hit[4], hit[5]

        return None


    def replace_puls(self, old_id, new_id):
        fails = 0
        # get sequences
        genomes_path = Path("src/data/genomes/selected_genomes/")
        old_sequence = read(genomes_path / f"{old_id}.fa", "fasta")
        subject_path = genomes_path / f"{new_id}.fa"

        # group by old_id
        old_puls = self.clusters_table.filter(polars.col("sequence_id") == old_id)
        for pul in old_puls.iter_rows():
            start = min(pul[2], pul[3])
            end = max(pul[2], pul[3])
            pul_sequence = old_sequence[start:end]

            if not len(pul_sequence) > 0:
                print(f"PUL sequence {pul_sequence.id} has length 0, cannot perform BLAST")
                fails += 1
                continue

            blast_result = self.blast_pul(pul_sequence, subject_path)
            if blast_result is None:
                # remove PUL from cluster table
                self.clusters_table = self.clusters_table.filter(~polars.col("cluster_id").eq(pul[0]))
                print(f"No good BLAST hit found for PUL {pul[0]}, removing from cluster table")
                fails += 1
                continue

            new_start, new_end = min(blast_result), max(blast_result)

            # replace pul coordinates with new coordinates based on blast alignment
            self.clusters_table = self.clusters_table.with_columns(
                polars.when(polars.col("cluster_id").eq(pul[0])).then(polars.lit(new_start)).otherwise(polars.col("start")).alias("start"),
                polars.when(polars.col("cluster_id").eq(pul[0])).then(polars.lit(new_end)).otherwise(polars.col("end")).alias("end"),
                polars.when(polars.col("cluster_id").eq(pul[0])).then(polars.lit(new_id)).otherwise(polars.col("sequence_id")).alias("sequence_id")
            )

        return fails


    def deduplicate_similar_sequences(self):
        ani_table = (
            self.ani_table
            .with_columns(
                # Save longer sequence as cluster representative.
                polars.when(polars.col("query_length") >= polars.col("reference_length"))
                .then(polars.col("query"))
                .otherwise(polars.col("reference"))
                .alias("longer"),
                polars.when(polars.col("query_length") >= polars.col("reference_length"))
                .then(polars.col("reference"))
                .otherwise(polars.col("query"))
                .alias("shorter"),
                polars.when(polars.col("query_length") >= polars.col("reference_length"))
                .then(polars.col("query_length"))
                .otherwise(polars.col("reference_length"))
                .alias("longer_length"),
                polars.when(polars.col("query_length") >= polars.col("reference_length"))
                .then(polars.col("reference_length"))
                .otherwise(polars.col("query_length"))
                .alias("shorter_length"),
            )
            .select("shorter", "longer", "shorter_length", "longer_length", "ani")
            .unique()
            .sort("shorter", "shorter_length", descending=False)
        )

        # group by shorter sequences, keep longest as cluster rep
        duplicated_clusters_grouped = ani_table.group_by("shorter")
        fails = 0
        for sequence_id, group in duplicated_clusters_grouped:
            group = group.sort("longer_length", descending=True)
            group = group.sort("ani", descending=True)
            representative = group[0, "longer"]
            fails += self.replace_puls(sequence_id[0], representative)
        print(f"{fails} failed replacements, PULs removed from cluster table due to no good BLAST hit found in reference sequences")
        return


    def filter_ani_table(self):
        self.ani_table = (
            self.ani_table
            .filter(polars.col("query") != polars.col("reference"), polars.col("ani") >= 0.99)
            .with_columns(
                polars.col("query").str.split(".").list.first().alias("query"),
                polars.col("reference").str.split(".").list.first().alias("reference"),
            )
            .join(self.lengths_table.rename({"sequence_id": "query", "length": "query_length"}), on="query", how="left")
            .join(self.lengths_table.rename({"sequence_id": "reference", "length": "reference_length"}), on="reference", how="left")
        )
        return self.ani_table

    def process_clusters(self):
        self.filter_ani_table()
        self.deduplicate_identical_sequences()
        self.deduplicate_similar_sequences()
        print(self.clusters_table.select("sequence_id").unique().shape[0], "unique sequences in final cluster table")

        return self.clusters_table



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter OrthoANI table for ANI >= 99% and query != reference")
    parser.add_argument("--input", "-i", help="Path to the ANI table file", default="src/data/results/orthoANI_output.txt")
    args = parser.parse_args()

    orthiANI_processor = orthoANIProcessor(args.input, "src/data/results/combined_clusters_blasted_gtdb_filtered.tsv")
    new_cluster_table = orthiANI_processor.process_clusters()
    new_cluster_table.write_csv("src/data/results/clusters_deduplicated.tsv", separator='\t')

#    new_cluster_table = polars.read_csv("src/data/results/clusters_deduplicated.tsv", separator='\t', infer_schema_length=1000)
#     ani_table = orthiANI_processor.filter_ani_table()
#     for sequence_id in new_cluster_table.select("sequence_id").unique().to_series():
#         group = ani_table.filter((polars.col("query") == sequence_id) | (polars.col("reference") == sequence_id))
#         similar_sequences = group.select("query").to_series().append(group.select("reference").to_series())
#         if similar_sequences.shape[0] > 0:
#             joined = polars.DataFrame({'sequence_id': similar_sequences}).join(new_cluster_table.select("sequence_id").unique().to_series().to_frame(), how="inner", left_on="sequence_id", right_on="sequence_id")
#             print(joined)
# #            print(f"Sequence {sequence_id} has {group.shape[0]} ANI hits >= 99%")