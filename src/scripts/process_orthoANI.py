import polars
import argparse
from Bio import Align
from Bio.SeqIO import read, FastaIO
from pathlib import Path
import tempfile
import subprocess
from tqdm import tqdm
from data_collection import merge_overlapping_puls

class orthoANIProcessor:
    def __init__(self, ani_table_path, clusters_table_path):
        self.ani_table = polars.read_csv(ani_table_path, separator='\t', has_header=False, new_columns=["query", "reference", "ani"])
        self.clusters_table = polars.read_csv(clusters_table_path, separator='\t', infer_schema_length=1000)
        self.lengths_table = self.clusters_table.select("sequence_id", "length").unique()
    
    def deduplicate_identical_sequences(self):
        merged_clusters_table = self.clusters_table
        identical_sequences = (
            self.ani_table
            .filter(polars.col("ani") == 1.0).filter(polars.col("longer_length") == polars.col("shorter_length"))
            .sort("shorter"))
        identical_sequences_grouped = identical_sequences.group_by("shorter")

        replaced_sequences = []
        for _, group in identical_sequences_grouped:
            query = group[0, "shorter"]
            references = group.select("longer").to_series()
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
                    polars.when(polars.col("shorter").eq(ref)).then(polars.lit(query)).otherwise(polars.col("shorter")).alias("shorter"),
                    polars.when(polars.col("longer").eq(ref)).then(polars.lit(query)).otherwise(polars.col("longer")).alias("longer"),
                )
                replaced_sequences.append(ref)

        print(f"Replaced {len(replaced_sequences)} identical sequences with their cluster representative")
        print(merged_clusters_table.select("sequence_id").unique().shape[0], "unique sequences in merged cluster table")

        self.clusters_table = merged_clusters_table
        self.ani_table = self.ani_table.filter(polars.col("shorter") != polars.col("longer")).unique()
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
            if hit[2] <= 95:
                print("Bad percent identity")
                continue
            elif hit[3] <= 0.85 * len(pul_sequence):
                print("Bad alignment length")
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
        print(self.ani_table.select("shorter").unique().shape[0], "unique shorter sequences in ANI table")
        print(self.ani_table.select("longer").unique().shape[0], "unique longer sequences in ANI table")

        # group by shorter sequences, keep longest as cluster rep
        duplicated_clusters_grouped = self.ani_table.group_by("shorter")
        fails = 0
        for sequence_id, group in duplicated_clusters_grouped:
            group = group.sort("ani", descending=True)
            group = group.sort("longer_length", descending=True)
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
            .sort("longer_length", descending=True)
            .sort("shorter_length", descending=False))

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
    new_cluster_table = merge_overlapping_puls(new_cluster_table, keep_original=False)
    new_cluster_table.write_csv("src/data/results/clusters_deduplicated.tsv", separator='\t')

    # new_cluster_table = polars.read_csv("src/data/results/clusters_deduplicated.tsv", separator='\t', infer_schema_length=1000)
    # orthiANI_processor = orthoANIProcessor(args.input, "src/data/results/combined_clusters_blasted_gtdb_filtered.tsv")
    # ani_table = orthiANI_processor.filter_ani_table()
    # remaining_sequences = new_cluster_table.select("sequence_id").unique()
    # ani_table = ani_table.join(remaining_sequences.rename({"sequence_id": "shorter"}), on="shorter", how="semi").join(remaining_sequences.rename({"sequence_id": "longer"}), on="longer", how="semi")
    # print(ani_table)

    # print(remaining_sequences.shape[0], "unique sequences in deduplicated clusters table")