import pyorthoani
from Bio.SeqIO import read
import os
from pathlib import Path
import argparse
import polars
import subprocess

def compare_sequences(genome_a, genome_b):
    ani = pyorthoani.orthoani(genome_a, genome_b)
    return ani


def read_sequence(genome_path):
    return read(genome_path, "fasta")


def calculate_ani_table(genomes_list):
    done_pairs = set()
    ani_table = []
    for i in range(len(genomes_list)):
        for j in range(i+1, len(genomes_list)):
            pair = frozenset([genomes_list[i], genomes_list[j]])
            if pair in done_pairs:
                continue
            genome_a = read_sequence(genomes_list[i])
            genome_b = read_sequence(genomes_list[j])
            ani = compare_sequences(genome_a, genome_b)
            done_pairs.add(pair)
            ani_table.append((genomes_list[i], genomes_list[j], ani))

    return ani_table


def main(genomes_dir, output):
    genomes_list = [str(path) for path in Path(genomes_dir).glob("*.fa")][:4]
    ani_table = calculate_ani_table(genomes_list)
    ani_dataframe = polars.DataFrame(ani_table, schema=["genome_a", "genome_b", "ani"])
    ani_dataframe.write_csv(Path(output).with_suffix('tsv'), separator='\t', has_header=True)
    return ani_dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate OrthoANI between genomes in a directory")
    parser.add_argument("--input", "-i", help="Directory containing genome FASTA files", default="src/data/genomes/gtdb_genomes")
    parser.add_argument("--output", "-o", help="Directory to save output file", default="src/data/results")
    args = parser.parse_args()
    main(args.input, args.output)