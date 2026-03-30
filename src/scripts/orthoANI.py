import pyorthoani
from Bio.SeqIO import read
import os
from pathlib import Path
import argparse


def read_sequence(genome_path):
    return read(genome_path, "fasta")


def calculate_ani_table(genomes_list):
    genomes = [read_sequence(genome_path) for genome_path in genomes_list]
    result = pyorthoani.orthoani_pairwise(genomes)
    return result


def main(genomes_dir, output):
    genomes_list = [str(path) for path in Path(genomes_dir).glob("*.fa")]
    print("computing pairwise ANI")
    ani_table = calculate_ani_table(genomes_list)
    print("writing to file")
    with open(output, 'a') as out_handle:
        for (q, r), ani in ani_table.items():
            out_handle.write(f"{q}\t{r}\t{ani}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate OrthoANI between genomes in a directory")
    parser.add_argument("--input", "-i", help="Directory containing genome FASTA files", default="src/data/genomes/gtdb_genomes")
    parser.add_argument("--output", "-o", help="Directory to save output file", default="src/data/results")
    args = parser.parse_args()
    main(args.input, args.output)
