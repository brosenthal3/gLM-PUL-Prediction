""" Adapted from genecat scripts, by Raymund Hackett """

import argparse
from pathlib import Path
from typing import List, Literal
import polars
from Bio import Entrez
import time


def fetch_ncbi_records(
    ids: List[str],
    output_path: Path,
    email: str,
    type: Literal["fasta", "genbank"],
):
    """
    Writes a single file containing all fetched records.
    """
    Entrez.email = email

    if type == "genbank":
        rettype = "gbwithparts"
    elif type == "fasta":
        rettype = "fasta"
    else:
        raise NotImplementedError(
            f"Unknown record type {type}. Please select `fasta` or `genbank`"
        )

    for acc in ids:
        print(f"Fetching {acc}")
        handle = Entrez.efetch(
            db="nuccore",
            id=acc,
            retmode="text",
            rettype=rettype,
            complexity=1,
        )
        record = handle.read()

        with open(output_path, "a") as out_handle:
            out_handle.write(record)
            out_handle.write("\n")

        with open(output_path.with_suffix(".ids.txt"), "a") as id_handle:
            id_handle.write(acc)
            id_handle.write("\n")

        handle.close()

def get_remaining_ids(output_path: Path, ids: List[str]) -> List[str]:
    with open(output_path.with_suffix(".ids.txt"), "r") as id_handle:
        fetched_ids = set(id_handle.read().splitlines())
    remaining_ids = [acc for acc in ids if acc not in fetched_ids]
    return remaining_ids


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GenBank records from a TSV containing NCBI IDs."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input TSV file",
    )
    parser.add_argument(
        "--col-name",
        type=str,
        default="sequence_id",
        help="Column name containing sequence IDs",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="r.e.hackett@lumc.nl",
        help="Email address required by NCBI Entrez",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file or directory, depending on mode"
    )
    parser.add_argument(
        "--type",
        choices=["fasta", "genbank"],
        default="fasta",
        help="Which NCBI record type should be fetched. Default fasta",
    )

    args = parser.parse_args()

    df = polars.read_csv(args.input, separator="\t", comment_prefix="#")
    if args.col_name not in df.columns:
        raise ValueError(f"Column '{args.col_name}' not found in TSV.")

    ids = (
        df[args.col_name]
        .unique()
        .str.replace(r"-([0-9]+)$", r".$1")
        .drop_nulls()
        .to_list()
    )
    # to avoid refetching if script crashes
    remaining_ids = get_remaining_ids(args.output, ids)

    try:
        fetch_ncbi_records(ids, args.output, args.email, args.type)
    except Exception as e:
        print(f"Error fetching records: {e}")
        print(f"Attempting again with saved progress")
        time.sleep(3)
        remaining_ids = get_remaining_ids(args.output, ids)
        print(f"Remaining IDs to fetch: {len(remaining_ids)}")
        fetch_ncbi_records(remaining_ids, args.output, args.email, args.type)


if __name__ == "__main__":
    main()
