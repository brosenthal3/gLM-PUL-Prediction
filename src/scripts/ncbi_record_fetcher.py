""" Adapted from genecat scripts, by Raymund Hackett """

import argparse
from pathlib import Path
from typing import List, Literal
import polars
from Bio import Entrez
import time
from tqdm import tqdm


def fetch_ncbi_records(
    ids: List[str],
    output_path: Path,
    email: str,
    type: Literal["fasta", "genbank"],
    separate: bool = False,
):
    """
    Writes a single file containing all fetched records.
    """
    Entrez.email = email

    if type == "genbank":
        rettype = "gbwithparts"
        type_suffix = "gb"
    elif type == "fasta":
        rettype = "fasta"
        type_suffix = "fa"
    else:
        raise NotImplementedError(
            f"Unknown record type {type}. Please select `fasta` or `genbank`"
        )

    # create output directory if it doesn't exist
    if separate and not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    for acc in tqdm(ids, desc="Fetching NCBI records"):
        handle = Entrez.efetch(
            db="nuccore",
            id=acc,
            retmode="text",
            rettype=rettype,
            complexity=1,
        )
        record = handle.read()

        if separate:
            record_path = output_path / f"{acc}.{type_suffix}"
            with open(record_path, "w") as out_handle:
                out_handle.write(record)
        else:
            with open(output_path, "a") as out_handle:
                out_handle.write(record)
                out_handle.write("\n")

        with open(output_path.with_suffix(".ids.txt"), "a") as id_handle:
            id_handle.write(acc)
            id_handle.write("\n")

        handle.close()
        time.sleep(5)

def get_remaining_ids(output_path: Path, ids: List[str]) -> List[str]:
    if output_path.with_suffix(".ids.txt").exists():
        with open(output_path.with_suffix(".ids.txt"), "r") as id_handle:
            fetched_ids = set(id_handle.read().splitlines())
        remaining_ids = [acc for acc in ids if acc not in fetched_ids]
    else:
        remaining_ids = ids

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
        default="b.rosenthal@lumc.nl",
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
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Whether to save each record in a separate file (named <ID>.<type>) in the output directory, or to save all records in a single file. Default False",
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

    # to avoid refetching if script crashes, don't repeat more than 5 times so we don't end up in an infinite loop
    error_count = 0
    remaining_ids = get_remaining_ids(args.output, ids)
    while remaining_ids and error_count < 5:
        try:
            remaining_ids = get_remaining_ids(args.output, ids)
            fetch_ncbi_records(remaining_ids, args.output, args.email, args.type, args.separate)
        
        except Exception as e:
            print(f"Error fetching records: {e}")
            print(f"Attempting again with saved progress")
            time.sleep(3)
            remaining_ids = get_remaining_ids(args.output, ids) # get again since some records might have been fetched before error
            print(f"Remaining IDs to fetch: {len(remaining_ids)}")
            error_count += 1


if __name__ == "__main__":
    main()
