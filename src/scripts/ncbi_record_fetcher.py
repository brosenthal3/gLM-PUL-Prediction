""" Adapted from genecat scripts, by Raymund Hackett """

import argparse
from pathlib import Path
from typing import List, Literal

import polars
from Bio import Entrez


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

    with open(output_path, "w") as out_handle:
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
            out_handle.write(record)
            out_handle.write("\n")
            handle.close()


def fetch_ncbi_records_individual(
    ids: List[str],
    output_dir: Path,
    email: str,
    type: Literal["fasta", "genbank"],
):
    """
    Writes one file per record into output_dir.
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        out_path = output_dir / f"{acc}.gb"
        with open(out_path, "w") as out_handle:
            out_handle.write(record)
        handle.close()


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
    parser.add_argument(
        "--split",
        action="store_true",
        help="Write individual GenBank files instead of one combined file (default: off)",
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

    if args.split:
        fetch_ncbi_records_individual(ids, args.output, args.email, type=args.type)
    else:
        fetch_ncbi_records(ids, args.output, args.email, args.type)


if __name__ == "__main__":
    main()
