import polars
import subprocess
import tempfile
import time
from Bio import Entrez, SeqIO
from pathlib import Path
import argparse


def fetch_subsequence(accession, start, end):
    """ Fetch full sequence in fasta format """

    print(f"Fetching {accession} from NCBI")
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    handle.close()

    # split into desired PUL subsequence
    pul_subsequence = record.seq[start-1:end]
    return record.id, pul_subsequence


def run_blast(fasta_path, taxid):
    """ Set up command and run using subprocess """
    output_file = fasta_path.with_suffix(".blast")

    commmand = [
        "blastn",
        "-task", "megablast",
        "-query", str(fasta_path),
        "-db", "nt",
        "-remote",
        "-outfmt", "6 sacc sstart send evalue staxid pident qacc",
        "-max_target_seqs", "5",
        "-out", str(output_file)
    ]
    if taxid is not None:
        commmand.extend(["-entrez_query", f"txid{taxid}[Organism]"])

    print(f"Running BLAST for {fasta_path.name}")
    subprocess.run(commmand, check=True)

    return output_file


def parse_filter_blast_output(blast_file):
    """ parses blast output from file """ 

    print("Parsing BLAST output")
    results = []
    with open(blast_file) as f:
        lines = f.readlines()
        if not lines:
            return None
        for line in lines:
            if not line:
                continue

            line = line.strip('\n')
            s_accession, s_start, s_end, evalue, staxid, pident, qacc = line.split() # Output: "sacc sstart send evalue staxid pident",

            # filter for high identity and exclude self-hits
            query_accession = qacc.split('.')[0]
            if float(pident) >= 95.0 and query_accession != s_accession:
                print(f"Found hit: {s_accession} for query {qacc}")    
                new_pul_range = (int(s_start), int(s_end)) # account for complementary strand hits
                results.append((s_accession, min(new_pul_range), max(new_pul_range), float(evalue), float(staxid), float(pident)))

    return results[0] if results else None


def get_blast_results(truncated_df):
    # iterate over sequences
    output = []
    for row in truncated_df.iter_rows(named=True):
        accession = row["sequence_id"]
        cluster_id = row["cluster_id"]
        start = int(row["start"])
        end = int(row["end"])
        taxid = row["tax_id"]

        print(f"Processing {accession}:{start}-{end} (taxid {taxid})")
        # fetch subsequence
        seq_id, subseq = fetch_subsequence(accession, start, end)

        # save to temporary fasta file for blast query
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as temp_file:
            fasta_path = Path(temp_file.name)
            temp_file.write(f">{seq_id}_{cluster_id}_{start}_{end}\n{subseq}\n")

        # run blast and parse output
        blast_output = run_blast(fasta_path, taxid)
        result = parse_filter_blast_output(blast_output)

        # save result, save NA if no hit found
        if result:
            sacc, sstart, send, evalue, staxid, pident = result
        else:
            sacc, sstart, send, evalue, staxid, pident = "NO_HIT", "NA", "NA", "NA", "NA", "NA" 

        output.append({
            "cluster_id": cluster_id,
            "query_accession": accession,
            "query_start": start,
            "query_end": end,
            "tax_id": taxid,
            "subject_accession": sacc,
            "subject_start": sstart,
            "subject_end": send,
            "evalue": evalue,
            "subject_tax_id": staxid,
            "pident": pident
        })
        time.sleep(5) # being nice to NCBI

    return output


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Fetch GenBank records from a TSV containing NCBI IDs.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="../data/truncated_genomes_test.tsv",
        help="Input TSV file",
    )
    parser.add_argument(
        "-o", 
        "--output", 
        type=str, 
        default="../data/blast_full_sequences.tsv",
        help="Output file"
    )
    parser.add_argument(
        "--email",
        type=str,
        default="b.rosenthal@lumc.nl",
        help="Email address required by NCBI Entrez",
    )
    args = parser.parse_args()
    Entrez.email = args.email

    truncated_df = polars.read_csv(args.input, separator="\t")
    output = get_blast_results(truncated_df)
    # save results
    output_df = polars.DataFrame(output)
    output_df.write_csv(args.output, separator="\t")


if __name__ == "__main__":
    main()