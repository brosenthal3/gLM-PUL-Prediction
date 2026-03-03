import polars
import subprocess
import tempfile
import time
from Bio import Entrez, SeqIO
from pathlib import Path
Entrez.email = "b.rosenthal@LUMC.nl"

INPUT_FILE = "truncated_sequences.tsv"
OUTPUT_FILE = "blast_full_sequences.tsv"

def fetch_subsequence(accession, start, end):
    """ Fetch full sequence in fasta format """
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
        "-entrez_query", f"txid{taxid}[Organism:exp]",
        "-outfmt", "6 sacc sstart send evalue bitscore",
        "-max_target_seqs", "1",
        "-out", str(output_file)
    ]
    subprocess.run(commmand, check=True)

    return output_file


def parse_blast_output(blast_file):
    """ parses blast output from file """ 
    with open(blast_file) as f:
        line = f.readline().strip()
        if not line: 
            return None
        s_accession, s_start, s_end, evalue, bitscore = line.split()
        return s_accession, int(s_start), int(s_end), float(evalue), float(bitscore)


if __name__ == "__main__":
    truncated_df = polars.read_csv(INPUT_FILE, separator="\t")
    output_df = polars.DataFrame(schema=[
        "cluster_id",
        "query_accession",
        "query_start",
        "query_end",
        "taxid",
        "subject_accession",
        "subject_start",
        "subject_end",
        "evalue",
        "bitscore"
    ])
    
    # iterate over sequences
    for row in truncated_df.iter_rows(named=True):
        accession = row["accession"]
        cluster_id = row["cluster_id"]
        start = int(row["start"])
        end = int(row["end"])
        taxid = row["taxid"]

        print(f"Processing {accession}:{start}-{end} (taxid {taxid})")
        # fetch subsequence
        seq_id, subseq = fetch_subsequence(accession, start, end)

        # save to temporary fasta file for blast query
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as temp_file:
            fasta_path = Path(temp_file.name)
            temp_file.write(f">{seq_id}_{cluster_id}_{start}_{end}\n{subseq}\n")

        # run blast and parse output
        blast_output = run_blast(fasta_path, taxid)
        result = parse_blast_output(blast_output)

        # save result, save NA if no hit found
        if result:
            sacc, sstart, send, evalue, bitscore = result
        else:
            sacc, sstart, send, evalue, bitscore = "NO_HIT", "NA", "NA", "NA", "NA" 

        output_df = output_df.vstack(polars.DataFrame({
                "cluster_id": [cluster_id],
                "query_accession": [accession],
                "query_start": [start],
                "query_end": [end],
                "taxid": [taxid],
                "subject_accession": [sacc],
                "subject_start": [sstart],
                "subject_end": [send],
                "evalue": [evalue],
                "bitscore": [bitscore]
            }))

        time.sleep(5) # being nice to NCBI

    # save results
    output_df.write_csv(OUTPUT_FILE, separator="\t")