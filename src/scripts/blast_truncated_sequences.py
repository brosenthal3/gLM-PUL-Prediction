import subprocess
import tempfile
import time
from pathlib import Path
import argparse
import polars
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML


def fetch_subsequence(accession, start, end):
    """ Fetch full sequence in fasta format """

    print(f"Fetching {accession} from NCBI")
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    handle.close()

    # split into desired PUL subsequence
    pul_subsequence = record.seq[start-1:end]
    return record.id, pul_subsequence


# def run_blast(fasta_path, taxid):
#     """ Set up command and run using subprocess """
#     output_file = fasta_path.with_suffix(".blast")

#     commmand = [
#         "blastn",
#         "-task", "megablast",
#         "-query", str(fasta_path),
#         "-db", "nt",
#         "-remote",
#         "-outfmt", "6 sacc sstart send evalue staxid pident qacc",
#         "-max_target_seqs", "5",
#         "-out", str(output_file)
#     ]
#     if taxid is not None:
#         commmand.extend(["-entrez_query", f"txid{taxid}[Organism]"])

#     print(f"Running BLAST for {fasta_path.name}")
#     subprocess.run(commmand, check=True)

#     return output_file

# def parse_filter_blast_output(blast_file):
#     """ parses blast output from file """ 

#     print("Parsing BLAST output")
#     results = []
#     with open(blast_file) as f:
#         lines = f.readlines()
#         if not lines:
#             return None
#         for line in lines:
#             if not line:
#                 continue

#             line = line.strip('\n')
#             s_accession, s_start, s_end, evalue, staxid, pident, qacc = line.split() # Output: "sacc sstart send evalue staxid pident",

#             # filter for high identity and exclude self-hits
#             query_accession = qacc.split('.')[0]
#             if float(pident) >= 95.0 and query_accession != s_accession:
#                 print(f"Found hit: {s_accession} for query {qacc}")    
#                 new_pul_range = (int(s_start), int(s_end)) # account for complementary strand hits
#                 results.append((s_accession, min(new_pul_range), max(new_pul_range), float(evalue), float(pident)))

#     return results[0] if results else None


def run_biopython_blast(fasta_path, taxid):
    print(f"Running BLAST for {fasta_path.name}.")
    fasta_string = fasta_path.read_text()
    entrez_query = f"txid{taxid}[Organism]" if taxid is not None else None

    result_handle = NCBIWWW.qblast(
        "blastn",
        "nt",
        fasta_string,
        megablast=True,
        entrez_query=entrez_query,
        hitlist_size=5,
    )

    blast_record = NCBIXML.read(result_handle)
    result = []
    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            sacc = alignment.accession
            sstart = hsp.sbjct_start
            send = hsp.sbjct_end
            evalue = hsp.expect
            pident = 100.0 * hsp.identities / hsp.align_length
            qacc = blast_record.query.split('.')[0]

            if float(pident) >= 95.0 and qacc != sacc:
                print(f"Found hit: {sacc} for query {qacc}")    
                new_pul_range = (int(sstart), int(send)) # account for complementary strand hits
                result.append((sacc, min(new_pul_range), max(new_pul_range), float(evalue), float(pident)))

    print(f"Found {len(result)} valid hits.")
    return result[0] if result else None


def get_pul_info(output, accession, cluster_id, start, end):
    for entry in output:
        if entry["query_accession"] == accession:
            shifted_bp = entry["subject_start"] - entry["query_start"]
            # shift new subject accordingly to get new PUL range
            s_start = start + shifted_bp
            s_end = end + shifted_bp

            return {
                "cluster_id": cluster_id,
                "query_accession": accession,
                "query_start": start,
                "query_end": end,
                "tax_id": entry["tax_id"],
                "subject_accession": entry["subject_accession"],
                "subject_start": s_start,
                "subject_end": s_end,
                "evalue": None,
                "pident": None
            }

    return {
        "cluster_id": cluster_id,
        "query_accession": accession,
        "query_start": start,
        "query_end": end,
        "tax_id": None,
        "subject_accession": "NO_HIT",
        "subject_start": "NA",
        "subject_end": "NA",
        "evalue": "NA",
        "pident": "NA"
    }


def get_blast_results(truncated_df):
    # iterate over sequences
    output = []
    tried_accessions = set()
    unsuccessful_accessions = []

    for row in truncated_df.iter_rows(named=True):
        accession = row["sequence_id"]
        cluster_id = row["cluster_id"]
        start = int(row["start"])
        end = int(row["end"])
        taxid = row["tax_id"]

        # # handle cases where we already tried to fetch this accession
        # if accession in tried_accessions:
        #     print(f"Already tried {accession}, adding PUL info based on previous attempt")
        #     output_dict = get_pul_info(output, accession, cluster_id, start, end)
        #     output.append(output_dict)
        #     continue

        print(f"Processing {accession}:{start}-{end} (taxid {taxid})")
        # fetch subsequence
        seq_id, subseq = fetch_subsequence(accession, start, end)

        # save to temporary fasta file for blast query
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as temp_file:
            fasta_path = Path(temp_file.name)
            temp_file.write(f">{seq_id}_{cluster_id}_{start}_{end}\n{subseq}\n")

        # run blast and parse output
        # blast_output = run_blast(fasta_path, taxid)
        # result = parse_filter_blast_output(blast_output)
        result = run_biopython_blast(fasta_path, taxid)
        fasta_path.unlink()

        # save result, save NA if no hit found
        if result:
            sacc, sstart, send, evalue, pident = result
        else:
            sacc, sstart, send, evalue, pident = "NO_HIT", "NA", "NA", "NA", "NA"
            unsuccessful_accessions.append(accession)

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
            "pident": pident
        })
        time.sleep(5) # being nice to NCBI
        tried_accessions.add(accession)
    
    print(f"Finished processing. Unsuccessful accessions: {len(unsuccessful_accessions)}")
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
    NCBIWWW.email = args.email
    truncated_df = polars.read_csv(args.input, separator="\t")
    output = get_blast_results(truncated_df)
    # save results
    output_df = polars.DataFrame(output)
    output_df.write_csv(args.output, separator="\t")


if __name__ == "__main__":
    main()