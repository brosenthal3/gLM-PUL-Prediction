import subprocess
import tempfile
import time
from pathlib import Path
import argparse
import polars
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from tqdm import tqdm


def fetch_subsequence(accession, start, end):
    """ Fetch full sequence in fasta format """

    print(f"Fetching {accession} from NCBI")
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    handle.close()

    # split into desired PUL subsequence
    pul_subsequence = record.seq[start-1:end]
    return record.id, pul_subsequence


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

            if float(pident) >= 99.5 and qacc != sacc:
                print(f"Found hit: {sacc} for query {qacc}")    
                new_pul_range = (int(sstart), int(send)) # account for complementary strand hits
                result.append((sacc, min(new_pul_range), max(new_pul_range), float(evalue), float(pident)))

    print(f"Found {len(result)} valid hits above 99.5% identity.")
    # return hit with longest alignment
    if result:
        longest_result = max(result, key=lambda x: x[2] - x[1])
        return longest_result
    else:
        return None


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


def get_blast_results(truncated_df, output_file):
    # iterate over sequences
    output = []
    tried_accessions = set()
    unsuccessful_accessions = []

    for row in tqdm(truncated_df.iter_rows(named=True), total=truncated_df.shape[0]):
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
        try:
            result = run_biopython_blast(fasta_path, taxid)
        except Exception as e:
            print(f"Error running BLAST for {accession}: {e}")
            result = None

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
        # write to file immediately to avoid losing results if script crashes and to keep track of progress
        with open(output_file, "a") as f:
            f.write(f"{cluster_id}\t{accession}\t{start}\t{end}\t{taxid}\t{sacc}\t{sstart}\t{send}\t{evalue}\t{pident}\n")

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

    # try checking output file first to avoid unnecessary blast runs
    if Path(args.output).exists():
        existing_output = polars.read_csv(args.output, separator='\t')
        print(f"Existing output file found with {existing_output.shape[0]} entries. Filtering out already processed sequences.")
        # filter out sequences with results
        truncated_df = truncated_df.join(existing_output.select("query_accession"), left_on="sequence_id", right_on="query_accession", how="anti")
    else:
        # create output file and write header
        with open(args.output, "w") as f:
            f.write("cluster_id\tquery_accession\tquery_start\tquery_end\ttax_id\tsubject_accession\tsubject_start\tsubject_end\tevalue\tpident\n")

    # get blast results for each sequence
    output = get_blast_results(truncated_df, args.output)


if __name__ == "__main__":
    main()