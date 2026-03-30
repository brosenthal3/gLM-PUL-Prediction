import subprocess
import tempfile
import time
from pathlib import Path
import argparse
import polars
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from tqdm import tqdm
from utility_scripts import request_sequence


class BlastHandler:
    def __init__(self, email, cluster_table: polars.DataFrame, output_file: str):
        Entrez.email = email
        NCBIWWW.email = email
        self.cluster_table = cluster_table
        self.output_file = output_file
        

    def run_biopython_blast(self, fasta_path, taxid, ignore_taxonomy=False):
        print(f"Running BLAST for {fasta_path.name}.")
        fasta_string = fasta_path.read_text()
        entrez_query = f"txid{taxid}[Organism]" if taxid is not None or ignore_taxonomy else None

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
            print("Alignment length: ", alignment.length)
            if abs(alignment.length) > len(fasta_string): # only consider longer hits
                sacc = alignment.accession
                # check all HSPs
                for hsp in alignment.hsps:
                    sstart, send = hsp.sbjct_start, hsp.sbjct_end
                    qstart, qend = hsp.query_start, hsp.query_end
                    evalue = hsp.expect
                    pident = 100.0 * hsp.identities / hsp.align_length
                    q_coverage = 100.0 * (qend - qstart + 1) / len(fasta_string)
                    qacc = blast_record.query.split('.')[0]

                    if float(pident) >= 99 and float(q_coverage) > 90 and qacc != sacc:
                        print(f"Found hit: {sacc} for query {qacc}, pident={pident}, length={abs(int(send)-int(sstart))}")
                        new_start = min(sstart, send)
                        new_end = max(sstart, send)
                        query_start = min(qstart, qend)
                        query_end = max(qstart, qend)

                        result.append((sacc, new_start, new_end, query_start, query_end, float(evalue), float(pident), alignment.length))

        print(f"Found {len(result)} valid hits above 99% identity and 90% query coverage.")
        # return hit with longest alignment
        if result:
            longest_result = max(result, key=lambda x: x[7]) # sort by alignment length
            return longest_result
        else:
            return None


    def fetch_sequence(self, accession):
        with Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text") as handle:
            record = SeqIO.read(handle, "fasta")

        # save to temporary fasta file for blast query
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as temp_file:
            fasta_path = Path(temp_file.name)
            temp_file.write(f">{record.id}\n{record.seq}\n")

        return fasta_path


    def save_result(self, result):
        # save result, save NA if no hit found
        if result:
            sacc, sstart, send, qstart, qend, evalue, pident, alignment_length = result
        else:
            sacc, sstart, send, qstart, qend, evalue, pident, alignment_length = "NO_HIT", "NA", "NA", "NA", "NA", "NA", "NA", "NA"
            unsuccessful_accessions.append(accession)

        # output.append({
        #     "query_accession": accession,
        #     "query_start": qstart,
        #     "query_end": qend,
        #     "tax_id": taxid,
        #     "subject_accession": sacc,
        #     "subject_start": sstart,
        #     "subject_end": send,
        #     "evalue": evalue,
        #     "pident": pident,
        #     "alignment_length": alignment_length
        # })

        # write to file immediately to avoid losing results if script crashes and to keep track of progress
        with open(self.output_file, "a") as f:
            f.write(f"{accession}\t{qstart}\t{qend}\t{taxid}\t{sacc}\t{sstart}\t{send}\t{evalue}\t{pident}\t{alignment_length}\n")


    def get_blast_results(self):
        # iterate over sequences
        accessions_table = self.cluster_table.select("sequence_id", "tax_id").unique()

        for row in tqdm(accessions_table.iter_rows(named=True), total=accessions_table.shape[0]):
            accession = row["sequence_id"]
            taxid = row["tax_id"]
            print(f"Processing {accession} (taxid {taxid})")

            # save sequence to fasta file
            fasta_path = self.fetch_sequence(accession)

            # run blast and parse output
            try:
                result = self.run_biopython_blast(fasta_path, taxid)
            except Exception as e:
                print(f"Error running BLAST for {accession}: {e}")
                result = None
            fasta_path.unlink()

            self.save_result(result)
            time.sleep(5) # being nice to NCBI
        
        print(f"Finished processing. Unsuccessful accessions: {len(unsuccessful_accessions)}")


# def get_pul_info(output, accession, cluster_id, start, end):
#     for entry in output:
#         if entry["query_accession"] == accession:
#             shifted_bp = entry["subject_start"] - entry["query_start"]
#             # shift new subject accordingly to get new PUL range
#             s_start = start + shifted_bp
#             s_end = end + shifted_bp

#             return {
#                 "cluster_id": cluster_id,
#                 "query_accession": accession,
#                 "query_start": start,
#                 "query_end": end,
#                 "tax_id": entry["tax_id"],
#                 "subject_accession": entry["subject_accession"],
#                 "subject_start": s_start,
#                 "subject_end": s_end,
#                 "evalue": None,
#                 "pident": None
#             }

#     return {
#         "cluster_id": cluster_id,
#         "query_accession": accession,
#         "query_start": start,
#         "query_end": end,
#         "tax_id": None,
#         "subject_accession": "NO_HIT",
#         "subject_start": "NA",
#         "subject_end": "NA",
#         "evalue": "NA",
#         "pident": "NA"
#     }


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Fetch GenBank records from a TSV containing NCBI IDs.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="src/data/results/truncated_genomes.tsv",
        help="Input TSV file",
    )
    parser.add_argument(
        "-o", 
        "--output", 
        type=str, 
        default="src/data/results/blast_full_sequences.tsv",
        help="Output file"
    )
    parser.add_argument(
        "--email",
        type=str,
        default="b.rosenthal@lumc.nl",
        help="Email address required by NCBI Entrez",
    )

    args = parser.parse_args()
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
            f.write("query_accession\tquery_start\tquery_end\ttax_id\tsubject_accession\tsubject_start\tsubject_end\tevalue\tpident\talignment_length\n")

    # get blast results for each sequence
    Blaster = BlastHandler(args.email, truncated_df, args.output)
    Blaster.get_blast_results()


if __name__ == "__main__":
    main()