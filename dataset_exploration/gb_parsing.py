import gb_io
import os
from tqdm import tqdm

PULS_dir = "gLM-PUL-Prediction/DBCAN-PUL/dbCAN-PUL"
saved_accessions = set()
all_records = []
error_log = []

# get all subdirectories in PULS_dir
subdirs = [d for d in os.listdir(PULS_dir) if os.path.isdir(os.path.join(PULS_dir, d))]
for subdir in tqdm(subdirs):
    # look for .gb file in subdir
    gb_filename = [f for f in os.listdir(os.path.join(PULS_dir, subdir)) if f.endswith(".gb")][0]
    gb_path = os.path.join(PULS_dir, subdir, gb_filename)
    print(f"Parsing {gb_filename}...")

    # parse gb file, catch any exceptions and log them
    try:
        gb_file = gb_io.load(gb_path)
    except Exception as e:
        error_log.append(f"Error parsing .gb file {gb_path}: {e}")
        continue

    # check for duplicate accessions and skip if found
    for record in gb_file:
        accession = record.accession
        if accession in saved_accessions:
            print(f"Duplicate accession {accession} found in {gb_filename}, skipping...")
            continue
        
        # save record and accession
        saved_accessions.add(accession)
        all_records.append(record)

# dump into a single file
print(f"Total unique records parsed: {len(all_records)}")
output_path = "gLM-PUL-Prediction/DBCAN-PUL/PUL_genomes.gb"
gb_io.dump(all_records, output_path)

# save error log
if error_log:
    with open("gLM-PUL-Prediction/DBCAN-PUL/gb_parsing_errors.log", "w") as f:
        for error in error_log:
            f.write(error + "\n")