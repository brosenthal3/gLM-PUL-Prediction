"""
Create a file with descriptions and names for all pfam domains.
"""

import requests
import polars
from tqdm import tqdm
from time import sleep
import os

def get_pfam_description(pfam_id):
    try:
        url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfam_id}"
        r = requests.get(url, timeout=5)

        if r.status_code != 200:
            print("Status code not 200")
            return (None, None)

        data = r.json()
        name = data["metadata"]["name"].get("name", None)
        description = data["metadata"]["description"]
        if description is not None:
            description = description[0].get("text", None).strip("<p>").strip("</p>")

        return name, description

    except Exception as e:
        print(e)
        print(f"Error fetching description for {pfam_id}")
        return (None, None)


def get_pfam_descriptions(domains_df):
    domains = domains_df["domain"].unique().to_list()
    print(f"Fetching descriptions for {len(domains)} PFAM domains...")

    pfam_metadata = []
    for domain in tqdm(domains):
        name, desc = get_pfam_description(domain)
        pfam_metadata.append({
            "domain": domain,
            "name": name,
            "description": desc
        })
        sleep(0.05) # be nice to the API

    pfam_metadata_df = polars.DataFrame({
        "domain": domains,
        "name": [desc["name"] for desc in pfam_metadata],
        "description": [desc["description"] for desc in pfam_metadata]
    })
    os.makedirs("src/data/analysis", exist_ok=True)
    pfam_metadata_df.write_parquet("src/data/analysis/pfam_metadata.parquet")

if __name__ == "__main__":
    features = polars.read_parquet("src/data/genecat_output/pfam.features.parquet")
    sequences = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", separator="\t")

    print(features)    
    selected_features = features.join(sequences.select("sequence_id"), how="semi", on="sequence_id")
    print(selected_features)
    get_pfam_descriptions(selected_features)
