import requests
import polars
from tqdm import tqdm
from time import sleep

def get_pfam_description(pfam_id):
    try:
        url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfam_id}"
        r = requests.get(url, timeout=5)

        if r.status_code != 200:
            return None

        data = r.json()
        name = data["metadata"]["name"]
        description = data["metadata"]["description"]["text"]
        return name, description

    except Exception:
        print(f"Error fetching description for {pfam_id}")
        return None

def get_pfam_descriptions(domains_df):
    domains = domains_df["domain"].unique().to_list()
    print(f"Fetching descriptions for {len(domains)} PFAM domains...")

    pfam_metadata = []
    for domain in tqdm(domains):
        desc = get_pfam_description(domain)
        if desc is not None:
            pfam_metadata.append({
                "domain": domain,
                "description": desc
            })
        else:
            pfam_metadata.append({
                "domain": domain,
                "description": (None, None)
            })
        sleep(0.05) # be nice to the API

    pfam_metadata_df = polars.DataFrame({
        "domain": domains,
        "name": [desc[0] for desc in pfam_metadata["description"]],
        "description": [desc[1] for desc in pfam_metadata["description"]]
    })

    pfam_metadata_df.write_parquet("src/data/analysis/pfam_metadata.parquet")

if __name__ == "__main__":
    features = polars.read_parquet("src/data/genecat_output/pfam.features.parquet")
    sequences = polars.read_csv("src/data/data_collection/clusters_deduplicated_cblaster.tsv", sep="\t")
    
    selected_features = features.join(sequences.select("sequence_id"), how="inner", left_on="protein_id", right_on="sequence_id")
    get_pfam_descriptions(selected_features)