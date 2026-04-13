import json
import polars
from Bio import Entrez
from __future__ import annotations
from typing import Iterable, Optional, List, Iterator
import itertools
import os
import hashlib
from pathlib import Path
import pyhmmer

Entrez.email = "b.rosenthal@lumc.nl"

# Entrez utility functions
def request_summary(acc, db="nuccore"):
    handle = Entrez.esummary(
        db=db,
        id=acc,
        retmode="json"
    )
    record = handle.read()
    handle.close()
    # parse the JSON response to get the sequence length
    record = json.loads(record)
    return record

def request_sequence(acc, db="nuccore"):
    handle = Entrez.efetch(
        db=db,
        id=acc,
        rettype="gb",
        retmode="json",
        complexity=1
    )
    record = handle.read()
    handle.close()
    return record


def get_length(acc):
    # get esummary from ncbi
    record = request_summary(acc)

    if 'error' in record.keys():
        # try getting full sequence and parsing length from there
        record = request_sequence(acc)
        length = record.split('\n')[0].split()[2]
    else:
        uid = record['result']['uids'][0]
        length = record['result'][uid]['slen']

    try:
        length = int(length) 
    except ValueError:
        length = None

    return length


def get_sequence_lengths(unique_accessions: polars.DataFrame) -> list:
    lengths = []
    for acc in tqdm(list(unique_accessions['sequence_id']), desc="Fetching sequence lengths"):
        try:
            length = get_length(acc)
        except Exception as e:
            print(f"Error fetching length for {acc}, skipping.")
            length = None

        lengths.append({'sequence_id': acc, 'length': length})
        time.sleep(0.1)

    return lengths


def reset_start_end(table: polars.DataFrame) -> polars.DataFrame:
    return table.with_columns(
        polars.when(polars.col("start") < polars.col("end")).then(polars.col("start")).otherwise(polars.col("end")).alias("start"),
        polars.when(polars.col("start") < polars.col("end")).then(polars.col("end")).otherwise(polars.col("start")).alias("end"),
    )


def join_gene_and_PUL_table(gene_table: polars.DataFrame, cluster_table: polars.DataFrame, buffer: int = 100,) -> polars.DataFrame:
    gene_table = reset_start_end(gene_table)
    cluster_table = reset_start_end(cluster_table)

    labled_gene_table = (
        cluster_table
        .rename({"start": "pul_start", "end": "pul_end"}) # avoid column name conflicts
        .join(
            gene_table,
            on="sequence_id",
            how="inner",
            validate="m:m",
        )
        .with_columns(
            polars.when(
                polars.col("start") >= polars.col("pul_start") - buffer, # allow for some buffer around the PUL boundaries
                polars.col("end") <= polars.col("pul_end") + buffer,
            )
            .then(polars.col("cluster_id"))
            .otherwise(None)
            .alias("cluster_id"),
            polars.when(
                polars.col("start") >= polars.col("pul_start") - buffer,
                polars.col("end") <= polars.col("pul_end") + buffer,
            )
            .then(True)
            .otherwise(False)
            .cast(polars.Boolean)
            .alias("is_PUL")
        )
        # aggregate by protein_id to determine if protein is in any PUL
        .group_by("protein_id")
        .agg(
            polars.col("is_PUL").any().alias("is_PUL"),
            polars.col("sequence_id").first().alias("sequence_id"),
            polars.col("start").first().alias("start"),
            polars.col("end").first().alias("end"),
            polars.col("cluster_id").drop_nulls().first().alias("cluster_id")
        )
        .sort(by=["sequence_id", "start", "end"])
        .with_row_index(name="gene_id", offset=0)  # important
        .select(["sequence_id", "protein_id", "start", "end", "is_PUL", "cluster_id"])
    )

    return labled_gene_table


def recompute_length_percentage(cluster_table: polars.DataFrame) -> polars.DataFrame:
    # recompute length and percentage in PUL for all clusters, since we added new ones
    # recalculate sum of length of PULs per genome and percentage of genome in PULs
    pul_lengths = (
        cluster_table
        .group_by('sequence_id')
        .agg(
            (polars.col('end') - polars.col('start')).sum().alias('pul_length_sum'),
            polars.col('length').first(),
        ) # length of all puls in sequence, full sequence length
        .with_columns((100 * polars.col('pul_length_sum') / polars.col('length')).alias('percentage_in_puls')) # % of puls in genome
        .select('sequence_id', 'length', 'pul_length_sum', 'percentage_in_puls') # select only relevant columns
    )
    # merge back with cluster table
    cluster_table = (
        cluster_table
        .drop(['length', 'pul_length_sum', 'percentage_in_puls'])
        .join(pul_lengths, on='sequence_id', how='left')
    )

    return cluster_table


def report_pul_statistics():
    genes = polars.read_parquet("src/data/genecat_output/genome.genes.parquet")
    for k in range(5):
        test = f"src/data/splits/test_fold_{k}.tsv"
        train = f"src/data/splits/train_fold_{k}.tsv"
        test_df = polars.read_csv(test, separator='\t')
        train_df = polars.read_csv(train, separator='\t')

        # join dfs
        test_joined = join_gene_and_PUL_table(genes, test_df)
        train_joined = join_gene_and_PUL_table(genes, train_df)

        # true proportion
        print(f"Fold {k}:")
        print(f"  Train: {train_joined['is_PUL'].mean()}")
        print(f"  Test: {test_joined['is_PUL'].mean()}")




class HMMLoader(Iterable[pyhmmer.plan7.HMM]):
    def __init__(
        self,
        hmms: Iterable[pyhmmer.plan7.HMM],
        whitelist: Optional[List[str]] = None,
    ):  
        if whitelist is not None:
            # We need to materialize hmms since we iterate twice by necessity
            hmms = list(hmms)
            hmms.sort(key=lambda x: x.name)
            self.hmms = (
                hmm
                for hmm in hmms
                if hmm.name in whitelist
            )
            if not self.hmms:
                raise ValueError(
                    "No hmms passed the selection!"
                )
        else:
            self.hmms = hmms

    def __iter__(self) -> Iterator[pyhmmer.plan7.HMM]:
        return iter(self.hmms)

    @staticmethod
    def _read_hmm_from_file(path: Path|str) -> Iterator[pyhmmer.plan7.HMM]:
        """
        Read HMM from file
        :param path: path to the HMM file
        :return: None
        """
        with pyhmmer.plan7.HMMFile(path) as hmm_file:
            for hmm in hmm_file:
                yield hmm

    @staticmethod
    def md5_of_file(path: Path) -> str:
        hasher = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def read_hmms(
        self,
        file_with_paths: Optional[Path]=None,
        hmmdb_path: Optional[Path]=None,
        whitelist: Optional[List[str]] = None,
    ) -> HMMLoader:
        """
        Read HMMs from a either a file with paths to HMM files
        Or from a directory containing HMM files
        Or from a single HMM file
        :param file_with_paths: path to the file with paths
        :param hmmdb_path: path to directory of HMMs or single HMM file
        :return: 'HMMLoader'
        """

        if file_with_paths is None and hmmdb_path is not None:
            if hmmdb_path.is_file():
                return HMMLoader(
                    self._read_hmm_from_file(hmmdb_path),
                    whitelist=whitelist,
                )
            elif hmmdb_path.is_dir():
                return HMMLoader(
                    itertools.chain.from_iterable(
                        self._read_hmm_from_file(os.path.join(hmmdb_path, f))
                        for f in os.listdir(hmmdb_path)
                    ),
                    whitelist=whitelist,
                )
            else:
                raise ValueError(
                    f"Encounterd path which isnt a file or a dir at {str(hmmdb_path)}"
                )

        elif hmmdb_path is None and file_with_paths is not None:
            with open(file_with_paths, 'r') as _in:
                return HMMLoader(
                    itertools.chain.from_iterable(
                        self._read_hmm_from_file(f.strip())
                        for f in _in
                    ),
                    whitelist=whitelist,
                )

        else:
            raise ValueError(
                "Pass either a file with paths or a path to HMMLoader.read_hmms()"
            )

    def write_to_h3m_file(
        self,
        output: Path,
    ):
        """Write the hmms to a single h3m binary file"""
        output = Path(output).with_suffix(".selected.h3m")
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as f:
            for hmm in self.hmms:
                hmm.write(f, binary=True)
