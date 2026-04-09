# Project PUL prediction /w Ben

## Data processing for ML tasks

GeneCAT provides a unified framework. This excludes preperation of a cluster/label table.
For CAZymes we will use Cayman for annotation, since Georg prefers Cayman over dbCAN.

### Requirements
Genome sequences in `genbank`, `.fna` (nucleotide fasta) or `.embl` formats,
split into one file for train and one file for test genomes.

### Desired outputs
- GECCO and GeneCAT require GeneTables and FeatureTables
- Just for convinience we will also create a GenomeDatabase
- FeatureTables should be (seperately) annotated with Pfam37.1 and with Cayman CAZymes v0.12.0
- A merged Pfam and CAZy FeatureTable
- Cayman and Protein LMs require `.faa` (amino acid fasta) input

### Commands to preprecess data
```bash
export PYTHONPATH='/path/to/your/GeneCAT/src/'
# using vocab for GeneCAT multilabel model trained only on Pfams here
VOCAB=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level/BERT_train.fold_0.unique_domains.min50.Pfam37.1.vocab.txt
PFAM=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/Pfam/Pfam-A.Pfam37.1.h3m

# Creating the GeneTable and Pfam FeatureTable and GenomeDatabase (with only Pfam info)
python -m genecat.cli preprocess -i train_genomes.gbff -v $VOCAB --hmms $PFAM -o out --output-name PUL.train_fold_0.Pfam37.1.unique_domains --call-genes --index-db --write-tables --unique-domains
python -m genecat.cli preprocess -i test_genomes.gbff -v $VOCAB --hmms $PFAM -o out --output-name PUL.test_fold_0.Pfam37.1.unique_domains --call-genes --index-db --write-tables --unique-domains

# Re-doing GeneCalling to get the `.faa` sequence of proteins
python -m genecat.cli call-genes -i train_genomes.gbff -o out --output-name PUL.train_fold_0.genes --type faa+gff
python -m genecat.cli call-genes -i test_genomes.gbff -o out --output-name PUL.test_fold_0.genes --type faa+gff
```

### Commands to run Cayman
! Cayman will generate `.csv` tables. The `start` and `end` columns refer to
nucleotide positions from the start of the protein,
wile `domain_start` and `domain_end` refer to amino-acid positions.
I'd strongly advise saving these `.csv` files as proper parquet tables to encode the schema too.

```bash
# Running Cayman v0.12.0 from the `single_hmm_model_annotation` branch
# For annotations with CAZymes with (very) strict and properly evaluated thresholds
# https://github.com/zellerlab/cayman/tree/single_hmm_model_annotation

export PYTHONPATH='/exports/archive/lucid-grpzeller-primary/hackett/software/cayman'
CUTOFFS=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/CAZy/Cayman/cutoffs.csv
CAZY_HMMS=//exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/hmms/CAZy/Cayman/hmms_bin/cayman.v3.seed42_selected.h3m
NUM_THREADS=6 # by default it will run just one thread

python -m cayman annotate_proteome $CAZY_HMMS ./out/PUL.train_fold_0.genes.faa --cutoffs $CUTOFFS -o ./out/PUL.train_fold_0.features.cayman_cazy_v0.12.0.csv -t $NUM_THREADS
python -m cayman annotate_proteome $CAZY_HMMS ./out/PUL.test_fold_0.genes.faa --cutoffs $CUTOFFS -o ./out/PUL.test_fold_0.features.cayman_cazy_v0.12.0.csv -t $NUM_THREADS
```

Ideally bring the cayman output into FeatureTable format (code below)

### Merging the Pfam and the CAZyme Feature Tables
This is easy to do in polars if both tables are in the same schema...

```python
import polars

df = polars.read_parquet("PUL.features.Pfam37.p1e-9.parquet")

cazy = (
    polars.scan_csv(
        "PUL.features.cazy_cayman_v0.12.parquet",
        separator=",",
        has_header=True,
        schema_overrides={
            "sequenceID": polars.Utf8,
            "domain_start": polars.UInt32,
            "domain_end": polars.UInt32,
            "family": polars.Utf8,
            "pvalue": polars.Float64,
        }
    )
    .rename({
        "sequenceID": "protein_id",
        "family": "domain",
    })
    .drop(["start", "end"])
    .with_columns(
        hmm=polars.lit("Cayman_v0.12"),
        i_evalue=polars.lit(0).cast(dtype=polars.Float64),
    )
)
cazy.write_parquet("PUL.features.cazy_cayman_v0.12.parquet")

joined = df.vstack(cazy).sort(by=["sequence_id", "domain_start"])
joined.write_parquet("PUL.features.Pfam37.1_cazy_cayman_v0.12.p1e-9.parquet")
```

## Embedding genomes with GeneCAT
- (Obviously) this requires a trained GeneCAT model

### Trained GeneCAT models
I have two versions of GeneCAT modesls. One trained on only Pfams and one trained on Pfams+Cayman CAZymes:
Each model has its own associated VOCAB!

`BASEPATH=/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level`

Multilabel GeneCAT with only Pfam features
- `${BASEPATH}/BERT_train.fold_0.unique_domains.min50.Pfam37.1.vocab.txt`
- `${BASEPATH}/models_multilabel_models/model_gene_multilabel_untied_march_s4spvlec_v0.pt`

Multilabel GeneCAT with Pfam and Cayman CAZyme features
- `${BASEPATH}/BERT_train.fold_0.unique_domains.min50.Pfam37.1_cazy_cayman_v0.12.vocab.txt`
- `${BASEPATH}/models_multilabel_models/model_gene_multilabel_pfam_cazy_march_1tuwvwd1_v0.pt`

### Commands to embed genomes with GeneCAT

You can probably do this on a CPU even though it takes a bit longer.
Feel free to adapt `--batch-size` and `--jobs`.
`--outtypes df db` will generate the embeddings in parquet and in SQL EmbeddingDatabase format.
You can normalize the embeddings (per Gene) by adding the `--normalize` flag

Either generate the embeddings from a Gene+FeatureTable + Vocab combo
```bash
export PYTHONPATH='/path/to/your/GeneCAT/src/'

GENES=
FEATURES=
VOCAB=
MODEL=
python -m genecat.cli extract-embeddings -g $GENES -f $FEATURES -m $MODEL --vocab $VOCAB --batch-size 16 -j 1 -o PUL_embs --outtypes df db
```

Or generate the embeddings from an SQL GenomeDatabase
```bash
export PYTHONPATH='/path/to/your/GeneCAT/src/'

DB=
MODEL=
python -m genecat.cli extract-embeddings -i $DB --batch-size 16 -j 1 -o PUL_embs --outtypes df db

```

# OLD GENECAT MODEL
`/exports/archive/lucid-grpzeller-primary/hackett/GeneCat/data/data_split_class_level/models_multilabel_models/jan_model`