from palettable.scientific.diverging import Cork_7
from palettable.scientific.sequential import Bilbao_5, Buda_4
from palettable.cartocolors.qualitative import Bold_10

model_names = {
    "gecco_pfam": "GECCO (Pfam)",
    "gecco_cazy": "GECCO (CAZy)",
    "genecat_zeroshot_pfam": "GeneCAT 0-Shot (Pfam)",
    "genecat_zeroshot_cazy": "GeneCAT 0-Shot (CAZy)",
    "genecat_zeroshot_pfam_masked": "GeneCAT 0-Shot Masked (Pfam)",
    "genecat_zeroshot_cazy_masked": "GeneCAT 0-Shot Masked (CAZy)",
    "genecat_finetuned_pfam": "GeneCAT Finetuned (Pfam)",
    "genecat_finetuned_cazy": "GeneCAT Finetuned (CAZy)",
    "genecat_finetuned_pfam_masked": "GeneCAT Finetuned Masked (Pfam)",
    "genecat_finetuned_cazy_masked": "GeneCAT Finetuned Masked (CAZy)",
    "esmc": "ESM-C",
    "bacformer": "Bacformer",
    "esmc_masked": "ESM-C Masked",
    "bacformer_masked": "Bacformer Masked",
    "genecat_untrained": "GeneCAT Untrained",
    "experimental": "Experimental PULs",
}

model_names_masked = {
    "gecco_pfam": "GECCO",
    "genecat_zeroshot_cazy": "GeneCAT 0-Shot",
    "genecat_zeroshot_cazy_masked": "GeneCAT 0-Shot Masked",
    "genecat_finetuned_cazy": "GeneCAT Finetuned",
    "genecat_finetuned_cazy_masked": "GeneCAT Finetuned Masked",
    "esmc": "ESM-C",
    "bacformer": "Bacformer",
    "esmc_masked": "ESM-C Masked",
    "bacformer_masked": "Bacformer Masked",
    "genecat_untrained": "GeneCAT Untrained"
}

model_names_features = {
    "gecco_pfam": "GECCO (Pfam)",
    "gecco_cazy": "GECCO (CAZy)",
    "genecat_zeroshot_pfam_masked": "GeneCAT 0-Shot (Pfam)",
    "genecat_zeroshot_cazy_masked": "GeneCAT 0-Shot (CAZy)",
    "genecat_finetuned_cazy_masked": "GeneCAT Finetuned (CAZy)",
    "genecat_finetuned_pfam_masked": "GeneCAT Finetuned (Pfam)",
}


Buda_4 = Buda_4.mpl_colors
Cork_7 = Cork_7.mpl_colors
Bilbao_5 = Bilbao_5.mpl_colors
Bold_10 = Bold_10.mpl_colors

model_colors_2 = {
    "gecco_pfam": Bold_10[0],
    "gecco_cazy": Bold_10[1],
    "genecat_zeroshot_cazy": Bold_10[2],
    "genecat_zeroshot_pfam_masked":  Bold_10[3],
    "genecat_zeroshot_cazy_masked":  Bold_10[4],
    "genecat_untrained": Bold_10[5],
    "genecat_finetuned_cazy":  Bold_10[6],
    "genecat_finetuned_pfam_masked": Bold_10[7],
    "genecat_finetuned_cazy_masked": Bold_10[8],
    "esmc": Cork_7[1],
    "esmc_masked": Cork_7[2],
    "bacformer": Cork_7[-1],
    "bacformer_masked": Cork_7[-2],
}

model_colors = {
    "gecco_pfam": Buda_4[0],
    "gecco_cazy": Buda_4[1],
    "genecat_zeroshot_cazy": Cork_7[0],
    "genecat_zeroshot_pfam_masked":  Cork_7[1],
    "genecat_zeroshot_cazy_masked":  Cork_7[2],
    "genecat_untrained": Cork_7[3],
    "genecat_finetuned_cazy":  Cork_7[4],
    "genecat_finetuned_pfam_masked": Cork_7[5],
    "genecat_finetuned_cazy_masked": Cork_7[6],
    "esmc": Bilbao_5[1],
    "esmc_masked": Bilbao_5[2],
    "bacformer": Bilbao_5[3],
    "bacformer_masked": Bilbao_5[4],
    "experimental": Cork_7[0],
}