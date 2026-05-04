import polars
import matplotlib.pyplot as plt
import os
from palettable.cartocolors.qualitative import Bold_10

out_path = "src/results/plots/genecat_fine_tuned"
os.makedirs(out_path, exist_ok=True)

pfam = "src/data/results/genecat_fine_tuned/logs_genecat_fine_tuned/wandb/offline-run-20260429_150219-gtv4ep3n/files/pfam_fold_0_finetune_log_gtv4ep3n/version_0/metrics.csv"

cazy = "src/data/results/genecat_fine_tuned/logs_genecat_fine_tuned/wandb/offline-run-20260429_221043-bgckiapc/files/pfam_cazy_fold_0_finetune_log_bgckiapc/version_0/metrics.csv"

df_pfam = polars.read_csv(pfam)
df_cazy = polars.read_csv(cazy)

# Figure size in inches
cm = 1 / 2.54  # centimeters in inches
textwidth = 19 * cm

figsize = (textwidth, cm*16)
dpi = 300
xlim = None  # can set manually if you want
ylim_loss = (0, 1)
ylim_pred = (0, 6.5)
left, right, bottom, top = 0.1, 0.9, 0.15, 0.9  # fixed axes layout
fig, axis = plt.subplots(2, 1, figsize=figsize)
bold10 = Bold_10.mpl_colors

for i, df in enumerate([df_pfam, df_cazy]):
    axs = axis[i]
    train_df = (
        df.filter(polars.col("train_loss").is_not_null())
    )

    if "val_loss" in df.columns:
        val_df = df.filter(polars.col("val_loss").is_not_null())
        axs.scatter(val_df["step"].to_numpy(), val_df["val_loss"].cast(polars.Float32).to_numpy(),label="Validation Loss", color="tab:orange", s=25, zorder=20)

    if "test_loss" in df.columns:
        test_df = df.filter(polars.col("test_loss").is_not_null())
        axs.scatter(test_df["step"].to_numpy(), test_df["test_loss"].cast(polars.Float32).to_numpy(), label="Test Loss", color="tab:green", s=25, zorder=20)

    axs.plot(train_df["step"].to_numpy(), train_df["train_loss"].cast(polars.Float32).to_numpy(), label="Train Loss", linestyle='-', linewidth=1, zorder=10)

    axs.set_xlabel("Step")
    axs.set_ylabel("Loss")
    axs.legend(loc="upper right")
    axs.set_title("Loss in training GeneCAT for PUL prediction " + ("(Pfam features)" if i==0 else "(CAZy+Pfam features)"))

# axs2 = axs.twinx()
# axs.set_zorder(axs2.get_zorder() + 1)
# axs.patch.set_visible(False)

# axs2.plot(train_df["step"], train_df["train_micro_accuracy"], linestyle="-", alpha=0.3, color=bold10[0], zorder=1)
# axs2.plot(train_df["step"], train_df["train_macro_accuracy"], linestyle="-", alpha=0.3, color=bold10[1], zorder=1)
# axs2.plot(train_df["step"], train_df["train_micro_accuracy_smooth"], label="Micro Acc", linestyle="--", color=bold10[0], zorder=2)
# axs2.plot(train_df["step"], train_df["train_macro_accuracy_smooth"], label="Macro Acc", linestyle="--", color=bold10[1], zorder=2)
# axs2.set_ylabel("Accuracy", color='k')
# axs2.set_ylim(0, 1)
# axs2.legend(loc="upper right")
# axs2.tick_params(axis="y")

print("saving plot?")
plt.tight_layout()
plt.savefig(f"{out_path}/train_loss.png")
