import polars
import matplotlib.pyplot as plt

no_mf = "src/data/results/genecat_fine_tuned/logs_genecat_fine_tuned/wandb/offline-run-20260422_112843-4q8y8k1i/files/fold_0_finetune_log_4q8y8k1i/version_0/metrics.csv"
mf = "src/data/results/genecat_fine_tuned/logs_genecat_fine_tuned/wandb/latest-run/files/fold_0_finetune_log_pctk1581/version_0/metrics.csv"
df = polars.read_csv(no_mf)

# Figure size in inches
cm = 1 / 2.54  # centimeters in inches
textwidth = 19 * cm

figsize = (textwidth*0.8, cm*8)
dpi = 300
xlim = None  # can set manually if you want
ylim_loss = (0, 1)
ylim_pred = (0, 6.5)
left, right, bottom, top = 0.1, 0.9, 0.15, 0.9  # fixed axes layout
fig, axs = plt.subplots(1, 1, figsize=figsize)
train_df = df.filter(polars.col("train_loss").is_not_null())

if "val_loss" in df.columns:
    val_df = df.filter(polars.col("val_loss").is_not_null())
    axs.scatter(val_df["step"].to_numpy(), val_df["val_loss"].cast(polars.Float32).to_numpy(),label="Validation Loss", color="tab:orange", s=25, zorder=20)

if "test_loss" in df.columns:
    test_df = df.filter(polars.col("test_loss").is_not_null())
    axs.scatter(test_df["step"].to_numpy(), test_df["test_loss"].cast(polars.Float32).to_numpy(), label="Test Loss", color="tab:green", s=25, zorder=20)

axs.plot(train_df["step"].to_numpy(), train_df["train_loss"].cast(polars.Float32).to_numpy(), label="Train Loss", linestyle='-', linewidth=1, zorder=10)

#axs.set_ylim(*ylim_loss)
axs.set_xlabel("Step")
axs.set_ylabel("Loss")
axs.legend(loc="upper right")
axs.set_title("GeneCAT PUL prediction")

#fig.patch.set_facecolor("none")
#axs.set_facecolor("none")
#fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
plt.tight_layout()
plt.savefig("src/data/plots/temp.png")
