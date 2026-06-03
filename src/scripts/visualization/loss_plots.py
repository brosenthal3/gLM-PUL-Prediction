import polars
import matplotlib.pyplot as plt
import os
from palettable.cartocolors.qualitative import Bold_10

def plot_loss(pretrained, untrained, out_path, model_name):
    df_pretrained = polars.read_csv(pretrained)
    df_untrained = polars.read_csv(untrained)

    # Figure size in inches
    cm = 1 / 2.54  # centimeters in inches
    textwidth = 19 * cm

    figsize = (textwidth, cm*16)
    dpi = 300
    xlim = None  # can set manually if you want
    ylim_loss = (0, 1)
    left, right, bottom, top = 0.1, 0.9, 0.15, 0.9  # fixed axes layout
    fig, axis = plt.subplots(2, 1, figsize=figsize, sharex=True)
    bold10 = Bold_10.mpl_colors

    for i, df in enumerate([df_pretrained, df_untrained]):
        print(df)
        print(df.columns)
        axs = axis[i]
        train_df = (
            df.filter(polars.col("train_loss").is_not_null())
        )

        val_df = df.filter(polars.col("val_loss").is_not_null())
        axs.plot(val_df["step"].to_numpy(), val_df["val_loss"].cast(polars.Float32).to_numpy(),label="Validation Loss", color=bold10[0], zorder=20)

        val_auprc_df = df.filter(polars.col("val_auprc").is_not_null())
        axs.plot(val_auprc_df["step"].to_numpy(), val_auprc_df["val_auprc"].cast(polars.Float32).to_numpy(), label="Validation AUPRC", color=bold10[3], zorder=20)

        val_auroc_df = df.filter(polars.col("val_auroc").is_not_null())
        axs.plot(val_auroc_df["step"].to_numpy(), val_auroc_df["val_auroc"].cast(polars.Float32).to_numpy(), label="Validation AUROC", color=bold10[4], zorder=20)

        test_df = df.filter(polars.col("test_loss").is_not_null())
        axs.scatter(test_df["step"].to_numpy(), test_df["test_loss"].cast(polars.Float32).to_numpy(), label="Test Loss", color=bold10[1], s=25, zorder=20)

        # test_auprc_df = df.filter(polars.col("test_auprc").is_not_null())
        # axs.scatter(test_auprc_df["step"].to_numpy(), test_auprc_df["test_auprc"].cast(polars.Float32).to_numpy(), label="Test AUPRC", color=bold10[2], s=25, zorder=20)

        axs.plot(train_df["step"].to_numpy(), train_df["train_loss"].cast(polars.Float32).to_numpy(), label="Train Loss", linestyle='-', linewidth=1, zorder=10, alpha=0.8, color=bold10[2])

        axs.set_ylabel("Loss")
        axs.set_ylim(ylim_loss)
        axs.grid(axis="y", alpha=0.5)
    
        axs.set_title("Loss in training - " + model_name + (" pretrained" if i==0 else " untrained"))

    axis[0].legend(loc="upper right")
    axis[1].set_xlabel("Step")
    print(f"Saving plot to {out_path}...")
    plt.tight_layout()
    plt.savefig(out_path)



if __name__ == "__main__":
    out_path = "results/plots/loss_plots/train_loss_pretrained_untrained.png"
    cazy_masked = "src/data/results/genecat_finetuned_cazy_masked/logs_fold_0/wandb/latest-run/files/cazy_fold_0_finetune_log_bqrf1ino/version_0/metrics.csv"
    untrained = "src/data/results/genecat_untrained/logs_fold_0/wandb/latest-run/files/cazy_fold_0_finetune_log_7t9fb3eu/version_0/metrics.csv"
    plot_loss(cazy_masked, untrained, out_path, "GeneCAT")