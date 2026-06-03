import polars
import matplotlib.pyplot as plt
import os
from palettable.cartocolors.qualitative import Bold_10

def plot_loss(pretrained, untrained, out_path, model_name):
    df_pretrained = polars.read_csv(pretrained)
    df_untrained = polars.read_csv(untrained)
    # step where each epoch starts
    epoch_ticks = (
        df_untrained.group_by(polars.col("epoch"))
        .agg(
            polars.col("step").min().alias("step")
        )
        .sort("epoch")
    )

    # Figure size in inches
    cm = 1 / 2.54  # centimeters in inches
    textwidth = 22 * cm

    figsize = (textwidth, cm*15)
    dpi = 300
    xlim = None  # can set manually if you want
    left, right, bottom, top = 0.1, 0.9, 0.15, 0.9  # fixed axes layout
    fig, axis = plt.subplots(2, 1, figsize=figsize, sharex=True)
    bold10 = Bold_10.mpl_colors
    ylim_loss = max(df_pretrained["train_loss"].max(), df_untrained["train_loss"].max()) 

    for i, df in enumerate([df_pretrained, df_untrained]):
        ax_left = axis[i]
        ax_right = ax_left.twinx()
        train_df = (
            df.filter(polars.col("train_loss").is_not_null())
        )

        # loss
        val_df = df.filter(polars.col("val_loss").is_not_null())
        val_loss_plot = ax_left.plot(val_df["step"].to_numpy(), val_df["val_loss"].cast(polars.Float32).to_numpy(), label="Validation Loss", color=bold10[0], zorder=20)
        test_df = df.filter(polars.col("test_loss").is_not_null())
        test_loss_plot = ax_left.scatter(test_df["step"].to_numpy(), test_df["test_loss"].cast(polars.Float32).to_numpy(), label="Test Loss", color=bold10[1], s=25, zorder=20)
        train_loss_plot = ax_left.plot(train_df["step"].to_numpy(), train_df["train_loss"].cast(polars.Float32).to_numpy(), label="Train Loss", linestyle='-', linewidth=1, zorder=10, color=bold10[2])

        # auprc and auroc
        val_auprc_df = df.filter(polars.col("val_auprc").is_not_null())
        auprc_plot = ax_right.plot(val_auprc_df["step"].to_numpy(), val_auprc_df["val_auprc"].cast(polars.Float32).to_numpy(), label="Validation AUPRC", color=bold10[3], zorder=20)
        val_auroc_df = df.filter(polars.col("val_auroc").is_not_null())
        auroc_plot = ax_right.plot(val_auroc_df["step"].to_numpy(), val_auroc_df["val_auroc"].cast(polars.Float32).to_numpy(), label="Validation AUROC", color=bold10[5], zorder=20)

        ax_left.set_ylabel("Loss")
        ax_left.set_ylim(0, ylim_loss)
        ax_left.set_title("Loss in training - " + model_name + (" pretrained" if i==0 else " untrained"))
        ax_right.set_ylabel("Performance")
        ax_left.grid(axis="y", alpha=0.4, linestyle="--")

    # for some reason matplotlib returns list of lines instead of line? so need to extract them
    handles = [h[0] if type(h) == list else h for h in [auroc_plot, auprc_plot, train_loss_plot, val_loss_plot, test_loss_plot]]
    axis[0].legend(loc="upper right", handles=handles)
    axis[1].set_xticks(epoch_ticks["step"][:-1])
    axis[1].set_xticklabels(epoch_ticks["epoch"][:-1])
    axis[1].set_xlabel("Epoch")
    print(f"Saving plot to {out_path}...")
    plt.tight_layout()
    plt.savefig(out_path)



if __name__ == "__main__":
    out_path = "results/plots/loss_plots/train_loss_pretrained_untrained.png"
    cazy_masked = "src/data/results/genecat_finetuned_cazy_masked/logs_fold_0/wandb/latest-run/files/cazy_fold_0_finetune_log_bqrf1ino/version_0/metrics.csv"
    untrained = "src/data/results/genecat_untrained/logs_fold_0/wandb/latest-run/files/cazy_fold_0_finetune_log_7t9fb3eu/version_0/metrics.csv"
    plot_loss(cazy_masked, untrained, out_path, "GeneCAT")