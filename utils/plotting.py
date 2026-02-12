import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter


# Define a custom formatting function to display percentages
def percentage_formatter(x, pos):
    return f"{100 * x:.0f}%"


def create_plot(
    full_data: dict, axis, fields: list[str], labels: list[str] = None, ylab=None
):
    # extract needed data
    selected_data = []
    for field in fields:
        selected_data.append(full_data[field.lower()])

    epochs = np.arange(len(selected_data[0])) + 1
    # Set the y-axis tick formatter to use the custom formatting function
    axis.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    axis.set_xlabel("Epoch")
    if ylab is None:
        axis.set_ylabel("Value in %")
    else:
        axis.set_ylabel(ylab)
    axis.set_ylim(-0.05, 1.05)

    if labels is None:
        labels = fields
    for data, label in zip(selected_data, labels):
        axis.plot(epochs, data, label=label)

    axis.grid()
    axis.legend()


def create_micro_macro_plot(
    data: dict,
    show: bool = True,
    save_micro: str = None,
    save_macro: str = None,
    model_type: str = "",
):
    micro_fig, micro_ax = plt.subplots(figsize=(6, 4))
    macro_fig, macro_ax = plt.subplots(figsize=(6, 4))

    # Create plots for micro and macro data
    create_plot(
        data["micro avg"], axis=micro_ax, fields=["F1-Score", "Precision", "Recall"]
    )
    create_plot(
        data["macro avg"], axis=macro_ax, fields=["F1-Score", "Precision", "Recall"]
    )

    # adjust figure name etc.
    micro_ax.set_title(f"{model_type}{': ' if model_type != '' else ''}Micro Average")
    macro_ax.set_title(f"{model_type}{': ' if model_type != '' else ''}Macro Average")

    plt.tight_layout()

    if save_micro is not None:
        micro_fig.savefig(save_micro)
    if save_macro is not None:
        macro_fig.savefig(save_macro)

    if show:
        plt.show()


def create_model_comparison_plot(
    data: dict,
    model_types: list[str],
    show: bool = True,
    save_micro: str = None,
    save_macro: str = None,
    labels: list[str] = None,
):
    micro_fig, micro_ax = plt.subplots(figsize=(6, 4))
    macro_fig, macro_ax = plt.subplots(figsize=(6, 4))

    # Create plots for micro and macro data
    create_plot(
        data["micro avg"], axis=micro_ax, fields=model_types, ylab="F1-Score in %", labels=labels
    )
    create_plot(
        data["macro avg"], axis=macro_ax, fields=model_types, ylab="F1-Score in %", labels=labels
    )

    # adjust figure name etc.
    micro_ax.set_title(f"F1-Score Micro Average")
    macro_ax.set_title(f"F1-Score Macro Average")

    plt.tight_layout()

    if save_micro is not None:
        micro_fig.savefig(save_micro)
    if save_macro is not None:
        macro_fig.savefig(save_macro)

    if show:
        plt.show()
