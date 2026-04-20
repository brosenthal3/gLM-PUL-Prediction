import contextlib
import typing
import warnings
from pathlib import Path

import rich
import torch

if typing.TYPE_CHECKING:
    ShowWarning = typing.Callable[
        [
            typing.Union[Warning, str],
            typing.Type[Warning],
            str,
            int,
            typing.Optional[typing.TextIO],
            typing.Optional[str],
        ],
        None,
    ]


@contextlib.contextmanager
def patch_showwarnings(new_showwarning: "ShowWarning") -> typing.Iterator[None]:
    """
    Make a context patching `warnings.showwarning` with the given function.
    """
    old_showwarning: "ShowWarning" = warnings.showwarning
    try:
        warnings.showwarning = new_showwarning  # type: ignore
        yield
    finally:
        warnings.showwarning = old_showwarning  # type: ignore


def show_device_summary():
    # show a summary of available devices
    rich.print("Checking platform devices")
    devices = rich.table.Table("", "Name", "CUDA", "Memory", "Cores")
    for device in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(device)
        devices.add_row(
            str(device),
            properties.name,
            f"{properties.major}.{properties.minor}",
            rich.filesize.decimal(properties.total_memory),
            str(properties.multi_processor_count),
        )
    rich.print(devices)


def create_versioned_path(filepath: Path):
    if not filepath.parent.is_dir():
        filepath.parent.mkdir(parents=True, exist_ok=True)

    version = 0
    while (
        path := Path(filepath.parent, f"{filepath.stem}_v{version}{filepath.suffix}")
    ).exists():
        version += 1

    return path
