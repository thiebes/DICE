"""
Where the GUI saves results and opens its file dialogs.

Like the command-line interface, the GUI works in the directory it was
started from. It cannot always write there: macOS starts an application
from the Finder in the filesystem root, and a launcher can start it in a
read-only folder. The GUI then works in a DICE folder in the user's
Documents folder instead.
"""

import tempfile
from pathlib import Path


def output_root() -> Path:
    """
    Folder that holds the output folder of each filename slug.

    output/ in the working directory when the GUI can write there; otherwise
    DICE/output/ in the user's Documents folder, or in the home folder when
    there is no Documents folder. The fallback folder is created here, so a
    file dialog can open in it.

    Returns
    -------
    Path
        The output root; in the working directory it may not exist yet.
    """
    working_directory = Path.cwd()
    if is_usable_folder(working_directory):
        return working_directory / 'output'

    documents = Path.home() / 'Documents'
    user_folder = documents if documents.is_dir() else Path.home()
    root = user_folder / 'DICE' / 'output'
    root.mkdir(parents=True, exist_ok=True)
    return root


def is_usable_folder(folder: Path) -> bool:
    """
    Report whether the GUI can work in a folder.

    A filesystem root is never usable. For any other folder, the check
    creates and removes a temporary file, because os.access does not report
    the effect of Windows access-control lists.

    Returns
    -------
    bool
        True if the folder is not a filesystem root and a file can be created in it.
    """
    if folder.parent == folder:
        return False
    try:
        with tempfile.TemporaryFile(dir=folder):
            pass
    except OSError:
        return False
    return True
