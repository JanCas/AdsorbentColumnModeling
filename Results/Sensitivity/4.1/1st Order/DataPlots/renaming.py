"""One-off bulk-rename of SVG plot files after a parameter-symbol relabel.

Renames every ``*.svg`` in `folder` in place, swapping the old sensitivity-axis
names for the current ones: "Lambda" -> "theta" and "phi" -> "lambda". Run once
after the non-dimensional parameters were renamed so existing figure filenames
match the new symbol convention. Set `folder` to the directory to process."""
from pathlib import Path

if __name__ == "__main__":

    folder = Path("")

    for file in folder.iterdir():
        if file.is_file() and ".svg" in file.name:
            new_name = file.name.replace("Lambda", "theta").replace("phi", "lambda")
            file.rename(file.with_name(new_name))
