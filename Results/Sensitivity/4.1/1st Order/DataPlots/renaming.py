from pathlib import Path

if __name__ == "__main__":

    folder = Path("")

    for file in folder.iterdir():
        if file.is_file() and ".svg" in file.name:
            new_name = file.name.replace("Lambda", "theta").replace("phi", "lambda")
            file.rename(file.with_name(new_name))
