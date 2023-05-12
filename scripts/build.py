import argparse
import os
import shutil

from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--pre", action="store_true", help="Apply pre build actions")
group.add_argument("--post", action="store_true", help="Apply post build actions")
args = parser.parse_args()

# Define the paths of the file to modify and its temporary backup
file_name = "README.md"
backup_path = "scripts"
github_repository = "https://github.com/maichmueller/cfrainbow"

if args.pre:
    # Make a backup of the original file
    shutil.copy2(os.path.join(".", file_name), os.path.join(".", backup_path))

    with open(os.path.join(".", file_name), "r") as readme_file:
        readme = readme_file.readlines()
        readme_banner_html = "\n".join(readme[:5])  # we only need the first 4 lines
        remaining_readme = readme[5:]

        # Parse the HTML code
        soup = BeautifulSoup(readme_banner_html, "html.parser")

        # Find the 'source' tag with 'media' attribute matching '(prefers-color-scheme: light)'
        source_tag = soup.find(
            "source", attrs={"media": "(prefers-color-scheme: light)"}
        )

        # Extract the 'srcset' attribute value
        image_repo_path = source_tag["srcset"]

        raw_image_url = github_repository.rstrip("/") + "/raw/main/" + image_repo_path
        # Output the markdown format with the image URL
        markdown_output = f"![Readme banner.]({raw_image_url})\n"

        replacement_readme = "".join([markdown_output] + remaining_readme)

    with open(os.path.join(".", file_name), "w") as readme_file:
        readme_file.write(replacement_readme)

if args.post:
    # Restore the original file from the backup
    shutil.move(os.path.join(".", backup_path, file_name), os.path.join(".", file_name))
