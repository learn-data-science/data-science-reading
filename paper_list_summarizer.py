# -*- coding: utf-8 -*-
"""Combine individual meetup paper lists into one file."""
from __future__ import print_function
import glob
from io import open  # pylint: disable=redefined-builtin
import time


_EXCLUDE_FILES = {
    "README.md",
    "archives_2016.md",
    "summary.md",
    "presentation_tips.md"}
_ACCEPTED_PAPER_HEADING = "## Chosen paper"
_OTHER_PAPERS_HEADING = "## Other suggestions"


def parse_markdown(markdown_content):
    """parse_markdown"""
    accepted_match = markdown_content.find(_ACCEPTED_PAPER_HEADING) \
                     + len(_ACCEPTED_PAPER_HEADING)
    other_suggestions_match = markdown_content.find(_OTHER_PAPERS_HEADING)

    accepted_paper = markdown_content[accepted_match:other_suggestions_match]
    suggested_papers = markdown_content[(other_suggestions_match +
                                         len(_OTHER_PAPERS_HEADING)):]

    return accepted_paper.strip(), suggested_papers.strip()


def parse_files(files, archive_file=None):
    """parse_files"""
    try:
        with open("summary.md", "w", encoding="utf8") as markdown_file:
            output_content = "# Papers suggested to date\n\n"
            for ffile in files:
                content = open(ffile, "r", encoding="utf8").read()
                accepted, suggested = parse_markdown(content)
                output_content += "# %s\n\n## Chosen Paper\n%s\n\n" \
                                  "## Other Suggestions\n%s\n\n" \
                                  % (ffile[:-3], accepted, suggested)
            if archive_file:
                archive_content = open(archive_file, "r", encoding="utf8").read()
                output_content += archive_content
            markdown_file.write(output_content)
    except Exception as exc:  # pylint: disable=broad-except
        print("Bailing. Caught an exception: %s" % exc)


def main():
    """Execute from command line."""
    files = [f for f in glob.glob("*.md") if f not in _EXCLUDE_FILES]
    # hack: sorts files by date in the title
    files.sort(key=lambda x: -time.mktime(time.strptime(x, "%Y-%m-%d.md")))
    parse_files(files, archive_file="archives_2016.md")


if __name__ == '__main__':
    main()
