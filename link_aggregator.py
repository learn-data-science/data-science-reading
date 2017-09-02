# -*- coding: utf-8 -*-
"""Combine individual meetup paper lists into one file."""
from __future__ import print_function
import glob
from io import open  # pylint: disable=redefined-builtin
import time


_EXCLUDE_FILES = {"README.md", "archives_2016.md", "summary.md"}
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


def parse_files(files):
    """parse_files"""
    markdown_file = open("summary.md", "w", encoding='utf8')
    output_content = '# Papers suggested to date\n\n'
    for ffile in files:
        try:
            content = open(ffile, 'r', encoding='utf8').read()
            accepted, suggested = parse_markdown(content)
            output_content += "# %s\n\n## Chosen Paper\n%s\n\n" \
                              "## Other Suggestions\n%s\n\n" \
                              % (ffile[:-3], accepted, suggested)
        except Exception as exc:  # pylint: disable=broad-except
            print("Caught an exception: %s " % exc)
    markdown_file.write(output_content)


def main():
    """Execute from command line."""
    files = [f for f in glob.glob("*.md") if f not in _EXCLUDE_FILES]
    files.sort(key=lambda x: -time.mktime(time.strptime(x, "%Y-%m-%d.md")))
    parse_files(files)


if __name__ == '__main__':
    main()
