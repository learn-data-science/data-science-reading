# -*- coding: utf-8 -*-
import glob
import time

_EXCLUDE_FILES = {"README.md", "archives_2016.md", "summary.md"}
_ACCEPTED_PAPER_HEADING = "## Chosen paper"
_OTHER_PAPERS_HEADING = "## Other suggestions"


def parse_markdown(markdown_content):
    accepted_match = markdown_content.find(_ACCEPTED_PAPER_HEADING) \
                     + len(_ACCEPTED_PAPER_HEADING)
    other_suggestions_match = markdown_content.find(_OTHER_PAPERS_HEADING)

    accepted_paper = markdown_content[accepted_match:other_suggestions_match]
    suggested_papers = markdown_content[(other_suggestions_match +
                                         len(_OTHER_PAPERS_HEADING)):]

    return accepted_paper.strip(), suggested_papers.strip()


def parse_files(files):
    markdown_file = open("summary.md", "w")
    output_content = '# Papers suggested to date\n\n'
    for ffile in files:
        try:
            content = open(ffile, 'r').read()
            accepted, suggested = parse_markdown(content)
            output_content += "# %s\n\n##Chosen Paper\n%s\n\n" \
                              "##Other Suggestions\n%s\n\n" \
                              % (ffile[:-3], accepted, suggested)
        except Exception as e:
            print "Caught an exception: %s " % e
    markdown_file.write(output_content)


if __name__ == '__main__':
    files = [f for f in glob.glob("*.md") if f not in _EXCLUDE_FILES]
    files.sort(key=lambda x: -time.mktime(time.strptime(x, "%Y-%m-%d.md")))
    parse_files(files)
