# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(1, os.path.abspath('../notebooks'))



# -- Project information -----------------------------------------------------

project = 'OpenVQE'
copyright = '2022, Mohammad HAIDAR'
author = 'Mohammad HAIDAR'
latex_engine = "pdflatex"
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    "papersize": "a4paper",
    # Sonny, Lenny, Glenn, Conny, Rejne, Bjarne and Bjornstrup
    # 'fncychap': '\\usepackage[Lenny]{fncychap}',
    "fncychap": "\\usepackage{fncychap}",
    "fontpkg": "\\usepackage{amsmath,amsfonts,amssymb,amsthm}",
    "figure_align": "htbp",
    # The font size ('10pt', '11pt' or '12pt').
    #
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    #
    "preamble": r"""
        %%%%%%%%%%%%%%%%%%%% Meher %%%%%%%%%%%%%%%%%%
        %%%add number to subsubsection 2=subsection, 3=subsubsection
        %%% below subsubsection is not good idea.
        \setcounter{secnumdepth}{3}
        %
        %%%% Table of content upto 2=subsection, 3=subsubsection
        \setcounter{tocdepth}{2}
        \usepackage{amsmath,amsfonts,amssymb,amsthm}
        \usepackage{graphicx}
        %%% reduce spaces for Table of contents, figures and tables
        %%% it is used "\addtocontents{toc}{\vskip -1.2cm}" etc. in the
        %%% document
        \usepackage[notlot,nottoc,notlof]{}
        \usepackage{color}
        \usepackage{transparent}
        \usepackage{eso-pic}
        \usepackage{lipsum}
        \usepackage{footnotebackref} %%link at the footnote to go to the place
                                     %%of footnote in the text
        %% spacing between line
        \usepackage{setspace}
        %%%%\onehalfspacing
        %%%%\doublespacing
        \singlespacing
        %%%%%%%%%%% datetime
        \usepackage{datetime}
        \newdateformat{MonthYearFormat}{%
            \monthname[\THEMONTH], \THEYEAR}
        %% RO, LE will not work for 'oneside' layout.
        %% Change oneside to twoside in document class
        \usepackage{fancyhdr}
        \pagestyle{fancy}
        \fancyhf{}
        %%% Alternating Header for oneside
        \fancyhead[L]{
            \includegraphics[width=1.2cm, trim=0 40cm 0 0]{Logo_Atos_RGB.png}
            \hskip 0.5cm
            \copyright Atos 2016-2020
        }
        \fancyhead[R]{
        \ifthenelse{\isodd{\value{page}}}
            {\small \nouppercase{\leftmark}}
            {\small \nouppercase{\rightmark}}
        }
        %%% page number
        \fancyfoot[CO, CE]{\thepage}
        \renewcommand{\headrulewidth}{0.5pt}
        \renewcommand{\footrulewidth}{0.5pt}
        \RequirePackage{tocbibind} %%% comment this to remove page number
                                   %%% for following
        \addto\captionsenglish{\renewcommand{\contentsname}{Table of contents}}
        % \addto\captionsenglish{\renewcommand{\listfigurename}{List of figures}}
        % \addto\captionsenglish{\renewcommand{\listtablename}{List of tables}}
        % \addto\captionsenglish{\renewcommand{\chaptername}{Chapter}}
        %%reduce spacing for itemize
        \usepackage{enumitem}
        \setlist{nosep}
        %%%%%%%%%%% Quote Styles at the top of chapter
        \usepackage{epigraph}
        \setlength{\epigraphwidth}{0.8\columnwidth}
        \newcommand{\chapterquote}[2]{\epigraphhead[60]{\epigraph{\textit{#1}}{\textbf {\textit{--#2}}}}}
        %%%%%%%%%%% Quote for all places except Chapter
        \newcommand{\sectionquote}[2]{{\quote{\textit{``#1''}}{\textbf {\textit{--#2}}}}}
        \fancypagestyle{plain}{%
        \fancyhead[L]{
            \includegraphics[width=1.2cm, trim=0 40cm 0 0]{Logo_Atos_RGB.png}
            \hskip 0.5cm
            \copyright Atos 2016-2020
        }
        \fancyfoot[C]{\thepage}%
        \fancyfoot[R]{}
  \renewcommand{\headrulewidth}{0pt}% Line at the header invisible
  \renewcommand{\footrulewidth}{0.4pt}% Line at the footer visible
  }
    """,
    "maketitle": r"""
        \pagenumbering{Roman} %%% to avoid page 1 conflict with actual page 1
        \begin{titlepage}
            \centering
            \vspace*{40mm} %%% * is used to give space from top
            \textbf{\Large {Quantum Learning Machine Documentation}}
            \vspace{0mm}
            \small{Quantum Application Toolset "QAT"}
            \vspace{0mm}
            \begin{figure}[!h]
                \centering
                \includegraphics[scale=0.3]{QLM.png}
            \end{figure}
            \vspace{0mm}
            \large \textbf{{Atos Quantum Team}}
            \vspace*{0mm}
            \small  Created on : \MonthYearFormat\today
            \vspace*{0mm}
            \small  Release: 0.8\_snapshot
            %% \vfill adds at the bottom
            \vfill
            \small \textit{More documents are freely available at }{\href{https://atos.net/fr/vision-et-innovation/atos-quantum}{Atos Quantum Project}}
        \end{titlepage}
        \clearpage
        \pagenumbering{roman}
        \tableofcontents
        %\listoffigures
        %\listoftables
        \clearpage
        \pagenumbering{arabic}
        """,
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
    "sphinxsetup": "hmargin={0.7in,0.7in}, vmargin={1in,1in}, \
        verbatimwithframe=true, \
        TitleColor={rgb}{0,0,0}, \
        HeaderFamily=\\rmfamily\\bfseries, \
        InnerLinkColor={rgb}{0,0,1}, \
        OuterLinkColor={rgb}{0,0,1}",
    "tableofcontents": " ",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author,
# documentclass [howto/manual]).
latex_documents = [
    (
        "index",
        "myqlm-fermion.tex",
        "MyQLM fermion Documentation",
        "Atos Quantum Lab",
        "manual",
    ),
]

# The full version, including alpha/beta/rc tags
release = '0.0.1'

master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',       # Test snippets in documentation
    'sphinx.ext.todo',          # to-do syntax highlighting
    'sphinx.ext.ifconfig',      # Content based configuration
    'm2r2',                     # Markdown support 
    "nbsphinx",
]

source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
