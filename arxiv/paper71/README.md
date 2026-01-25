# Paper 71 (Zenodo/Preprint Packaging)

Source: `71.The-Sanctuary-Inside-the-Black-Hole.md`

## Build PDF (Pandoc + XeLaTeX)

```bash
pandoc 71.The-Sanctuary-Inside-the-Black-Hole.md -o arxiv/paper71/paper71.pdf --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V mainfont="DejaVu Serif" \
  -V monofont="DejaVu Sans Mono" \
  -V CJKmainfont="Noto Serif CJK SC" \
  --variable=header-includes:"\\usepackage{xeCJK}"
```

Notes:
- If the build fails due to missing fonts, run `fc-list` to see available fonts and swap variables accordingly.
- The embedded figure path `assets/E8Petrie.svg` must exist for the PDF to include it.
