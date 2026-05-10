Subject: Open-source reference implementation of your self-driving perfusion paper

Dear Drs. Gadiyar, [co-authors],

I hope this message finds you well.

I am writing to share an open-source Python library, **perfusio**
(https://github.com/lynchaos/perfusio), which I developed as a
peer-reviewable reference implementation of the methodology described in:

> Gadiyar, C. J., et al. (2026). Self-Driving Development of Perfusion
> Processes for Monoclonal Antibody Production.
> *Biotechnology and Bioengineering*, 123(2), 391–405.

`perfusio` implements:

- The stepwise Gaussian Process hybrid model (SW-GP, your Eqs. 1–6).
- Entity-embedding transfer learning across CHO cell lines (Hutter et al. 2021).
- All 11 Bayesian experimental design acquisitions (PI, EI, LogEI, UCB,
  qEI, qLogEI, qUCB, qEHVI, qNEHVI, qNParEGO).
- An online-retraining digital twin with OPC UA and SQL connectors.
- Reproducible generation of Figs. 4, 6, 7, and 8 of your paper.

The library is released under the Apache-2.0 licence and is accompanied by
full test coverage (>92% lines), CI on three operating systems, and
Sphinx documentation with MathJax-rendered theory pages.

I would be delighted if you could:
1. Review the implementation for accuracy against the paper.
2. Point out any discrepancies or missing details.
3. Optionally, co-author a software paper describing `perfusio`.

I am happy to add any acknowledgement or authorship as you see fit.

Warm regards,
Kemal Yaylali
Independent Researcher / Developer
https://kemal.yaylali.uk | support@yaylali.uk
GitHub: https://github.com/lynchaos
X: https://x.com/kmlyyll
LinkedIn: https://www.linkedin.com/in/kemalyaylali/
