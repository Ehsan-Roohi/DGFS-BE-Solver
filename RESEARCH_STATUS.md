# Fork research status

Last audit: 2026-08-18.  This repository is a fork of DGFS-BE; upstream
authorship, citation, and GPLv2 terms remain authoritative.

| PR | Work | Current status | Gate before merge |
| ---: | --- | --- | --- |
| [#1](https://github.com/Ehsan-Roohi/DGFS-BE-Solver/pull/1) | read-only kinetic diagnostics and troubled-cell sensor | draft; syntax and simulated lifecycle/self-tests passed | run the declared Mach-1.592 CUDA/Unity restart and validate CSV/moment behavior on the real solver |
| [#2](https://github.com/Ehsan-Roohi/DGFS-BE-Solver/pull/2) | JCP 2019 Figure 14(b) normal-shock reproduction | draft; exact-parameter/static case audit passed | complete the full CUDA calculation and compare the declared profile/residual with the published reference |

Neither draft should be described as a reproduced physical result before its
execution gate passes.  General fork maintenance must not obscure upstream
provenance.
