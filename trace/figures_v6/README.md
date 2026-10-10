# TRACE figures, manuscript presentation v6

These four drawings accompany *TRACE: When high discrimination fails to improve
quantized code screening*. They use the unchanged statement-aware-v3 evidence.

| File prefix | Manuscript display |
| --- | --- |
| Fig1_TRACE_workflow | Target checks, review capacity, passing-target ordering and risk calibration |
| Fig2_validation_cost_frontier | Acquisition time and excess residual errors relative to Public/NLL |
| Fig3_guarded_capacity_curve | Additional errors captured across all 175 queue sizes in six conditions |
| FigS1_retrospective_differences | Ten paired retrospective estimates and original 95% intervals |

PDF and SVG contain editable vectors and embedded/live text. PNG is 600 dpi;
TIFF is 1000 dpi. Python/Matplotlib drawing code and three plotting CSVs are in
`figure_source/`. From this folder:

```sh
python figure_source/build_visual_figures.py --out-dir reproduced
```

Install `matplotlib`, `numpy` and `pandas` if needed. This command performs no
new inference, bootstrapping or calibration. Figure 2 subtracts the Public/NLL
residual-error coordinate within each condition; absolute errors remain in the
validation tables. Coincident method points retain their exact coordinates.
The two six-panel grids are alignment-checked at a 1.5-point tolerance.

The drawing code was developed with OpenAI Codex assistance and reviewed by
the authors. Project code uses MIT; derived figures and tables use CC BY 4.0.
