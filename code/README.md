# Code and reproduction

## Verify the delivered evidence

From the package root run `python code/verify_package.py`. The default checks use the Python standard library. Install Shapely to enable the additional independent geometric checks. The script writes verification/package_verification.json; it does not run optimizers or change result files.

## Controlled optimizer replay

The eight core files match data/controlled/manifest.json byte for byte. The original runner resolves its project root one directory above water_plant_layout and writes paper/revision_final_experiments. For an isolated replay, copy code/water_plant_layout to <replay_root>/water_plant_layout, copy the archived controlled files to <replay_root>/paper/revision_final_experiments, and read `python water_plant_layout/run_revision_final_experiments.py --help` before running. Keep the archive in this distribution unchanged. The original analysis uses the same convention. The environment recorded by the run is Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, and pymoo 0.6.1.6; requirement files are retained as historical specifications. The solver reference separately requires OR Tools 9.14.6206.

## Complete algorithm and synthetic adaptation

water_plant_layout_v3.py defines the full engineering fitness and island search. compare_rl_mechanism.py defines the complete-mechanism ablation, whose disabled branch uses ordinary mutation. revision_ga.py and revision_rl.py define the separate matched four-action one-step experiment. These comparisons are different experiments.

process_projection_rrp_ga_20261016 contains the frozen process-skeleton construction, projection repair and ancillary RRP fill implementation; its imports retain the recorded relative project layout. The frozen per-instance inputs and full layout CSVs are in data/synthetic. Workflow scripts may refer to their original project directories; they are provided as method source, not advertised as one-command fresh installations. Use the verification script for a portable audit of the delivered records.

The seven real-source candidates are supplied as deidentified frozen geometries and audit inputs. No confidential CAD is included and no end-to-end real-site regeneration is claimed. All reported optimizer outcomes were retained; the packaging step did not rerun optimization.
