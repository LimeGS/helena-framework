# PHerc0332 qualified strip pack

Five strips exported by stb.strips.export_strip from the PHerc0332 band (fixtures/band_r1145_200_xyz.npz), each independently re-qualified (stb.strips.load_strip + qualify_strip) after the export/load round-trip -- see build_pherc0332_pack.py. The 4 "v2" windows are the ones fixtures/windows_v2.json selected (stratified low/low/median/median over eligible kappa, gate-c pitch-agreement pass); window 12750 is the original v1 anchor window (benchmark_core.C0:C1), which predates gate c and carries no CT pitch/stratum/kappa provenance in this repo's fixtures.

| start | stratum | kappa | coverage | gate a | gate b | requalified a | requalified b | pitch (p2_ct, vox) | file |
|---|---|---|---|---|---|---|---|---|---|
| 10600 | median | 0.1862 | 0.5723 | True | True | True | True | 9.5 | `strip_s10600.npz` |
| 11000 | low | 0.1325 | 0.5448 | True | True | True | True | 10.0 | `strip_s11000.npz` |
| 11300 | low | 0.1446 | 0.5323 | True | True | True | True | 10.0 | `strip_s11300.npz` |
| 12750 | - | - | 0.4783 | True | True | True | True | - | `strip_s12750.npz` |
| 13400 | median | 0.1833 | 0.4811 | True | True | True | True | 9.0 | `strip_s13400.npz` |

"gate a"/"gate b" are stb.gates.coverage_and_gates_ab's own result at export time; "requalified a"/"requalified b" are stb.strips.qualify_strip's independent re-computation from ONLY the exported .npz (no band npz, no original Reference) -- both columns agreeing for all 5 rows is exactly what tests/test_strips.py::test_export_and_qualify_all_5_windows pins.

Pitch is windows_v2.json's gate-c CT-measured spacing (`p2_ct`, vox) for the 4 v2 windows; window 12750 has none recorded (see above).

