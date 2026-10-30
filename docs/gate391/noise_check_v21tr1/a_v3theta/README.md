# #587 (a): v3 θ on gens 47-50 (colour_v2.1+tr1), luminous + spurious (WD off)

Luminous gens 47-50 (colour_v2.1+tr1) weighted at the rung-3 v3 best-fit θ (N_s as v3), refit §12.16 spurious template (v21tr1), WD off. 1-D: DR3 with Poisson errors, mock total with an MC + Poisson band, components, ratio panel. 2-D: DR3 | mock | (k−μ)/√(μ+σ²_MC). χ² = Σ (k−μ)²/(max(μ, 0.5)+σ²_MC) over bins with μ > 0 or k > 0; MC error from a 30-fold bootstrap of the mock draws.

Fit: -lnL 36651.9; 1/1 starts within 5; N_s pull +13.7; at bound ['dgamma_q', 'b1'].


## Ranking (worst first, χ²/dof)

| kind | distribution | split | χ²/dof | χ² ± MC | figure |
|---|---|---|---:|---:|---|
| accelerations | g x f2 | all | 20745.12 | 6700675.1 ± 5303571.2 | [accelerations_2d_g_f2.png](accelerations_2d_g_f2.png) |
| accelerations | g x d | all | 15484.44 | 4924052.5 ± 4575011.9 | [accelerations_2d_g_d.png](accelerations_2d_g_d.png) |
| accelerations | g x colour0 | all | 15106.03 | 4456279.8 ± 7125931.6 | [accelerations_2d_g_colour0.png](accelerations_2d_g_colour0.png) |
| accelerations | g x ruwe | all | 13646.10 | 4366753.2 ± 3363797.5 | [accelerations_2d_g_ruwe.png](accelerations_2d_g_ruwe.png) |
| accelerations | g x signif | all | 10735.65 | 3456878.0 ± 2990158.6 | [accelerations_2d_g_signif.png](accelerations_2d_g_signif.png) |
| accelerations | signif x ruwe | all | 10000.34 | 3240108.8 ± 1444664.5 | [accelerations_2d_signif_ruwe.png](accelerations_2d_signif_ruwe.png) |
| accelerations | signif x f2 | all | 9458.87 | 3064674.5 ± 1438440.4 | [accelerations_2d_signif_f2.png](accelerations_2d_signif_f2.png) |
| accelerations | colour0 x mg0 | all | 9122.49 | 2070806.3 ± 1365871.8 | [accelerations_2d_colour0_mg0.png](accelerations_2d_colour0_mg0.png) |
| accelerations | plxsnr x f2 | all | 8857.49 | 2843254.7 ± 1428843.8 | [accelerations_2d_plxsnr_f2.png](accelerations_2d_plxsnr_f2.png) |
| accelerations | signif | d0.3-0.7 | 7982.65 | 199566.2 ± 118052.9 | [accelerations_1d_signif_d0.3-0.7.png](accelerations_1d_signif_d0.3-0.7.png) |
| accelerations | colour0 x signif | all | 7778.62 | 2512495.5 ± 779656.3 | [accelerations_2d_colour0_signif.png](accelerations_2d_colour0_signif.png) |
| accelerations | g x plxsnr | all | 7356.59 | 2383535.6 ± 8443562.4 | [accelerations_2d_g_plxsnr.png](accelerations_2d_g_plxsnr.png) |
| accelerations | d x mg0 | all | 6892.86 | 1571572.6 ± 4262530.6 | [accelerations_2d_d_mg0.png](accelerations_2d_d_mg0.png) |
| accelerations | plxsnr x ruwe | all | 6492.16 | 2019062.3 ± 934858.1 | [accelerations_2d_plxsnr_ruwe.png](accelerations_2d_plxsnr_ruwe.png) |
| accelerations | plxsnr x signif | all | 6469.99 | 1727487.9 ± 1683848.6 | [accelerations_2d_plxsnr_signif.png](accelerations_2d_plxsnr_signif.png) |
| accelerations | g x mg0 | all | 6359.32 | 1799688.2 ± 4689647.1 | [accelerations_2d_g_mg0.png](accelerations_2d_g_mg0.png) |
| accelerations | d x signif | all | 6268.61 | 1874313.2 ± 1334142.0 | [accelerations_2d_d_signif.png](accelerations_2d_d_signif.png) |
| accelerations | mg0 x f2 | all | 5974.36 | 1935692.5 ± 1625356.0 | [accelerations_2d_mg0_f2.png](accelerations_2d_mg0_f2.png) |
| accelerations | f2 x ruwe | all | 5922.64 | 1913012.7 ± 1044801.0 | [accelerations_2d_f2_ruwe.png](accelerations_2d_f2_ruwe.png) |
| accelerations | colour0 x plxsnr | all | 5459.98 | 1703512.4 ± 3306507.5 | [accelerations_2d_colour0_plxsnr.png](accelerations_2d_colour0_plxsnr.png) |
| accelerations | mg0 x ruwe | all | 5298.09 | 1679493.4 ± 1212990.5 | [accelerations_2d_mg0_ruwe.png](accelerations_2d_mg0_ruwe.png) |
| accelerations | mg0 x plxsnr | all | 4758.58 | 1418056.8 ± 4655971.9 | [accelerations_2d_mg0_plxsnr.png](accelerations_2d_mg0_plxsnr.png) |
| accelerations | d x colour0 | all | 4724.16 | 1351109.4 ± 2599610.8 | [accelerations_2d_d_colour0.png](accelerations_2d_d_colour0.png) |
| accelerations | colour0 x f2 | all | 4521.30 | 1464902.1 ± 1640222.3 | [accelerations_2d_colour0_f2.png](accelerations_2d_colour0_f2.png) |
| accelerations | d x f2 | all | 4423.60 | 1419976.1 ± 2125247.7 | [accelerations_2d_d_f2.png](accelerations_2d_d_f2.png) |
| accelerations | mg0 x signif | all | 4123.93 | 1323781.4 ± 1559256.4 | [accelerations_2d_mg0_signif.png](accelerations_2d_mg0_signif.png) |
| accelerations | f2 | d0.7-1.5 | 3752.73 | 93818.1 ± 674468.7 | [accelerations_1d_f2_d0.7-1.5.png](accelerations_1d_f2_d0.7-1.5.png) |
| accelerations | signif | d0.7-1.5 | 3616.68 | 90416.9 ± 376507.9 | [accelerations_1d_signif_d0.7-1.5.png](accelerations_1d_signif_d0.7-1.5.png) |
| accelerations | g | d0.7-1.5 | 3158.46 | 78961.4 ± 1597058.6 | [accelerations_1d_g_d0.7-1.5.png](accelerations_1d_g_d0.7-1.5.png) |
| accelerations | colour0 x ruwe | all | 3063.97 | 977406.4 ± 872004.8 | [accelerations_2d_colour0_ruwe.png](accelerations_2d_colour0_ruwe.png) |
| accelerations | f2 | d1.5-6 | 2315.20 | 55564.9 ± 20974.6 | [accelerations_1d_f2_d1.5-6.png](accelerations_1d_f2_d1.5-6.png) |
| accelerations | d x ruwe | all | 1424.74 | 410324.5 ± 806656.5 | [accelerations_2d_d_ruwe.png](accelerations_2d_d_ruwe.png) |
| accelerations | ruwe | d0.3-0.7 | 1299.73 | 32493.2 ± 40782.3 | [accelerations_1d_ruwe_d0.3-0.7.png](accelerations_1d_ruwe_d0.3-0.7.png) |
| accelerations | plxsnr | d0-0.3 | 901.70 | 22542.5 ± 70451.5 | [accelerations_1d_plxsnr_d0-0.3.png](accelerations_1d_plxsnr_d0-0.3.png) |
| accelerations | d x plxsnr | all | 865.12 | 199843.4 ± 1128178.8 | [accelerations_2d_d_plxsnr.png](accelerations_2d_d_plxsnr.png) |
| accelerations | ruwe | d0.7-1.5 | 820.42 | 18049.3 ± 5721.3 | [accelerations_1d_ruwe_d0.7-1.5.png](accelerations_1d_ruwe_d0.7-1.5.png) |
| accelerations | colour0 | d0.7-1.5 | 591.67 | 13608.3 ± 21314.9 | [accelerations_1d_colour0_d0.7-1.5.png](accelerations_1d_colour0_d0.7-1.5.png) |
| accelerations | signif | d1.5-6 | 574.16 | 8038.2 ± 7462.1 | [accelerations_1d_signif_d1.5-6.png](accelerations_1d_signif_d1.5-6.png) |
| accelerations | d | d0-0.3 | 504.23 | 5546.5 ± 6311.2 | [accelerations_1d_d_d0-0.3.png](accelerations_1d_d_d0-0.3.png) |
| accelerations | mg0 | d0.7-1.5 | 450.43 | 7657.3 ± 546872.7 | [accelerations_1d_mg0_d0.7-1.5.png](accelerations_1d_mg0_d0.7-1.5.png) |
| accelerations | f2 | d0-0.3 | 433.79 | 10844.7 ± 24735.6 | [accelerations_1d_f2_d0-0.3.png](accelerations_1d_f2_d0-0.3.png) |
| accelerations | mg0 | all | 307.21 | 7680.3 ± 1529.9 | [accelerations_1d_mg0.png](accelerations_1d_mg0.png) |
| accelerations | d | all | 293.78 | 7344.4 ± 7784.9 | [accelerations_1d_d.png](accelerations_1d_d.png) |
| accelerations | signif | all | 270.86 | 6771.5 ± 4556.2 | [accelerations_1d_signif.png](accelerations_1d_signif.png) |
| accelerations | signif | d0-0.3 | 262.18 | 6554.5 ± 2910.6 | [accelerations_1d_signif_d0-0.3.png](accelerations_1d_signif_d0-0.3.png) |
| accelerations | mg0 | d0.3-0.7 | 243.23 | 5594.4 ± 8275.9 | [accelerations_1d_mg0_d0.3-0.7.png](accelerations_1d_mg0_d0.3-0.7.png) |
| accelerations | g | d0-0.3 | 228.66 | 5716.5 ± 20250.7 | [accelerations_1d_g_d0-0.3.png](accelerations_1d_g_d0-0.3.png) |
| accelerations | ruwe | d0-0.3 | 223.65 | 5591.2 ± 4387.1 | [accelerations_1d_ruwe_d0-0.3.png](accelerations_1d_ruwe_d0-0.3.png) |
| accelerations | d | d0.3-0.7 | 221.30 | 1770.4 ± 437.1 | [accelerations_1d_d_d0.3-0.7.png](accelerations_1d_d_d0.3-0.7.png) |
| accelerations | colour0 | d0.3-0.7 | 202.37 | 5059.2 ± 47278.7 | [accelerations_1d_colour0_d0.3-0.7.png](accelerations_1d_colour0_d0.3-0.7.png) |
| accelerations | colour0 | all | 194.51 | 4862.8 ± 2243.4 | [accelerations_1d_colour0.png](accelerations_1d_colour0.png) |
| accelerations | mg0 | d0-0.3 | 172.06 | 3957.4 ± 2936.1 | [accelerations_1d_mg0_d0-0.3.png](accelerations_1d_mg0_d0-0.3.png) |
| accelerations | g | d0.3-0.7 | 161.10 | 4027.4 ± 5883.3 | [accelerations_1d_g_d0.3-0.7.png](accelerations_1d_g_d0.3-0.7.png) |
| accelerations | g | all | 150.29 | 3757.2 ± 1046.9 | [accelerations_1d_g.png](accelerations_1d_g.png) |
| accelerations | colour0 | d0-0.3 | 150.05 | 3751.2 ± 5122.2 | [accelerations_1d_colour0_d0-0.3.png](accelerations_1d_colour0_d0-0.3.png) |
| accelerations | f2 | all | 137.76 | 3444.0 ± 6659.6 | [accelerations_1d_f2.png](accelerations_1d_f2.png) |
| orbits | A | d0-0.3 | 135.61 | 3390.3 ± 326.8 | [orbits_1d_A_d0-0.3.png](orbits_1d_A_d0-0.3.png) |
| accelerations | plxsnr | all | 98.53 | 2463.3 ± 892.5 | [accelerations_1d_plxsnr.png](accelerations_1d_plxsnr.png) |
| accelerations | ruwe | all | 95.20 | 2379.9 ± 16405.0 | [accelerations_1d_ruwe.png](accelerations_1d_ruwe.png) |
| accelerations | g | d1.5-6 | 94.41 | 1888.3 ± 99527.2 | [accelerations_1d_g_d1.5-6.png](accelerations_1d_g_d1.5-6.png) |
| accelerations | ruwe | d1.5-6 | 92.09 | 828.9 ± 7903.2 | [accelerations_1d_ruwe_d1.5-6.png](accelerations_1d_ruwe_d1.5-6.png) |
| accelerations | f2 | d0.3-0.7 | 87.83 | 2195.6 ± 44928.3 | [accelerations_1d_f2_d0.3-0.7.png](accelerations_1d_f2_d0.3-0.7.png) |
| orbits | g x a0snr | all | 73.82 | 23622.1 ± 3736.5 | [orbits_2d_g_a0snr.png](orbits_2d_g_a0snr.png) |
| accelerations | plxsnr | d0.3-0.7 | 72.13 | 1586.9 ± 232283.3 | [accelerations_1d_plxsnr_d0.3-0.7.png](accelerations_1d_plxsnr_d0.3-0.7.png) |
| orbits | A x a0snr | all | 59.94 | 19120.1 ± 2158.3 | [orbits_2d_A_a0snr.png](orbits_2d_A_a0snr.png) |
| accelerations | colour0 | d1.5-6 | 56.74 | 1191.5 ± 125262.0 | [accelerations_1d_colour0_d1.5-6.png](accelerations_1d_colour0_d1.5-6.png) |
| orbits | a0snr x ruwe | all | 54.37 | 16093.0 ± 1557.5 | [orbits_2d_a0snr_ruwe.png](orbits_2d_a0snr_ruwe.png) |
| orbits | d x a0snr | all | 53.96 | 16565.3 ± 2246.2 | [orbits_2d_d_a0snr.png](orbits_2d_d_a0snr.png) |
| orbits | a0snr x plxsnr | all | 53.74 | 16981.3 ± 2465.6 | [orbits_2d_a0snr_plxsnr.png](orbits_2d_a0snr_plxsnr.png) |
| orbits | colour0 x a0snr | all | 53.03 | 16915.8 ± 1523.5 | [orbits_2d_colour0_a0snr.png](orbits_2d_colour0_a0snr.png) |
| orbits | mg0 x A | all | 50.05 | 15615.9 ± 1135.2 | [orbits_2d_mg0_A.png](orbits_2d_mg0_A.png) |
| orbits | mg0 x a0snr | all | 49.25 | 15710.4 ± 2668.2 | [orbits_2d_mg0_a0snr.png](orbits_2d_mg0_a0snr.png) |
| orbits | A | d0.3-0.7 | 48.62 | 1215.6 ± 803.9 | [orbits_1d_A_d0.3-0.7.png](orbits_1d_A_d0.3-0.7.png) |
| orbits | f2 | d0-0.3 | 47.91 | 1197.8 ± 204.7 | [orbits_1d_f2_d0-0.3.png](orbits_1d_f2_d0-0.3.png) |
| orbits | e x a0snr | all | 47.27 | 15267.0 ± 3053.1 | [orbits_2d_e_a0snr.png](orbits_2d_e_a0snr.png) |
| orbits | P x a0snr | all | 45.79 | 14605.5 ± 1648.6 | [orbits_2d_P_a0snr.png](orbits_2d_P_a0snr.png) |
| orbits | d x A | all | 45.63 | 13689.1 ± 850.3 | [orbits_2d_d_A.png](orbits_2d_d_A.png) |
| orbits | P | d0-0.3 | 45.43 | 1135.6 ± 391.7 | [orbits_1d_P_d0-0.3.png](orbits_1d_P_d0-0.3.png) |
| orbits | A x ruwe | all | 44.60 | 14315.1 ± 1015.9 | [orbits_2d_A_ruwe.png](orbits_2d_A_ruwe.png) |
| orbits | a0snr | d0.3-0.7 | 43.47 | 1086.8 ± 1622.4 | [orbits_1d_a0snr_d0.3-0.7.png](orbits_1d_a0snr_d0.3-0.7.png) |
| orbits | e | d0-0.3 | 41.20 | 1030.0 ± 317.1 | [orbits_1d_e_d0-0.3.png](orbits_1d_e_d0-0.3.png) |
| orbits | a0snr x f2 | all | 40.31 | 13059.6 ± 1897.7 | [orbits_2d_a0snr_f2.png](orbits_2d_a0snr_f2.png) |
| orbits | g x ruwe | all | 39.84 | 12669.9 ± 1875.1 | [orbits_2d_g_ruwe.png](orbits_2d_g_ruwe.png) |
| orbits | A x plxsnr | all | 39.15 | 12330.9 ± 740.9 | [orbits_2d_A_plxsnr.png](orbits_2d_A_plxsnr.png) |
| orbits | colour0 x ruwe | all | 38.80 | 12182.6 ± 861.0 | [orbits_2d_colour0_ruwe.png](orbits_2d_colour0_ruwe.png) |
| orbits | g x d | all | 37.55 | 12017.0 ± 1753.3 | [orbits_2d_g_d.png](orbits_2d_g_d.png) |
| orbits | plxsnr x ruwe | all | 33.61 | 10823.8 ± 1040.3 | [orbits_2d_plxsnr_ruwe.png](orbits_2d_plxsnr_ruwe.png) |
| orbits | d x ruwe | all | 32.76 | 9270.1 ± 1188.6 | [orbits_2d_d_ruwe.png](orbits_2d_d_ruwe.png) |
| orbits | e x ruwe | all | 30.40 | 9848.1 ± 1859.5 | [orbits_2d_e_ruwe.png](orbits_2d_e_ruwe.png) |
| orbits | mg0 x ruwe | all | 30.19 | 9330.1 ± 1610.3 | [orbits_2d_mg0_ruwe.png](orbits_2d_mg0_ruwe.png) |
| orbits | g x A | all | 29.82 | 9662.8 ± 1253.6 | [orbits_2d_g_A.png](orbits_2d_g_A.png) |
| orbits | f2 x ruwe | all | 29.62 | 9539.1 ± 1197.5 | [orbits_2d_f2_ruwe.png](orbits_2d_f2_ruwe.png) |
| orbits | colour0 x A | all | 28.94 | 9146.4 ± 763.3 | [orbits_2d_colour0_A.png](orbits_2d_colour0_A.png) |
| accelerations | plxsnr | d0.7-1.5 | 28.76 | 402.6 ± 8829.3 | [accelerations_1d_plxsnr_d0.7-1.5.png](accelerations_1d_plxsnr_d0.7-1.5.png) |
| orbits | g x mg0 | all | 28.67 | 8257.9 ± 1229.1 | [orbits_2d_g_mg0.png](orbits_2d_g_mg0.png) |
| orbits | d x mg0 | all | 28.34 | 6234.8 ± 994.4 | [orbits_2d_d_mg0.png](orbits_2d_d_mg0.png) |
| orbits | g x plxsnr | all | 27.89 | 9037.7 ± 1288.0 | [orbits_2d_g_plxsnr.png](orbits_2d_g_plxsnr.png) |
| orbits | g x e | all | 27.76 | 8995.4 ± 1562.4 | [orbits_2d_g_e.png](orbits_2d_g_e.png) |
| orbits | a0snr | d0-0.3 | 27.66 | 691.4 ± 610.0 | [orbits_1d_a0snr_d0-0.3.png](orbits_1d_a0snr_d0-0.3.png) |
| orbits | d x colour0 | all | 27.10 | 7397.1 ± 999.4 | [orbits_2d_d_colour0.png](orbits_2d_d_colour0.png) |
| orbits | P x ruwe | all | 26.43 | 8432.3 ± 865.9 | [orbits_2d_P_ruwe.png](orbits_2d_P_ruwe.png) |
| orbits | d | d0-0.3 | 25.99 | 311.8 ± 327.4 | [orbits_1d_d_d0-0.3.png](orbits_1d_d_d0-0.3.png) |
| orbits | colour0 x plxsnr | all | 25.67 | 7983.5 ± 984.4 | [orbits_2d_colour0_plxsnr.png](orbits_2d_colour0_plxsnr.png) |
| orbits | d x P | all | 25.43 | 7119.1 ± 663.0 | [orbits_2d_d_P.png](orbits_2d_d_P.png) |
| orbits | e x plxsnr | all | 23.65 | 7664.0 ± 1159.8 | [orbits_2d_e_plxsnr.png](orbits_2d_e_plxsnr.png) |
| accelerations | mg0 | d1.5-6 | 23.53 | 235.3 ± 42479.9 | [accelerations_1d_mg0_d1.5-6.png](accelerations_1d_mg0_d1.5-6.png) |
| orbits | P x plxsnr | all | 23.51 | 5783.9 ± 935.4 | [orbits_2d_P_plxsnr.png](orbits_2d_P_plxsnr.png) |
| orbits | ruwe | d0-0.3 | 23.38 | 584.6 ± 330.7 | [orbits_1d_ruwe_d0-0.3.png](orbits_1d_ruwe_d0-0.3.png) |
| orbits | g x colour0 | all | 22.95 | 6815.3 ± 1072.9 | [orbits_2d_g_colour0.png](orbits_2d_g_colour0.png) |
| orbits | a0snr | all | 22.87 | 571.6 ± 409.4 | [orbits_1d_a0snr.png](orbits_1d_a0snr.png) |
| orbits | colour0 x P | all | 22.83 | 7282.0 ± 569.9 | [orbits_2d_colour0_P.png](orbits_2d_colour0_P.png) |
| orbits | plxsnr | d0-0.3 | 22.70 | 567.5 ± 271.9 | [orbits_1d_plxsnr_d0-0.3.png](orbits_1d_plxsnr_d0-0.3.png) |
| orbits | P x e | all | 22.63 | 7331.3 ± 859.2 | [orbits_2d_P_e.png](orbits_2d_P_e.png) |
| orbits | mg0 x P | all | 21.89 | 6720.0 ± 1013.9 | [orbits_2d_mg0_P.png](orbits_2d_mg0_P.png) |
| orbits | mg0 x plxsnr | all | 21.79 | 6342.1 ± 903.3 | [orbits_2d_mg0_plxsnr.png](orbits_2d_mg0_plxsnr.png) |
| orbits | g x P | all | 21.79 | 6907.5 ± 1236.7 | [orbits_2d_g_P.png](orbits_2d_g_P.png) |
| orbits | mg0 | d0.7-1.5 | 20.90 | 355.2 ± 223.4 | [orbits_1d_mg0_d0.7-1.5.png](orbits_1d_mg0_d0.7-1.5.png) |
| orbits | mg0 x e | all | 20.64 | 6687.4 ± 1636.1 | [orbits_2d_mg0_e.png](orbits_2d_mg0_e.png) |
| orbits | e x A | all | 20.53 | 6653.3 ± 1348.8 | [orbits_2d_e_A.png](orbits_2d_e_A.png) |
| orbits | plxsnr x f2 | all | 20.48 | 6634.0 ± 937.9 | [orbits_2d_plxsnr_f2.png](orbits_2d_plxsnr_f2.png) |
| orbits | g x f2 | all | 20.47 | 6385.6 ± 1165.5 | [orbits_2d_g_f2.png](orbits_2d_g_f2.png) |
| orbits | d x f2 | all | 20.19 | 6502.7 ± 649.2 | [orbits_2d_d_f2.png](orbits_2d_d_f2.png) |
| orbits | P x A | all | 19.97 | 6389.8 ± 780.9 | [orbits_2d_P_A.png](orbits_2d_P_A.png) |
| orbits | d | d0.3-0.7 | 19.82 | 138.7 ± 69.6 | [orbits_1d_d_d0.3-0.7.png](orbits_1d_d_d0.3-0.7.png) |
| orbits | mg0 x f2 | all | 19.44 | 6240.2 ± 1100.8 | [orbits_2d_mg0_f2.png](orbits_2d_mg0_f2.png) |
| orbits | d x e | all | 19.03 | 6165.5 ± 1082.1 | [orbits_2d_d_e.png](orbits_2d_d_e.png) |
| orbits | colour0 | d0-0.3 | 18.71 | 467.7 ± 194.2 | [orbits_1d_colour0_d0-0.3.png](orbits_1d_colour0_d0-0.3.png) |
| orbits | colour0 x f2 | all | 18.66 | 5934.0 ± 723.0 | [orbits_2d_colour0_f2.png](orbits_2d_colour0_f2.png) |
| accelerations | d | d0.7-1.5 | 17.91 | 125.4 ± 188.0 | [accelerations_1d_d_d0.7-1.5.png](accelerations_1d_d_d0.7-1.5.png) |
| orbits | d | all | 17.69 | 442.2 ± 179.4 | [orbits_1d_d.png](orbits_1d_d.png) |
| orbits | g | d0-0.3 | 17.43 | 435.7 ± 276.7 | [orbits_1d_g_d0-0.3.png](orbits_1d_g_d0-0.3.png) |
| orbits | ruwe | d0.7-1.5 | 17.36 | 347.2 ± 178.1 | [orbits_1d_ruwe_d0.7-1.5.png](orbits_1d_ruwe_d0.7-1.5.png) |
| orbits | colour0 x mg0 | all | 17.33 | 3587.1 ± 599.8 | [orbits_2d_colour0_mg0.png](orbits_2d_colour0_mg0.png) |
| orbits | colour0 x e | all | 17.32 | 5612.9 ± 887.6 | [orbits_2d_colour0_e.png](orbits_2d_colour0_e.png) |
| orbits | P x f2 | all | 17.20 | 5572.2 ± 495.3 | [orbits_2d_P_f2.png](orbits_2d_P_f2.png) |
| orbits | P | d0.7-1.5 | 17.17 | 343.4 ± 188.1 | [orbits_1d_P_d0.7-1.5.png](orbits_1d_P_d0.7-1.5.png) |
| orbits | plxsnr | all | 16.97 | 424.3 ± 222.6 | [orbits_1d_plxsnr.png](orbits_1d_plxsnr.png) |
| orbits | d x plxsnr | all | 15.82 | 3686.9 ± 684.7 | [orbits_2d_d_plxsnr.png](orbits_2d_d_plxsnr.png) |
| orbits | e x f2 | all | 15.36 | 4975.3 ± 1170.4 | [orbits_2d_e_f2.png](orbits_2d_e_f2.png) |
| orbits | A x f2 | all | 14.95 | 4828.2 ± 781.6 | [orbits_2d_A_f2.png](orbits_2d_A_f2.png) |
| orbits | g | d0.7-1.5 | 13.91 | 347.7 ± 374.2 | [orbits_1d_g_d0.7-1.5.png](orbits_1d_g_d0.7-1.5.png) |
| orbits | ruwe | d0.3-0.7 | 13.79 | 344.9 ± 357.3 | [orbits_1d_ruwe_d0.3-0.7.png](orbits_1d_ruwe_d0.3-0.7.png) |
| orbits | mg0 | d0-0.3 | 13.55 | 298.1 ± 174.6 | [orbits_1d_mg0_d0-0.3.png](orbits_1d_mg0_d0-0.3.png) |
| orbits | mg0 | all | 13.41 | 335.3 ± 125.5 | [orbits_1d_mg0.png](orbits_1d_mg0.png) |
| orbits | a0snr | d0.7-1.5 | 13.04 | 300.0 ± 758.0 | [orbits_1d_a0snr_d0.7-1.5.png](orbits_1d_a0snr_d0.7-1.5.png) |
| orbits | g | d0.3-0.7 | 12.68 | 317.0 ± 602.8 | [orbits_1d_g_d0.3-0.7.png](orbits_1d_g_d0.3-0.7.png) |
| orbits | P | all | 12.35 | 308.8 ± 242.7 | [orbits_1d_P.png](orbits_1d_P.png) |
| orbits | f2 | d0.3-0.7 | 11.81 | 295.3 ± 117.1 | [orbits_1d_f2_d0.3-0.7.png](orbits_1d_f2_d0.3-0.7.png) |
| orbits | g | all | 10.41 | 260.2 ± 151.8 | [orbits_1d_g.png](orbits_1d_g.png) |
| orbits | mg0 | d1.5-6 | 9.57 | 95.7 ± 66.2 | [orbits_1d_mg0_d1.5-6.png](orbits_1d_mg0_d1.5-6.png) |
| orbits | colour0 | all | 9.07 | 226.9 ± 135.7 | [orbits_1d_colour0.png](orbits_1d_colour0.png) |
| orbits | ruwe | all | 8.63 | 215.7 ± 188.6 | [orbits_1d_ruwe.png](orbits_1d_ruwe.png) |
| orbits | e | d0.3-0.7 | 8.23 | 205.8 ± 194.1 | [orbits_1d_e_d0.3-0.7.png](orbits_1d_e_d0.3-0.7.png) |
| orbits | colour0 | d0.3-0.7 | 7.59 | 182.1 ± 153.3 | [orbits_1d_colour0_d0.3-0.7.png](orbits_1d_colour0_d0.3-0.7.png) |
| orbits | g | d1.5-6 | 7.41 | 155.7 ± 57.8 | [orbits_1d_g_d1.5-6.png](orbits_1d_g_d1.5-6.png) |
| orbits | mg0 | d0.3-0.7 | 7.21 | 165.8 ± 95.8 | [orbits_1d_mg0_d0.3-0.7.png](orbits_1d_mg0_d0.3-0.7.png) |
| orbits | e | d1.5-6 | 6.77 | 169.3 ± 77.4 | [orbits_1d_e_d1.5-6.png](orbits_1d_e_d1.5-6.png) |
| orbits | A | all | 6.49 | 162.3 ± 148.5 | [orbits_1d_A.png](orbits_1d_A.png) |
| orbits | P | d0.3-0.7 | 6.22 | 155.5 ± 145.6 | [orbits_1d_P_d0.3-0.7.png](orbits_1d_P_d0.3-0.7.png) |
| orbits | a0snr | d1.5-6 | 5.08 | 91.4 ± 57.5 | [orbits_1d_a0snr_d1.5-6.png](orbits_1d_a0snr_d1.5-6.png) |
| orbits | A | d1.5-6 | 4.95 | 69.3 ± 18.9 | [orbits_1d_A_d1.5-6.png](orbits_1d_A_d1.5-6.png) |
| orbits | f2 | all | 4.90 | 122.5 ± 207.2 | [orbits_1d_f2.png](orbits_1d_f2.png) |
| orbits | plxsnr | d0.3-0.7 | 4.75 | 90.3 ± 135.0 | [orbits_1d_plxsnr_d0.3-0.7.png](orbits_1d_plxsnr_d0.3-0.7.png) |
| orbits | f2 | d1.5-6 | 4.72 | 99.2 ± 77.1 | [orbits_1d_f2_d1.5-6.png](orbits_1d_f2_d1.5-6.png) |
| orbits | e | all | 4.19 | 104.7 ± 103.8 | [orbits_1d_e.png](orbits_1d_e.png) |
| orbits | colour0 | d0.7-1.5 | 4.08 | 93.9 ± 53.8 | [orbits_1d_colour0_d0.7-1.5.png](orbits_1d_colour0_d0.7-1.5.png) |
| orbits | colour0 | d1.5-6 | 3.04 | 60.9 ± 64.7 | [orbits_1d_colour0_d1.5-6.png](orbits_1d_colour0_d1.5-6.png) |
| orbits | ruwe | d1.5-6 | 2.73 | 32.7 ± 34.3 | [orbits_1d_ruwe_d1.5-6.png](orbits_1d_ruwe_d1.5-6.png) |
| orbits | f2 | d0.7-1.5 | 2.55 | 63.8 ± 106.2 | [orbits_1d_f2_d0.7-1.5.png](orbits_1d_f2_d0.7-1.5.png) |
| orbits | e | d0.7-1.5 | 2.45 | 61.3 ± 221.5 | [orbits_1d_e_d0.7-1.5.png](orbits_1d_e_d0.7-1.5.png) |
| orbits | A | d0.7-1.5 | 2.44 | 56.2 ± 82.6 | [orbits_1d_A_d0.7-1.5.png](orbits_1d_A_d0.7-1.5.png) |
| orbits | d | d1.5-6 | 2.14 | 4.3 ± 2.7 | [orbits_1d_d_d1.5-6.png](orbits_1d_d_d1.5-6.png) |
| orbits | plxsnr | d0.7-1.5 | 1.97 | 25.6 ± 64.5 | [orbits_1d_plxsnr_d0.7-1.5.png](orbits_1d_plxsnr_d0.7-1.5.png) |
| orbits | plxsnr | d1.5-6 | 1.93 | 15.4 ± 17.3 | [orbits_1d_plxsnr_d1.5-6.png](orbits_1d_plxsnr_d1.5-6.png) |
| orbits | d | d0.7-1.5 | 1.44 | 10.1 ± 98.4 | [orbits_1d_d_d0.7-1.5.png](orbits_1d_d_d0.7-1.5.png) |
| orbits | P | d1.5-6 | 1.07 | 9.7 ± 23.8 | [orbits_1d_P_d1.5-6.png](orbits_1d_P_d1.5-6.png) |
| accelerations | d | d1.5-6 | 0.99 | 2.0 ± 19.9 | [accelerations_1d_d_d1.5-6.png](accelerations_1d_d_d1.5-6.png) |
| accelerations | plxsnr | d1.5-6 | 0.76 | 5.3 ± 589.9 | [accelerations_1d_plxsnr_d1.5-6.png](accelerations_1d_plxsnr_d1.5-6.png) |

## All figures

- orbits 1-D: [$G$ (mag)](orbits_1d_g.png)
- orbits 1-D, d 0–0.3 kpc: [$G$ (mag)](orbits_1d_g_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [$G$ (mag)](orbits_1d_g_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [$G$ (mag)](orbits_1d_g_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [$G$ (mag)](orbits_1d_g_d1.5-6.png)
- orbits 1-D: [distance (kpc)](orbits_1d_d.png)
- orbits 1-D, d 0–0.3 kpc: [distance (kpc)](orbits_1d_d_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [distance (kpc)](orbits_1d_d_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [distance (kpc)](orbits_1d_d_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [distance (kpc)](orbits_1d_d_d1.5-6.png)
- orbits 1-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag)](orbits_1d_colour0.png)
- orbits 1-D, d 0–0.3 kpc: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag)](orbits_1d_colour0_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag)](orbits_1d_colour0_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag)](orbits_1d_colour0_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag)](orbits_1d_colour0_d1.5-6.png)
- orbits 1-D: [$M_{G,0}$ (mag)](orbits_1d_mg0.png)
- orbits 1-D, d 0–0.3 kpc: [$M_{G,0}$ (mag)](orbits_1d_mg0_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [$M_{G,0}$ (mag)](orbits_1d_mg0_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [$M_{G,0}$ (mag)](orbits_1d_mg0_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [$M_{G,0}$ (mag)](orbits_1d_mg0_d1.5-6.png)
- orbits 1-D: [$P$ (d)](orbits_1d_P.png)
- orbits 1-D, d 0–0.3 kpc: [$P$ (d)](orbits_1d_P_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [$P$ (d)](orbits_1d_P_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [$P$ (d)](orbits_1d_P_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [$P$ (d)](orbits_1d_P_d1.5-6.png)
- orbits 1-D: [$e$](orbits_1d_e.png)
- orbits 1-D, d 0–0.3 kpc: [$e$](orbits_1d_e_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [$e$](orbits_1d_e_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [$e$](orbits_1d_e_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [$e$](orbits_1d_e_d1.5-6.png)
- orbits 1-D: [$A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_1d_A.png)
- orbits 1-D, d 0–0.3 kpc: [$A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_1d_A_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [$A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_1d_A_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [$A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_1d_A_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [$A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_1d_A_d1.5-6.png)
- orbits 1-D: [$a_0/\sigma_{a_0}$](orbits_1d_a0snr.png)
- orbits 1-D, d 0–0.3 kpc: [$a_0/\sigma_{a_0}$](orbits_1d_a0snr_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [$a_0/\sigma_{a_0}$](orbits_1d_a0snr_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [$a_0/\sigma_{a_0}$](orbits_1d_a0snr_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [$a_0/\sigma_{a_0}$](orbits_1d_a0snr_d1.5-6.png)
- orbits 1-D: [$\varpi/\sigma_\varpi$](orbits_1d_plxsnr.png)
- orbits 1-D, d 0–0.3 kpc: [$\varpi/\sigma_\varpi$](orbits_1d_plxsnr_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [$\varpi/\sigma_\varpi$](orbits_1d_plxsnr_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [$\varpi/\sigma_\varpi$](orbits_1d_plxsnr_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [$\varpi/\sigma_\varpi$](orbits_1d_plxsnr_d1.5-6.png)
- orbits 1-D: [$F_2$](orbits_1d_f2.png)
- orbits 1-D, d 0–0.3 kpc: [$F_2$](orbits_1d_f2_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [$F_2$](orbits_1d_f2_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [$F_2$](orbits_1d_f2_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [$F_2$](orbits_1d_f2_d1.5-6.png)
- orbits 1-D: [RUWE](orbits_1d_ruwe.png)
- orbits 1-D, d 0–0.3 kpc: [RUWE](orbits_1d_ruwe_d0-0.3.png)
- orbits 1-D, d 0.3–0.7 kpc: [RUWE](orbits_1d_ruwe_d0.3-0.7.png)
- orbits 1-D, d 0.7–1.5 kpc: [RUWE](orbits_1d_ruwe_d0.7-1.5.png)
- orbits 1-D, d 1.5–6 kpc: [RUWE](orbits_1d_ruwe_d1.5-6.png)
- orbits 2-D: [$G$ (mag) vs distance (kpc)](orbits_2d_g_d.png)
- orbits 2-D: [$G$ (mag) vs $(G_{\rm BP}-G_{\rm RP})_0$ (mag)](orbits_2d_g_colour0.png)
- orbits 2-D: [$G$ (mag) vs $M_{G,0}$ (mag)](orbits_2d_g_mg0.png)
- orbits 2-D: [$G$ (mag) vs $P$ (d)](orbits_2d_g_P.png)
- orbits 2-D: [$G$ (mag) vs $e$](orbits_2d_g_e.png)
- orbits 2-D: [$G$ (mag) vs $A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_2d_g_A.png)
- orbits 2-D: [$G$ (mag) vs $a_0/\sigma_{a_0}$](orbits_2d_g_a0snr.png)
- orbits 2-D: [$G$ (mag) vs $\varpi/\sigma_\varpi$](orbits_2d_g_plxsnr.png)
- orbits 2-D: [$G$ (mag) vs $F_2$](orbits_2d_g_f2.png)
- orbits 2-D: [$G$ (mag) vs RUWE](orbits_2d_g_ruwe.png)
- orbits 2-D: [distance (kpc) vs $(G_{\rm BP}-G_{\rm RP})_0$ (mag)](orbits_2d_d_colour0.png)
- orbits 2-D: [distance (kpc) vs $M_{G,0}$ (mag)](orbits_2d_d_mg0.png)
- orbits 2-D: [distance (kpc) vs $P$ (d)](orbits_2d_d_P.png)
- orbits 2-D: [distance (kpc) vs $e$](orbits_2d_d_e.png)
- orbits 2-D: [distance (kpc) vs $A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_2d_d_A.png)
- orbits 2-D: [distance (kpc) vs $a_0/\sigma_{a_0}$](orbits_2d_d_a0snr.png)
- orbits 2-D: [distance (kpc) vs $\varpi/\sigma_\varpi$](orbits_2d_d_plxsnr.png)
- orbits 2-D: [distance (kpc) vs $F_2$](orbits_2d_d_f2.png)
- orbits 2-D: [distance (kpc) vs RUWE](orbits_2d_d_ruwe.png)
- orbits 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs $M_{G,0}$ (mag)](orbits_2d_colour0_mg0.png)
- orbits 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs $P$ (d)](orbits_2d_colour0_P.png)
- orbits 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs $e$](orbits_2d_colour0_e.png)
- orbits 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs $A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_2d_colour0_A.png)
- orbits 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs $a_0/\sigma_{a_0}$](orbits_2d_colour0_a0snr.png)
- orbits 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs $\varpi/\sigma_\varpi$](orbits_2d_colour0_plxsnr.png)
- orbits 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs $F_2$](orbits_2d_colour0_f2.png)
- orbits 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs RUWE](orbits_2d_colour0_ruwe.png)
- orbits 2-D: [$M_{G,0}$ (mag) vs $P$ (d)](orbits_2d_mg0_P.png)
- orbits 2-D: [$M_{G,0}$ (mag) vs $e$](orbits_2d_mg0_e.png)
- orbits 2-D: [$M_{G,0}$ (mag) vs $A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_2d_mg0_A.png)
- orbits 2-D: [$M_{G,0}$ (mag) vs $a_0/\sigma_{a_0}$](orbits_2d_mg0_a0snr.png)
- orbits 2-D: [$M_{G,0}$ (mag) vs $\varpi/\sigma_\varpi$](orbits_2d_mg0_plxsnr.png)
- orbits 2-D: [$M_{G,0}$ (mag) vs $F_2$](orbits_2d_mg0_f2.png)
- orbits 2-D: [$M_{G,0}$ (mag) vs RUWE](orbits_2d_mg0_ruwe.png)
- orbits 2-D: [$P$ (d) vs $e$](orbits_2d_P_e.png)
- orbits 2-D: [$P$ (d) vs $A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_2d_P_A.png)
- orbits 2-D: [$P$ (d) vs $a_0/\sigma_{a_0}$](orbits_2d_P_a0snr.png)
- orbits 2-D: [$P$ (d) vs $\varpi/\sigma_\varpi$](orbits_2d_P_plxsnr.png)
- orbits 2-D: [$P$ (d) vs $F_2$](orbits_2d_P_f2.png)
- orbits 2-D: [$P$ (d) vs RUWE](orbits_2d_P_ruwe.png)
- orbits 2-D: [$e$ vs $A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$)](orbits_2d_e_A.png)
- orbits 2-D: [$e$ vs $a_0/\sigma_{a_0}$](orbits_2d_e_a0snr.png)
- orbits 2-D: [$e$ vs $\varpi/\sigma_\varpi$](orbits_2d_e_plxsnr.png)
- orbits 2-D: [$e$ vs $F_2$](orbits_2d_e_f2.png)
- orbits 2-D: [$e$ vs RUWE](orbits_2d_e_ruwe.png)
- orbits 2-D: [$A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$) vs $a_0/\sigma_{a_0}$](orbits_2d_A_a0snr.png)
- orbits 2-D: [$A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$) vs $\varpi/\sigma_\varpi$](orbits_2d_A_plxsnr.png)
- orbits 2-D: [$A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$) vs $F_2$](orbits_2d_A_f2.png)
- orbits 2-D: [$A=(a_0/\varpi)/P^{2/3}$ (AU yr$^{-2/3}$) vs RUWE](orbits_2d_A_ruwe.png)
- orbits 2-D: [$a_0/\sigma_{a_0}$ vs $\varpi/\sigma_\varpi$](orbits_2d_a0snr_plxsnr.png)
- orbits 2-D: [$a_0/\sigma_{a_0}$ vs $F_2$](orbits_2d_a0snr_f2.png)
- orbits 2-D: [$a_0/\sigma_{a_0}$ vs RUWE](orbits_2d_a0snr_ruwe.png)
- orbits 2-D: [$\varpi/\sigma_\varpi$ vs $F_2$](orbits_2d_plxsnr_f2.png)
- orbits 2-D: [$\varpi/\sigma_\varpi$ vs RUWE](orbits_2d_plxsnr_ruwe.png)
- orbits 2-D: [$F_2$ vs RUWE](orbits_2d_f2_ruwe.png)
- accelerations 1-D: [$G$ (mag)](accelerations_1d_g.png)
- accelerations 1-D, d 0–0.3 kpc: [$G$ (mag)](accelerations_1d_g_d0-0.3.png)
- accelerations 1-D, d 0.3–0.7 kpc: [$G$ (mag)](accelerations_1d_g_d0.3-0.7.png)
- accelerations 1-D, d 0.7–1.5 kpc: [$G$ (mag)](accelerations_1d_g_d0.7-1.5.png)
- accelerations 1-D, d 1.5–6 kpc: [$G$ (mag)](accelerations_1d_g_d1.5-6.png)
- accelerations 1-D: [distance (kpc)](accelerations_1d_d.png)
- accelerations 1-D, d 0–0.3 kpc: [distance (kpc)](accelerations_1d_d_d0-0.3.png)
- accelerations 1-D, d 0.3–0.7 kpc: [distance (kpc)](accelerations_1d_d_d0.3-0.7.png)
- accelerations 1-D, d 0.7–1.5 kpc: [distance (kpc)](accelerations_1d_d_d0.7-1.5.png)
- accelerations 1-D, d 1.5–6 kpc: [distance (kpc)](accelerations_1d_d_d1.5-6.png)
- accelerations 1-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag)](accelerations_1d_colour0.png)
- accelerations 1-D, d 0–0.3 kpc: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag)](accelerations_1d_colour0_d0-0.3.png)
- accelerations 1-D, d 0.3–0.7 kpc: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag)](accelerations_1d_colour0_d0.3-0.7.png)
- accelerations 1-D, d 0.7–1.5 kpc: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag)](accelerations_1d_colour0_d0.7-1.5.png)
- accelerations 1-D, d 1.5–6 kpc: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag)](accelerations_1d_colour0_d1.5-6.png)
- accelerations 1-D: [$M_{G,0}$ (mag)](accelerations_1d_mg0.png)
- accelerations 1-D, d 0–0.3 kpc: [$M_{G,0}$ (mag)](accelerations_1d_mg0_d0-0.3.png)
- accelerations 1-D, d 0.3–0.7 kpc: [$M_{G,0}$ (mag)](accelerations_1d_mg0_d0.3-0.7.png)
- accelerations 1-D, d 0.7–1.5 kpc: [$M_{G,0}$ (mag)](accelerations_1d_mg0_d0.7-1.5.png)
- accelerations 1-D, d 1.5–6 kpc: [$M_{G,0}$ (mag)](accelerations_1d_mg0_d1.5-6.png)
- accelerations 1-D: [$\varpi/\sigma_\varpi$](accelerations_1d_plxsnr.png)
- accelerations 1-D, d 0–0.3 kpc: [$\varpi/\sigma_\varpi$](accelerations_1d_plxsnr_d0-0.3.png)
- accelerations 1-D, d 0.3–0.7 kpc: [$\varpi/\sigma_\varpi$](accelerations_1d_plxsnr_d0.3-0.7.png)
- accelerations 1-D, d 0.7–1.5 kpc: [$\varpi/\sigma_\varpi$](accelerations_1d_plxsnr_d0.7-1.5.png)
- accelerations 1-D, d 1.5–6 kpc: [$\varpi/\sigma_\varpi$](accelerations_1d_plxsnr_d1.5-6.png)
- accelerations 1-D: [significance $s$](accelerations_1d_signif.png)
- accelerations 1-D, d 0–0.3 kpc: [significance $s$](accelerations_1d_signif_d0-0.3.png)
- accelerations 1-D, d 0.3–0.7 kpc: [significance $s$](accelerations_1d_signif_d0.3-0.7.png)
- accelerations 1-D, d 0.7–1.5 kpc: [significance $s$](accelerations_1d_signif_d0.7-1.5.png)
- accelerations 1-D, d 1.5–6 kpc: [significance $s$](accelerations_1d_signif_d1.5-6.png)
- accelerations 1-D: [$F_2$](accelerations_1d_f2.png)
- accelerations 1-D, d 0–0.3 kpc: [$F_2$](accelerations_1d_f2_d0-0.3.png)
- accelerations 1-D, d 0.3–0.7 kpc: [$F_2$](accelerations_1d_f2_d0.3-0.7.png)
- accelerations 1-D, d 0.7–1.5 kpc: [$F_2$](accelerations_1d_f2_d0.7-1.5.png)
- accelerations 1-D, d 1.5–6 kpc: [$F_2$](accelerations_1d_f2_d1.5-6.png)
- accelerations 1-D: [RUWE](accelerations_1d_ruwe.png)
- accelerations 1-D, d 0–0.3 kpc: [RUWE](accelerations_1d_ruwe_d0-0.3.png)
- accelerations 1-D, d 0.3–0.7 kpc: [RUWE](accelerations_1d_ruwe_d0.3-0.7.png)
- accelerations 1-D, d 0.7–1.5 kpc: [RUWE](accelerations_1d_ruwe_d0.7-1.5.png)
- accelerations 1-D, d 1.5–6 kpc: [RUWE](accelerations_1d_ruwe_d1.5-6.png)
- accelerations 2-D: [$G$ (mag) vs distance (kpc)](accelerations_2d_g_d.png)
- accelerations 2-D: [$G$ (mag) vs $(G_{\rm BP}-G_{\rm RP})_0$ (mag)](accelerations_2d_g_colour0.png)
- accelerations 2-D: [$G$ (mag) vs $M_{G,0}$ (mag)](accelerations_2d_g_mg0.png)
- accelerations 2-D: [$G$ (mag) vs $\varpi/\sigma_\varpi$](accelerations_2d_g_plxsnr.png)
- accelerations 2-D: [$G$ (mag) vs significance $s$](accelerations_2d_g_signif.png)
- accelerations 2-D: [$G$ (mag) vs $F_2$](accelerations_2d_g_f2.png)
- accelerations 2-D: [$G$ (mag) vs RUWE](accelerations_2d_g_ruwe.png)
- accelerations 2-D: [distance (kpc) vs $(G_{\rm BP}-G_{\rm RP})_0$ (mag)](accelerations_2d_d_colour0.png)
- accelerations 2-D: [distance (kpc) vs $M_{G,0}$ (mag)](accelerations_2d_d_mg0.png)
- accelerations 2-D: [distance (kpc) vs $\varpi/\sigma_\varpi$](accelerations_2d_d_plxsnr.png)
- accelerations 2-D: [distance (kpc) vs significance $s$](accelerations_2d_d_signif.png)
- accelerations 2-D: [distance (kpc) vs $F_2$](accelerations_2d_d_f2.png)
- accelerations 2-D: [distance (kpc) vs RUWE](accelerations_2d_d_ruwe.png)
- accelerations 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs $M_{G,0}$ (mag)](accelerations_2d_colour0_mg0.png)
- accelerations 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs $\varpi/\sigma_\varpi$](accelerations_2d_colour0_plxsnr.png)
- accelerations 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs significance $s$](accelerations_2d_colour0_signif.png)
- accelerations 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs $F_2$](accelerations_2d_colour0_f2.png)
- accelerations 2-D: [$(G_{\rm BP}-G_{\rm RP})_0$ (mag) vs RUWE](accelerations_2d_colour0_ruwe.png)
- accelerations 2-D: [$M_{G,0}$ (mag) vs $\varpi/\sigma_\varpi$](accelerations_2d_mg0_plxsnr.png)
- accelerations 2-D: [$M_{G,0}$ (mag) vs significance $s$](accelerations_2d_mg0_signif.png)
- accelerations 2-D: [$M_{G,0}$ (mag) vs $F_2$](accelerations_2d_mg0_f2.png)
- accelerations 2-D: [$M_{G,0}$ (mag) vs RUWE](accelerations_2d_mg0_ruwe.png)
- accelerations 2-D: [$\varpi/\sigma_\varpi$ vs significance $s$](accelerations_2d_plxsnr_signif.png)
- accelerations 2-D: [$\varpi/\sigma_\varpi$ vs $F_2$](accelerations_2d_plxsnr_f2.png)
- accelerations 2-D: [$\varpi/\sigma_\varpi$ vs RUWE](accelerations_2d_plxsnr_ruwe.png)
- accelerations 2-D: [significance $s$ vs $F_2$](accelerations_2d_signif_f2.png)
- accelerations 2-D: [significance $s$ vs RUWE](accelerations_2d_signif_ruwe.png)
- accelerations 2-D: [$F_2$ vs RUWE](accelerations_2d_f2_ruwe.png)
