# #587 (b): luminous + spurious refit on gens 47-50 (colour_v2.1+tr1), WD off

Luminous gens 47-50 (colour_v2.1+tr1) + refit §12.16 spurious template (v21tr1), refit (8 starts), WD off, R_dark = 0. 1-D: DR3 with Poisson errors, mock total with an MC + Poisson band, components, ratio panel. 2-D: DR3 | mock | (k−μ)/√(μ+σ²_MC). χ² = Σ (k−μ)²/(max(μ, 0.5)+σ²_MC) over bins with μ > 0 or k > 0; MC error from a 30-fold bootstrap of the mock draws.

Fit: -lnL 35610.5; 1/8 starts within 5; N_s pull +3.6; at bound ['a1', 'a2'].


## Ranking (worst first, χ²/dof)

| kind | distribution | split | χ²/dof | χ² ± MC | figure |
|---|---|---|---:|---:|---|
| accelerations | g x f2 | all | 47873.33 | 15463085.0 ± 6993365.6 | [accelerations_2d_g_f2.png](accelerations_2d_g_f2.png) |
| accelerations | g x ruwe | all | 27998.01 | 8959362.8 ± 4320965.8 | [accelerations_2d_g_ruwe.png](accelerations_2d_g_ruwe.png) |
| accelerations | g x colour0 | all | 20905.40 | 6167092.4 ± 6919959.2 | [accelerations_2d_g_colour0.png](accelerations_2d_g_colour0.png) |
| accelerations | plxsnr | d0.3-0.7 | 16685.55 | 367082.0 ± 364555.3 | [accelerations_1d_plxsnr_d0.3-0.7.png](accelerations_1d_plxsnr_d0.3-0.7.png) |
| accelerations | g x signif | all | 14605.58 | 4702995.6 ± 3272729.1 | [accelerations_2d_g_signif.png](accelerations_2d_g_signif.png) |
| accelerations | g x d | all | 14451.68 | 4595633.9 ± 8313905.1 | [accelerations_2d_g_d.png](accelerations_2d_g_d.png) |
| accelerations | g x plxsnr | all | 13162.72 | 4264721.3 ± 8254599.8 | [accelerations_2d_g_plxsnr.png](accelerations_2d_g_plxsnr.png) |
| accelerations | plxsnr x f2 | all | 12512.47 | 4016503.7 ± 1526319.0 | [accelerations_2d_plxsnr_f2.png](accelerations_2d_plxsnr_f2.png) |
| accelerations | colour0 x mg0 | all | 10717.11 | 2432783.5 ± 1517644.1 | [accelerations_2d_colour0_mg0.png](accelerations_2d_colour0_mg0.png) |
| accelerations | signif x ruwe | all | 10496.66 | 3400919.2 ± 1876443.9 | [accelerations_2d_signif_ruwe.png](accelerations_2d_signif_ruwe.png) |
| accelerations | colour0 x plxsnr | all | 10143.17 | 3164668.5 ± 3797433.5 | [accelerations_2d_colour0_plxsnr.png](accelerations_2d_colour0_plxsnr.png) |
| accelerations | mg0 x f2 | all | 9703.27 | 3143858.9 ± 2004463.3 | [accelerations_2d_mg0_f2.png](accelerations_2d_mg0_f2.png) |
| accelerations | signif x f2 | all | 8534.70 | 2765241.5 ± 1602338.0 | [accelerations_2d_signif_f2.png](accelerations_2d_signif_f2.png) |
| accelerations | signif | d0.3-0.7 | 7524.54 | 188113.6 ± 179814.9 | [accelerations_1d_signif_d0.3-0.7.png](accelerations_1d_signif_d0.3-0.7.png) |
| accelerations | colour0 x signif | all | 7364.59 | 2378763.3 ± 1509068.1 | [accelerations_2d_colour0_signif.png](accelerations_2d_colour0_signif.png) |
| accelerations | d x signif | all | 7356.30 | 2199534.4 ± 1699186.3 | [accelerations_2d_d_signif.png](accelerations_2d_d_signif.png) |
| accelerations | f2 x ruwe | all | 6401.83 | 2067790.5 ± 1063821.3 | [accelerations_2d_f2_ruwe.png](accelerations_2d_f2_ruwe.png) |
| accelerations | d x colour0 | all | 6322.72 | 1808297.9 ± 4355535.0 | [accelerations_2d_d_colour0.png](accelerations_2d_d_colour0.png) |
| accelerations | d x mg0 | all | 6320.97 | 1441180.0 ± 5081767.2 | [accelerations_2d_d_mg0.png](accelerations_2d_d_mg0.png) |
| accelerations | plxsnr x ruwe | all | 6244.50 | 1942039.8 ± 947860.1 | [accelerations_2d_plxsnr_ruwe.png](accelerations_2d_plxsnr_ruwe.png) |
| accelerations | plxsnr x signif | all | 5993.26 | 1600199.5 ± 1994983.9 | [accelerations_2d_plxsnr_signif.png](accelerations_2d_plxsnr_signif.png) |
| accelerations | mg0 x plxsnr | all | 5459.20 | 1626842.9 ± 4016912.0 | [accelerations_2d_mg0_plxsnr.png](accelerations_2d_mg0_plxsnr.png) |
| accelerations | mg0 x ruwe | all | 4959.32 | 1572105.8 ± 1214947.1 | [accelerations_2d_mg0_ruwe.png](accelerations_2d_mg0_ruwe.png) |
| accelerations | colour0 x ruwe | all | 4460.97 | 1423050.8 ± 987722.6 | [accelerations_2d_colour0_ruwe.png](accelerations_2d_colour0_ruwe.png) |
| accelerations | f2 | d1.5-6 | 4220.92 | 101302.1 ± 16198.9 | [accelerations_1d_f2_d1.5-6.png](accelerations_1d_f2_d1.5-6.png) |
| accelerations | colour0 x f2 | all | 4139.31 | 1341137.4 ± 1865563.8 | [accelerations_2d_colour0_f2.png](accelerations_2d_colour0_f2.png) |
| accelerations | d x f2 | all | 3937.24 | 1263852.6 ± 2219839.2 | [accelerations_2d_d_f2.png](accelerations_2d_d_f2.png) |
| accelerations | mg0 x signif | all | 3858.47 | 1238569.8 ± 1499710.0 | [accelerations_2d_mg0_signif.png](accelerations_2d_mg0_signif.png) |
| accelerations | f2 | d0.7-1.5 | 3730.94 | 93273.4 ± 867432.8 | [accelerations_1d_f2_d0.7-1.5.png](accelerations_1d_f2_d0.7-1.5.png) |
| accelerations | signif | d0.7-1.5 | 3573.89 | 89347.2 ± 373753.8 | [accelerations_1d_signif_d0.7-1.5.png](accelerations_1d_signif_d0.7-1.5.png) |
| accelerations | g x mg0 | all | 3092.03 | 875045.6 ± 5921043.2 | [accelerations_2d_g_mg0.png](accelerations_2d_g_mg0.png) |
| accelerations | g | d0.7-1.5 | 3067.42 | 76685.5 ± 1624184.7 | [accelerations_1d_g_d0.7-1.5.png](accelerations_1d_g_d0.7-1.5.png) |
| accelerations | d x plxsnr | all | 1624.71 | 375309.2 ± 1406982.4 | [accelerations_2d_d_plxsnr.png](accelerations_2d_d_plxsnr.png) |
| accelerations | ruwe | d0.3-0.7 | 1225.33 | 30633.3 ± 64861.0 | [accelerations_1d_ruwe_d0.3-0.7.png](accelerations_1d_ruwe_d0.3-0.7.png) |
| accelerations | d x ruwe | all | 919.46 | 263886.0 ± 1841160.6 | [accelerations_2d_d_ruwe.png](accelerations_2d_d_ruwe.png) |
| accelerations | ruwe | d0.7-1.5 | 788.58 | 17348.8 ± 17915.4 | [accelerations_1d_ruwe_d0.7-1.5.png](accelerations_1d_ruwe_d0.7-1.5.png) |
| accelerations | mg0 | d1.5-6 | 771.47 | 7714.7 ± 41012.4 | [accelerations_1d_mg0_d1.5-6.png](accelerations_1d_mg0_d1.5-6.png) |
| accelerations | plxsnr | d0-0.3 | 704.38 | 17609.5 ± 25973.6 | [accelerations_1d_plxsnr_d0-0.3.png](accelerations_1d_plxsnr_d0-0.3.png) |
| accelerations | colour0 | d0.7-1.5 | 532.02 | 12236.4 ± 15083.6 | [accelerations_1d_colour0_d0.7-1.5.png](accelerations_1d_colour0_d0.7-1.5.png) |
| accelerations | signif | d1.5-6 | 498.62 | 6980.7 ± 6604.0 | [accelerations_1d_signif_d1.5-6.png](accelerations_1d_signif_d1.5-6.png) |
| accelerations | mg0 | d0.7-1.5 | 370.73 | 6302.5 ± 548120.2 | [accelerations_1d_mg0_d0.7-1.5.png](accelerations_1d_mg0_d0.7-1.5.png) |
| orbits | e | d0-0.3 | 263.26 | 6581.4 ± 619.1 | [orbits_1d_e_d0-0.3.png](orbits_1d_e_d0-0.3.png) |
| orbits | A | d0-0.3 | 258.75 | 6468.7 ± 456.0 | [orbits_1d_A_d0-0.3.png](orbits_1d_A_d0-0.3.png) |
| orbits | e x a0snr | all | 160.38 | 51802.4 ± 7085.5 | [orbits_2d_e_a0snr.png](orbits_2d_e_a0snr.png) |
| orbits | g x a0snr | all | 157.15 | 50289.5 ± 7481.4 | [orbits_2d_g_a0snr.png](orbits_2d_g_a0snr.png) |
| orbits | a0snr x ruwe | all | 139.20 | 41203.6 ± 4138.2 | [orbits_2d_a0snr_ruwe.png](orbits_2d_a0snr_ruwe.png) |
| orbits | e x ruwe | all | 131.18 | 42503.0 ± 5452.7 | [orbits_2d_e_ruwe.png](orbits_2d_e_ruwe.png) |
| orbits | P x e | all | 113.47 | 36762.7 ± 3248.2 | [orbits_2d_P_e.png](orbits_2d_P_e.png) |
| orbits | a0snr x plxsnr | all | 111.54 | 35246.6 ± 4806.5 | [orbits_2d_a0snr_plxsnr.png](orbits_2d_a0snr_plxsnr.png) |
| orbits | g x e | all | 110.68 | 35858.9 ± 4660.5 | [orbits_2d_g_e.png](orbits_2d_g_e.png) |
| orbits | mg0 x A | all | 109.24 | 34083.1 ± 2909.2 | [orbits_2d_mg0_A.png](orbits_2d_mg0_A.png) |
| orbits | A x a0snr | all | 99.89 | 31863.7 ± 3712.4 | [orbits_2d_A_a0snr.png](orbits_2d_A_a0snr.png) |
| orbits | e x plxsnr | all | 97.79 | 31684.4 ± 4130.3 | [orbits_2d_e_plxsnr.png](orbits_2d_e_plxsnr.png) |
| orbits | e x A | all | 96.87 | 31387.1 ± 6698.0 | [orbits_2d_e_A.png](orbits_2d_e_A.png) |
| orbits | A | d0.3-0.7 | 96.57 | 2414.3 ± 901.2 | [orbits_1d_A_d0.3-0.7.png](orbits_1d_A_d0.3-0.7.png) |
| accelerations | ruwe | d1.5-6 | 90.56 | 815.0 ± 3241.5 | [accelerations_1d_ruwe_d1.5-6.png](accelerations_1d_ruwe_d1.5-6.png) |
| orbits | d x e | all | 85.45 | 27685.0 ± 4168.9 | [orbits_2d_d_e.png](orbits_2d_d_e.png) |
| accelerations | g | d1.5-6 | 81.63 | 1632.6 ± 112200.6 | [accelerations_1d_g_d1.5-6.png](accelerations_1d_g_d1.5-6.png) |
| accelerations | mg0 | d0.3-0.7 | 78.55 | 1806.6 ± 8731.2 | [accelerations_1d_mg0_d0.3-0.7.png](accelerations_1d_mg0_d0.3-0.7.png) |
| orbits | g x ruwe | all | 74.99 | 23846.1 ± 4616.9 | [orbits_2d_g_ruwe.png](orbits_2d_g_ruwe.png) |
| orbits | e x f2 | all | 74.48 | 24131.1 ± 3598.1 | [orbits_2d_e_f2.png](orbits_2d_e_f2.png) |
| orbits | mg0 x e | all | 74.14 | 24020.4 ± 5094.5 | [orbits_2d_mg0_e.png](orbits_2d_mg0_e.png) |
| orbits | colour0 x a0snr | all | 73.59 | 23475.6 ± 2507.5 | [orbits_2d_colour0_a0snr.png](orbits_2d_colour0_a0snr.png) |
| orbits | a0snr x f2 | all | 72.46 | 23478.4 ± 2280.0 | [orbits_2d_a0snr_f2.png](orbits_2d_a0snr_f2.png) |
| orbits | colour0 x e | all | 70.94 | 22984.3 ± 3721.9 | [orbits_2d_colour0_e.png](orbits_2d_colour0_e.png) |
| orbits | d x a0snr | all | 70.16 | 21540.5 ± 6448.6 | [orbits_2d_d_a0snr.png](orbits_2d_d_a0snr.png) |
| orbits | g x d | all | 68.89 | 22045.5 ± 4819.7 | [orbits_2d_g_d.png](orbits_2d_g_d.png) |
| orbits | g x A | all | 68.29 | 22126.2 ± 3257.7 | [orbits_2d_g_A.png](orbits_2d_g_A.png) |
| orbits | d x A | all | 63.25 | 18973.9 ± 2309.2 | [orbits_2d_d_A.png](orbits_2d_d_A.png) |
| orbits | g x colour0 | all | 62.09 | 18440.7 ± 2900.4 | [orbits_2d_g_colour0.png](orbits_2d_g_colour0.png) |
| orbits | colour0 x ruwe | all | 60.82 | 19099.0 ± 1632.9 | [orbits_2d_colour0_ruwe.png](orbits_2d_colour0_ruwe.png) |
| accelerations | signif | all | 60.14 | 1503.5 ± 27736.5 | [accelerations_1d_signif.png](accelerations_1d_signif.png) |
| orbits | colour0 x A | all | 59.81 | 18900.8 ± 2029.1 | [orbits_2d_colour0_A.png](orbits_2d_colour0_A.png) |
| orbits | g x plxsnr | all | 57.43 | 18608.8 ± 3503.7 | [orbits_2d_g_plxsnr.png](orbits_2d_g_plxsnr.png) |
| orbits | mg0 | d0.7-1.5 | 56.17 | 954.9 ± 670.3 | [orbits_1d_mg0_d0.7-1.5.png](orbits_1d_mg0_d0.7-1.5.png) |
| orbits | A x ruwe | all | 55.97 | 17966.0 ± 2132.1 | [orbits_2d_A_ruwe.png](orbits_2d_A_ruwe.png) |
| orbits | g x P | all | 54.50 | 17278.0 ± 3399.6 | [orbits_2d_g_P.png](orbits_2d_g_P.png) |
| orbits | mg0 x a0snr | all | 54.11 | 17259.6 ± 5180.5 | [orbits_2d_mg0_a0snr.png](orbits_2d_mg0_a0snr.png) |
| orbits | a0snr | d0.3-0.7 | 53.51 | 1337.6 ± 3318.7 | [orbits_1d_a0snr_d0.3-0.7.png](orbits_1d_a0snr_d0.3-0.7.png) |
| orbits | a0snr | d0.7-1.5 | 52.57 | 1209.1 ± 2041.4 | [orbits_1d_a0snr_d0.7-1.5.png](orbits_1d_a0snr_d0.7-1.5.png) |
| orbits | g | d0.7-1.5 | 51.86 | 1296.6 ± 1211.0 | [orbits_1d_g_d0.7-1.5.png](orbits_1d_g_d0.7-1.5.png) |
| orbits | P x a0snr | all | 50.91 | 16238.9 ± 4611.3 | [orbits_2d_P_a0snr.png](orbits_2d_P_a0snr.png) |
| orbits | g x f2 | all | 47.75 | 14898.2 ± 2368.0 | [orbits_2d_g_f2.png](orbits_2d_g_f2.png) |
| orbits | plxsnr x ruwe | all | 47.33 | 15241.4 ± 2516.8 | [orbits_2d_plxsnr_ruwe.png](orbits_2d_plxsnr_ruwe.png) |
| orbits | A x plxsnr | all | 45.94 | 14471.4 ± 1670.6 | [orbits_2d_A_plxsnr.png](orbits_2d_A_plxsnr.png) |
| orbits | d x colour0 | all | 45.92 | 12536.7 ± 2249.5 | [orbits_2d_d_colour0.png](orbits_2d_d_colour0.png) |
| accelerations | colour0 | d1.5-6 | 45.55 | 956.5 ± 255717.3 | [accelerations_1d_colour0_d1.5-6.png](accelerations_1d_colour0_d1.5-6.png) |
| orbits | g x mg0 | all | 45.51 | 13107.8 ± 3472.6 | [orbits_2d_g_mg0.png](orbits_2d_g_mg0.png) |
| orbits | colour0 x plxsnr | all | 40.27 | 12525.4 ± 1666.3 | [orbits_2d_colour0_plxsnr.png](orbits_2d_colour0_plxsnr.png) |
| orbits | ruwe | d0.7-1.5 | 39.64 | 792.8 ± 869.0 | [orbits_1d_ruwe_d0.7-1.5.png](orbits_1d_ruwe_d0.7-1.5.png) |
| orbits | colour0 x mg0 | all | 38.55 | 7979.7 ± 1201.8 | [orbits_2d_colour0_mg0.png](orbits_2d_colour0_mg0.png) |
| orbits | f2 x ruwe | all | 37.50 | 12075.2 ± 2561.5 | [orbits_2d_f2_ruwe.png](orbits_2d_f2_ruwe.png) |
| orbits | colour0 x P | all | 37.37 | 11919.8 ± 1930.3 | [orbits_2d_colour0_P.png](orbits_2d_colour0_P.png) |
| orbits | colour0 x f2 | all | 37.28 | 11855.9 ± 1486.4 | [orbits_2d_colour0_f2.png](orbits_2d_colour0_f2.png) |
| orbits | e | d1.5-6 | 36.64 | 915.9 ± 265.3 | [orbits_1d_e_d1.5-6.png](orbits_1d_e_d1.5-6.png) |
| orbits | mg0 x ruwe | all | 33.68 | 10407.9 ± 3273.8 | [orbits_2d_mg0_ruwe.png](orbits_2d_mg0_ruwe.png) |
| orbits | mg0 x f2 | all | 31.69 | 10173.4 ± 1810.3 | [orbits_2d_mg0_f2.png](orbits_2d_mg0_f2.png) |
| orbits | P x ruwe | all | 31.49 | 10045.3 ± 1953.3 | [orbits_2d_P_ruwe.png](orbits_2d_P_ruwe.png) |
| orbits | d x f2 | all | 31.11 | 10015.9 ± 1754.0 | [orbits_2d_d_f2.png](orbits_2d_d_f2.png) |
| orbits | P x A | all | 30.47 | 9749.5 ± 1911.1 | [orbits_2d_P_A.png](orbits_2d_P_A.png) |
| orbits | d x ruwe | all | 30.38 | 8596.3 ± 2471.9 | [orbits_2d_d_ruwe.png](orbits_2d_d_ruwe.png) |
| orbits | e | d0.3-0.7 | 30.32 | 758.0 ± 3071.5 | [orbits_1d_e_d0.3-0.7.png](orbits_1d_e_d0.3-0.7.png) |
| orbits | d x mg0 | all | 30.29 | 6664.4 ± 3490.3 | [orbits_2d_d_mg0.png](orbits_2d_d_mg0.png) |
| orbits | f2 | d0-0.3 | 29.68 | 742.1 ± 401.5 | [orbits_1d_f2_d0-0.3.png](orbits_1d_f2_d0-0.3.png) |
| orbits | d x P | all | 29.32 | 8209.6 ± 2097.9 | [orbits_2d_d_P.png](orbits_2d_d_P.png) |
| orbits | plxsnr x f2 | all | 29.08 | 9421.3 ± 1408.4 | [orbits_2d_plxsnr_f2.png](orbits_2d_plxsnr_f2.png) |
| orbits | A x f2 | all | 29.03 | 9377.3 ± 1381.1 | [orbits_2d_A_f2.png](orbits_2d_A_f2.png) |
| orbits | mg0 x P | all | 28.36 | 8706.9 ± 2262.0 | [orbits_2d_mg0_P.png](orbits_2d_mg0_P.png) |
| orbits | g | d1.5-6 | 27.25 | 572.2 ± 248.7 | [orbits_1d_g_d1.5-6.png](orbits_1d_g_d1.5-6.png) |
| orbits | P x f2 | all | 25.95 | 8408.7 ± 1755.9 | [orbits_2d_P_f2.png](orbits_2d_P_f2.png) |
| orbits | g | d0.3-0.7 | 25.02 | 625.6 ± 1971.1 | [orbits_1d_g_d0.3-0.7.png](orbits_1d_g_d0.3-0.7.png) |
| accelerations | colour0 | d0.3-0.7 | 24.00 | 600.0 ± 86413.8 | [accelerations_1d_colour0_d0.3-0.7.png](accelerations_1d_colour0_d0.3-0.7.png) |
| orbits | ruwe | d0.3-0.7 | 23.69 | 592.3 ± 743.0 | [orbits_1d_ruwe_d0.3-0.7.png](orbits_1d_ruwe_d0.3-0.7.png) |
| orbits | P | d0.7-1.5 | 21.11 | 422.2 ± 785.9 | [orbits_1d_P_d0.7-1.5.png](orbits_1d_P_d0.7-1.5.png) |
| orbits | e | all | 20.99 | 524.7 ± 2757.3 | [orbits_1d_e.png](orbits_1d_e.png) |
| orbits | mg0 | d1.5-6 | 20.54 | 205.4 ± 275.9 | [orbits_1d_mg0_d1.5-6.png](orbits_1d_mg0_d1.5-6.png) |
| accelerations | colour0 | d0-0.3 | 20.43 | 510.7 ± 5540.2 | [accelerations_1d_colour0_d0-0.3.png](accelerations_1d_colour0_d0-0.3.png) |
| orbits | mg0 x plxsnr | all | 20.09 | 5847.3 ± 2953.0 | [orbits_2d_mg0_plxsnr.png](orbits_2d_mg0_plxsnr.png) |
| orbits | P x plxsnr | all | 19.09 | 4696.6 ± 1573.1 | [orbits_2d_P_plxsnr.png](orbits_2d_P_plxsnr.png) |
| orbits | a0snr | d1.5-6 | 17.71 | 318.8 ± 142.0 | [orbits_1d_a0snr_d1.5-6.png](orbits_1d_a0snr_d1.5-6.png) |
| orbits | f2 | d1.5-6 | 15.87 | 333.3 ± 227.4 | [orbits_1d_f2_d1.5-6.png](orbits_1d_f2_d1.5-6.png) |
| accelerations | signif | d0-0.3 | 15.80 | 394.9 ± 799.3 | [accelerations_1d_signif_d0-0.3.png](accelerations_1d_signif_d0-0.3.png) |
| orbits | e | d0.7-1.5 | 14.29 | 357.3 ± 2166.8 | [orbits_1d_e_d0.7-1.5.png](orbits_1d_e_d0.7-1.5.png) |
| orbits | P | d0.3-0.7 | 12.38 | 309.4 ± 855.9 | [orbits_1d_P_d0.3-0.7.png](orbits_1d_P_d0.3-0.7.png) |
| orbits | colour0 | d0.7-1.5 | 12.08 | 277.8 ± 91.2 | [orbits_1d_colour0_d0.7-1.5.png](orbits_1d_colour0_d0.7-1.5.png) |
| orbits | A | d1.5-6 | 11.95 | 167.3 ± 79.4 | [orbits_1d_A_d1.5-6.png](orbits_1d_A_d1.5-6.png) |
| accelerations | mg0 | d0-0.3 | 10.71 | 246.2 ± 166.1 | [accelerations_1d_mg0_d0-0.3.png](accelerations_1d_mg0_d0-0.3.png) |
| accelerations | d | d0.7-1.5 | 10.41 | 72.8 ± 844.6 | [accelerations_1d_d_d0.7-1.5.png](accelerations_1d_d_d0.7-1.5.png) |
| orbits | d x plxsnr | all | 10.39 | 2420.3 ± 1279.4 | [orbits_2d_d_plxsnr.png](orbits_2d_d_plxsnr.png) |
| orbits | f2 | d0.7-1.5 | 7.79 | 194.8 ± 270.5 | [orbits_1d_f2_d0.7-1.5.png](orbits_1d_f2_d0.7-1.5.png) |
| orbits | g | all | 6.98 | 174.4 ± 25.5 | [orbits_1d_g.png](orbits_1d_g.png) |
| orbits | a0snr | all | 6.89 | 172.2 ± 72.7 | [orbits_1d_a0snr.png](orbits_1d_a0snr.png) |
| orbits | ruwe | d1.5-6 | 6.31 | 75.7 ± 155.4 | [orbits_1d_ruwe_d1.5-6.png](orbits_1d_ruwe_d1.5-6.png) |
| orbits | A | all | 6.29 | 157.1 ± 88.3 | [orbits_1d_A.png](orbits_1d_A.png) |
| orbits | plxsnr | d0.3-0.7 | 6.25 | 118.8 ± 271.1 | [orbits_1d_plxsnr_d0.3-0.7.png](orbits_1d_plxsnr_d0.3-0.7.png) |
| orbits | d | d0.3-0.7 | 6.12 | 42.8 ± 9.7 | [orbits_1d_d_d0.3-0.7.png](orbits_1d_d_d0.3-0.7.png) |
| orbits | plxsnr | d0.7-1.5 | 5.12 | 66.6 ± 169.0 | [orbits_1d_plxsnr_d0.7-1.5.png](orbits_1d_plxsnr_d0.7-1.5.png) |
| orbits | P | all | 5.10 | 127.4 ± 45.6 | [orbits_1d_P.png](orbits_1d_P.png) |
| orbits | mg0 | all | 5.09 | 127.4 ± 12.3 | [orbits_1d_mg0.png](orbits_1d_mg0.png) |
| orbits | g | d0-0.3 | 4.86 | 121.4 ± 115.4 | [orbits_1d_g_d0-0.3.png](orbits_1d_g_d0-0.3.png) |
| orbits | colour0 | d1.5-6 | 4.70 | 94.0 ± 283.6 | [orbits_1d_colour0_d1.5-6.png](orbits_1d_colour0_d1.5-6.png) |
| orbits | d | all | 4.66 | 116.6 ± 12.4 | [orbits_1d_d.png](orbits_1d_d.png) |
| orbits | plxsnr | all | 4.51 | 112.8 ± 36.9 | [orbits_1d_plxsnr.png](orbits_1d_plxsnr.png) |
| orbits | colour0 | all | 4.46 | 111.5 ± 151.5 | [orbits_1d_colour0.png](orbits_1d_colour0.png) |
| orbits | d | d0-0.3 | 4.43 | 53.2 ± 6.3 | [orbits_1d_d_d0-0.3.png](orbits_1d_d_d0-0.3.png) |
| orbits | a0snr | d0-0.3 | 4.42 | 110.4 ± 151.9 | [orbits_1d_a0snr_d0-0.3.png](orbits_1d_a0snr_d0-0.3.png) |
| orbits | f2 | d0.3-0.7 | 4.39 | 109.8 ± 489.5 | [orbits_1d_f2_d0.3-0.7.png](orbits_1d_f2_d0.3-0.7.png) |
| orbits | f2 | all | 4.30 | 107.5 ± 375.6 | [orbits_1d_f2.png](orbits_1d_f2.png) |
| orbits | ruwe | all | 4.22 | 105.5 ± 27.7 | [orbits_1d_ruwe.png](orbits_1d_ruwe.png) |
| orbits | colour0 | d0.3-0.7 | 4.21 | 101.0 ± 392.1 | [orbits_1d_colour0_d0.3-0.7.png](orbits_1d_colour0_d0.3-0.7.png) |
| orbits | mg0 | d0.3-0.7 | 4.04 | 93.0 ± 63.4 | [orbits_1d_mg0_d0.3-0.7.png](orbits_1d_mg0_d0.3-0.7.png) |
| accelerations | g | d0-0.3 | 3.91 | 97.7 ± 324079.6 | [accelerations_1d_g_d0-0.3.png](accelerations_1d_g_d0-0.3.png) |
| orbits | colour0 | d0-0.3 | 3.77 | 94.2 ± 91.9 | [orbits_1d_colour0_d0-0.3.png](orbits_1d_colour0_d0-0.3.png) |
| orbits | plxsnr | d0-0.3 | 3.72 | 93.1 ± 88.5 | [orbits_1d_plxsnr_d0-0.3.png](orbits_1d_plxsnr_d0-0.3.png) |
| accelerations | f2 | all | 3.59 | 89.8 ± 520.8 | [accelerations_1d_f2.png](accelerations_1d_f2.png) |
| accelerations | g | all | 3.49 | 87.2 ± 134.9 | [accelerations_1d_g.png](accelerations_1d_g.png) |
| accelerations | g | d0.3-0.7 | 3.39 | 84.8 ± 3101.8 | [accelerations_1d_g_d0.3-0.7.png](accelerations_1d_g_d0.3-0.7.png) |
| orbits | plxsnr | d1.5-6 | 3.22 | 25.8 ± 68.9 | [orbits_1d_plxsnr_d1.5-6.png](orbits_1d_plxsnr_d1.5-6.png) |
| orbits | mg0 | d0-0.3 | 3.18 | 69.9 ± 15.3 | [orbits_1d_mg0_d0-0.3.png](orbits_1d_mg0_d0-0.3.png) |
| accelerations | d | d0.3-0.7 | 3.05 | 24.4 ± 35.5 | [accelerations_1d_d_d0.3-0.7.png](accelerations_1d_d_d0.3-0.7.png) |
| accelerations | ruwe | all | 2.88 | 71.9 ± 3893.5 | [accelerations_1d_ruwe.png](accelerations_1d_ruwe.png) |
| orbits | d | d0.7-1.5 | 2.75 | 19.2 ± 550.0 | [orbits_1d_d_d0.7-1.5.png](orbits_1d_d_d0.7-1.5.png) |
| accelerations | f2 | d0-0.3 | 2.72 | 68.0 ± 5523.4 | [accelerations_1d_f2_d0-0.3.png](accelerations_1d_f2_d0-0.3.png) |
| orbits | A | d0.7-1.5 | 2.72 | 62.5 ± 321.7 | [orbits_1d_A_d0.7-1.5.png](orbits_1d_A_d0.7-1.5.png) |
| orbits | P | d0-0.3 | 2.64 | 66.0 ± 247.0 | [orbits_1d_P_d0-0.3.png](orbits_1d_P_d0-0.3.png) |
| orbits | ruwe | d0-0.3 | 2.62 | 65.4 ± 35.5 | [orbits_1d_ruwe_d0-0.3.png](orbits_1d_ruwe_d0-0.3.png) |
| accelerations | f2 | d0.3-0.7 | 2.53 | 63.3 ± 184544.8 | [accelerations_1d_f2_d0.3-0.7.png](accelerations_1d_f2_d0.3-0.7.png) |
| accelerations | ruwe | d0-0.3 | 2.02 | 50.6 ± 1827.1 | [accelerations_1d_ruwe_d0-0.3.png](accelerations_1d_ruwe_d0-0.3.png) |
| accelerations | d | all | 1.59 | 39.7 ± 75.4 | [accelerations_1d_d.png](accelerations_1d_d.png) |
| accelerations | d | d0-0.3 | 1.56 | 17.1 ± 84.8 | [accelerations_1d_d_d0-0.3.png](accelerations_1d_d_d0-0.3.png) |
| accelerations | colour0 | all | 1.46 | 36.6 ± 2864.0 | [accelerations_1d_colour0.png](accelerations_1d_colour0.png) |
| accelerations | plxsnr | d0.7-1.5 | 1.26 | 17.7 ± 2393.8 | [accelerations_1d_plxsnr_d0.7-1.5.png](accelerations_1d_plxsnr_d0.7-1.5.png) |
| accelerations | mg0 | all | 1.21 | 30.3 ± 130.1 | [accelerations_1d_mg0.png](accelerations_1d_mg0.png) |
| accelerations | plxsnr | all | 1.14 | 28.5 ± 83.6 | [accelerations_1d_plxsnr.png](accelerations_1d_plxsnr.png) |
| orbits | d | d1.5-6 | 1.05 | 2.1 ± 1.8 | [orbits_1d_d_d1.5-6.png](orbits_1d_d_d1.5-6.png) |
| orbits | P | d1.5-6 | 0.99 | 9.0 ± 68.4 | [orbits_1d_P_d1.5-6.png](orbits_1d_P_d1.5-6.png) |
| accelerations | plxsnr | d1.5-6 | 0.94 | 6.6 ± 393.0 | [accelerations_1d_plxsnr_d1.5-6.png](accelerations_1d_plxsnr_d1.5-6.png) |
| accelerations | d | d1.5-6 | 0.28 | 0.6 ± 57.6 | [accelerations_1d_d_d1.5-6.png](accelerations_1d_d_d1.5-6.png) |

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
