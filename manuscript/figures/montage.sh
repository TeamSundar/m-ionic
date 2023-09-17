#!/bin/bash

###### Figure 2 ######

convert Only_Recent_ROC_per_ion_5x2.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(a)" fig2a.png
convert Only_Recent_AUPR_per_ion_5x2.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(b)" fig2b.png

montage fig2a.png fig2b.png -tile 2x1 -geometry +2+2 fig2u.png

convert OnlyRecent_precision_recall_f1_mcc_esm2_lmetal_msa_2x2.png  -resize 130% OnlyRecent_precision_recall_f1_mcc_esm2_lmetal_msa_2x2_1.png 
convert OnlyRecent_precision_recall_f1_mcc_esm2_lmetal_msa_2x2_1.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(c)" fig2c.png

montage fig2u.png fig2c.png -tile 1x2 -geometry +2+2 fig2.png

rm fig2a.png fig2b.png fig2u.png fig2c.png

###### Figure 3 ######
# scrambled_allmeasures.png

: '
###### Figure 4 ######

convert FP_TP.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(a)" fig4a.png

convert all_aa/A_dist.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(b)" fig4b.png

convert all_aa/G_dist.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(c)" fig4c.png

#montage fig4b.png fig4c.png -tile 2x1 -geometry +2+2 fig4u.png
#montage fig4a.png fig4u.png -tile 1x2 -geometry +2+2 fig4.png

montage fig4a.png fig4b.png fig4c.png -tile 3x1 -geometry +2+2 fig4.png

rm fig4a.png fig4b.png fig4c.png fig4u.png 
'

##### Figure S1 #####

convert ROC_per_ion_5x2_withNegatives.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(a)" figs1a.png
convert AUPR_per_ion_5x2_withNegatives.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(b)" figs1b.png

montage figs1a.png figs1b.png -tile 2x1 -geometry +2+2 figs1u.png

convert TestFold6_precision_recall_f1_mcc_esm2_lmetal_2x2.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(c)" figs1c.png

montage figs1u.png figs1c.png -tile 1x2 -geometry +2+2 figs1.png

rm figs1a.png figs1b.png figs1u.png figs1c.png


: '
#convert FP_TP.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(c)" fig1c.png
#montage fig1b.png fig1c.png -tile 2x1 -geometry +2+2 figp1.png

#convert Only_Recent_ROC_per_ion_2x5.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(a)" fig1a.png
#convert OnlyRecent_precision_recall_f1_mcc_esm2_lmetal_2x2.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(b)" fig1b.png

#montage fig1a.png fig1b.png -tile 1x2 -geometry +2+2 fig1.png

#rm  fig1a.png fig1b.png fig1c.png figp1.png



convert ROC_per_ion_5x2_withNegatives.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(a)" fig-s2a.png
convert TestFold6_precision_recall_f1_mcc_esm2_lmetal_2x2.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+3 "(b)" fig-s2b.png
montage fig-s2a.png fig-s2b.png -tile 1x2 -geometry +2+2 fig-s2.png
rm fig-s2a.png fig-s2b.png

#convert logodds_CA.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(a)" fig1a.png
#convert logodds_CO.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(b)" fig1b.png
#convert logodds_CU.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(c)" fig1c.png

#montage fig1a.png fig1b.png fig1c.png fig1d.png fig1e.png fig1f.png fig1g.png fig1h.png fig1i.png fig1j.png -tile 2x5 -geometry +2+2 fig_odds.png

#rm fig1a.png fig1b.png fig1c.png fig1d.png fig1e.png fig1f.png fig1g.png fig1h.png fig1i.png fig1j.png
'
