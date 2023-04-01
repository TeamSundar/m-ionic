#!/bin/bash

# Figure 1

convert ROC_per_ion_5x2_withNegatives.png -background white -alpha remove -alpha off fig1a.png
convert fig1a.png -pointsize 70 -font "Times-Bold" -gravity NorthWest -annotate +0+0 "(a)" fig1a1.jpg

convert f1_mcc_esm2_esmMSA_lmetal.png -background white -alpha remove -alpha off fig1b.png
convert fig1b.png -pointsize 70 -font "Times-Bold" -gravity NorthWest -annotate +0+0 "(b)" fig1b1.jpg

montage fig1a1.jpg fig1b1.jpg -gravity center -tile 1x2  -geometry +10+10 figure1.jpg
rm fig1a.png fig1b.png fig1a1.jpg fig1b1.jpg
