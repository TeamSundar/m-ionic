#!/bin/bash

convert logodds_CA.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(a)" fig1a.png
convert logodds_CO.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(b)" fig1b.png
convert logodds_CU.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(c)" fig1c.png
convert logodds_FE2.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(d)" fig1d.png
convert logodds_FE.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(e)" fig1e.png
convert logodds_MG.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(f)" fig1f.png
convert logodds_MN.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(g)" fig1g.png
convert logodds_PO4.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(h)" fig1h.png
convert logodds_SO4.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(i)" fig1i.png
convert logodds_ZN.png -pointsize 100 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(j)" fig1j.png

montage fig1a.png fig1b.png fig1c.png fig1d.png fig1e.png fig1f.png fig1g.png fig1h.png fig1i.png fig1j.png -tile 2x5 -geometry +2+2 FigureS2.png
rm fig1a.png fig1b.png fig1c.png fig1d.png fig1e.png fig1f.png fig1g.png fig1h.png fig1i.png fig1j.png

