#!/bin/bash

convert A_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(a)" a1.png
convert C_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(b)" b1.png
convert D_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(c)" c1.png
convert E_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(d)" d1.png
convert F_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(e)" e1.png
convert G_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(f)" f1.png
convert H_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(g)" g1.png
convert I_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(h)" h1.png
convert K_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(i)" i1.png
convert L_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(j)" j1.png
convert M_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(k)" k1.png
convert N_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(l)" l1.png
convert P_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(m)" m1.png
convert Q_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(n)" n1.png
convert R_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(o)" o1.png
convert S_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(p)" p1.png
convert T_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(q)" q1.png
convert V_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(r)" r1.png
convert W_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(s)" s1.png
convert X_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(t)" t1.png
convert Y_dist.png -pointsize 90 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(u)" u1.png

#montage a1.png a2.png b1.png b2.png c1.png c2.png d1.png d2.png e1.png e2.png f1.png f2.png g1.png g2.png h1.png h2.png i1.png i2.png k1.png k2.png l1.png l2.png m1.png m2.png n1.png n2.png p1.png p2.png q1.png q2.png r1.png r2.png s1.png s2.png t1.png t2.png u1.png u2.png -tile 2x21 -geometry +2+2 figs_aa.png

#montage a1.png a2.png b1.png b2.png c1.png c2.png d1.png d2.png -tile 2x4 -geometry +2+2 figs_aa1.png
#montage e1.png e2.png f1.png f2.png g1.png g2.png h1.png h2.png -tile 2x4 -geometry +2+2 figs_aa2.png
#montage i1.png i2.png k1.png k2.png l1.png l2.png m1.png m2.png -tile 2x4 -geometry +2+2 figs_aa3.png
#montage n1.png n2.png p1.png p2.png q1.png q2.png r1.png r2.png -tile 2x4 -geometry +2+2 figs_aa4.png
#montage s1.png s2.png t1.png t2.png u1.png u2.png -tile 2x3 -geometry +2+2 figs_aa5.png

montage a1.png b1.png c1.png d1.png e1.png f1.png g1.png h1.png i1.png j1.png k1.png -tile 3x4 -geometry +2+2 figs31.png
montage l1.png m1.png n1.png o1.png p1.png q1.png r1.png s1.png t1.png u1.png -tile 3x4 -geometry +2+2 figs32.png

