#!/bin/bash

convert A_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(a.1)" a1.png
convert A_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(a.2)" a2.png

convert C_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(b.1)" b1.png
convert C_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(b.2)" b2.png

convert D_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(c.1)" c1.png
convert D_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(c.2)" c2.png

convert E_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(d.1)" d1.png
convert E_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(d.2)" d2.png

convert F_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(e.1)" e1.png
convert F_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(e.2)" e2.png

convert G_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(f.1)" f1.png
convert G_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(f.2)" f2.png

convert H_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(g.1)" g1.png
convert H_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(g.2)" g2.png

convert I_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(h.1)" h1.png
convert I_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(h.2)" h2.png

convert K_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(i.1)" i1.png
convert K_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(i.2)" i2.png

convert L_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(j.1)" j1.png
convert L_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(j.2)" j2.png

convert M_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(k.1)" k1.png
convert M_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(k.2)" k2.png

convert N_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(l.1)" l1.png
convert N_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(l.2)" l2.png

convert P_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(m.1)" m1.png
convert P_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(m.2)" m2.png

convert Q_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(n.1)" n1.png
convert Q_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(n.2)" n2.png

convert R_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(o.1)" o1.png
convert R_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(o.2)" o2.png

convert S_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(p.1)" p1.png
convert S_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(p.2)" p2.png

convert T_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(q.1)" q1.png
convert T_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(q.2)" q2.png

convert V_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(r.1)" r1.png
convert V_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(r.2)" r2.png

convert W_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(s.1)" s1.png
convert W_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(s.2)" s2.png

convert X_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(t.1)" t1.png
convert X_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(t.2)" t2.png

convert Y_dist.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(u.1)" u1.png
convert Y_scatter.png -pointsize 30 -font "Helvetica-Bold" -gravity NorthWest -annotate +0+0 "(u.2)" u2.png

#montage a1.png a2.png b1.png b2.png c1.png c2.png d1.png d2.png e1.png e2.png f1.png f2.png g1.png g2.png h1.png h2.png i1.png i2.png k1.png k2.png l1.png l2.png m1.png m2.png n1.png n2.png p1.png p2.png q1.png q2.png r1.png r2.png s1.png s2.png t1.png t2.png u1.png u2.png -tile 2x21 -geometry +2+2 figs_aa.png

montage a1.png a2.png b1.png b2.png c1.png c2.png d1.png d2.png -tile 2x4 -geometry +2+2 figs_aa1.png
montage e1.png e2.png f1.png f2.png g1.png g2.png h1.png h2.png -tile 2x4 -geometry +2+2 figs_aa2.png
montage i1.png i2.png k1.png k2.png l1.png l2.png m1.png m2.png -tile 2x4 -geometry +2+2 figs_aa3.png
montage n1.png n2.png p1.png p2.png q1.png q2.png r1.png r2.png -tile 2x4 -geometry +2+2 figs_aa4.png
montage s1.png s2.png t1.png t2.png u1.png u2.png -tile 2x3 -geometry +2+2 figs_aa5.png

