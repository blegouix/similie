Include "magnus_common.pro";

A = (BoxSize-1)/2.;
B = A;
R = 0.5;
lc  = A/10;


If( Flag_Object == 0 ) // Cylinder

lcr = 0.1;
Point(1) = { 2*R, 0.0, 0.0, lcr};
Point(2) = {   R,  -R, 0.0, lcr};
Point(3) = { 0.0, 0.0, 0.0, lcr};
Point(4) = {   R,   R, 0.0 , lcr};
Point(5) = {   R, 0.0, 0.0 , lcr};
Circle(1) = {1,5,2};
Circle(2) = {2,5,3};
Circle(3) = {3,5,4};
Circle(4) = {4,5,1};

// Points to be connected with the outer boundary
PtA = 4;
PtB = 2;
PtC = 3;
RightSplitY = 0;

Else // naca airfoil

lca = 0.03;
Include "nacaAirfoil.geo";
PtA = 121;
PtB = 81;
PtC = 101;
RightSplitY = A*Tan(Incidence);

EndIf

Point(306)  = { 0, B, 0, lc};
Point(307)  = { 0,-B, 0, lc};

Line(5) = { PtA, 306 };
Line(6) = { PtB, 307 };

Point(308) = {-A,-B, 0, lc};
Point(309) = {-A, B, 0, lc};
Point(310) = { A+1, B, 0, lc};
Point(311) = { A+1,-B, 0, lc};
Point(312) = { A+1, RightSplitY, 0, lc};
Point(313) = {-A, 0, 0, lc};

Line( 7) = { 306, 309 };
Line( 8) = { 309, 313 };
Line( 9) = { 308, 307 };
Line(10) = { 307, 311 };
Line(11) = { 311, 312 };
Line(12) = { 310, 306 };
Line(14) = { 1, 312 };
Line(15) = { PtC, 313 };
Line(16) = { 313, 308 };
Line(17) = { 312, 310 };

Curve Loop(21) = { 1, 6, 10, 11, -14 };
Curve Loop(22) = { 2, 15, 16, 9, -6 };
Curve Loop(23) = { 3, 5, 7, 8, -15 };
Curve Loop(24) = { 4, 14, 17, 12, -5 };

Plane Surface(24) = { 21 };
Plane Surface(25) = { 22 };
Plane Surface(26) = { 23 };
Plane Surface(27) = { 24 };

// Two fluid domains meet along the connecting curves 5 and 6. Each domain
// uses two mapped quadrilateral patches; the obstacle boundary remains a hole.
// Curve 5 is the cut carrying the potential jump in the DEC 1-cochain.
CellsAlongObjectPatch = 40;
CellsAcrossFluidPatch = 24;
Transfinite Curve {1, 2, 3, 4} = CellsAlongObjectPatch + 1;
Transfinite Curve {5, 6, 14, 15} = CellsAcrossFluidPatch + 1;
Transfinite Curve {7, 8, 9, 10, 11, 12, 16, 17} = CellsAlongObjectPatch/2 + 1;
Transfinite Surface {24} = {1, PtB, 307, 312};
Transfinite Surface {25} = {PtB, PtC, 313, 307};
Transfinite Surface {26} = {PtC, PtA, 306, 313};
Transfinite Surface {27} = {PtA, 1, 312, 306};
Recombine Surface {24, 25, 26, 27};

Physical Surface("FluidRight", 2) = { 24, 27 };
Physical Surface("FluidLeft", 3) = { 25, 26 };
Physical Curve("UpStream", 10) = { 8, 16 };
Physical Curve("DownStream", 11) = { 11, 17 };
Physical Curve("Airfoil", 12) = { 1 ... 4 };
Physical Curve("Wake", 13) = { 5 };
