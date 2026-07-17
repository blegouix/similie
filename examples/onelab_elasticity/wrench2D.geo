// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

Include "wrench2D_common.pro";

Solver.AutoMesh = 2;

WRENCH = DefineNumber[1, Name "Input/90SimiLie/0Wrench material physical tag", Visible 0];

Young = 1e9 * DefineNumber[200, Name "Material/Young modulus [GPa]"];
Poisson = DefineNumber[0.3, Name "Material/Poisson coefficient []"];
AppliedForce = DefineNumber[100, Name "Material/Applied force [N]"];

lc = Refine;
lc2 = lc;
lc3 = lc / 5;

Ri = 0.494 * in;
Re = 0.8 * in;
L = LLength;
LL = 1 * in;
E = Width;
F = Width;
D = 0.45 * in;
theta = Inclination;

Point(1) = {0, 0, 0, lc2};
Point(2) = {-Ri, 0, 0, lc2};
th1 = Asin[E / 2. / Ri];
Point(3) = {-Ri + Ri * Cos[th1], Ri * Sin[th1], 0, lc2};
th2 = Asin[E / 2. / Re];
Point(4) = {-Re * Cos[th2] + D, Re * Sin[th2], 0, lc3};
Point(5) = {-Re * Cos[th2], Re * Sin[th2], 0, lc2};
th3 = th2 - theta;
Point(6) = {Re * Cos[th3], Re * Sin[th3], 0, lc};
Point(7) = {(L - LL) * Cos[theta] + F / 2. * Sin[theta],
            -(L - LL) * Sin[theta] + F / 2. * Cos[theta], 0, lc};
Point(8) = {L * Cos[theta] + F / 2. * Sin[theta],
            -L * Sin[theta] + F / 2. * Cos[theta], 0, lc};
Point(9) = {L * Cos[theta] - F / 2. * Sin[theta],
            -L * Sin[theta] - F / 2. * Cos[theta], 0, lc};
th4 = -th2 - theta;
Point(10) = {Re * Cos[th4], Re * Sin[th4], 0, lc};
Point(11) = {-Re * Cos[th2], -Re * Sin[th2], 0, lc2};
Point(12) = {-Re * Cos[th2] + D, -Re * Sin[th2], 0, lc3};
Point(13) = {-Ri + Ri * Cos[th1], -Ri * Sin[th1], 0, lc2};

lA1 = newl; Circle(lA1) = {1, 2, 3};
lA2 = newl; Line(lA2) = {3, 4};
lB1 = newl; Line(lB1) = {4, 5};
lB2 = newl; Circle(lB2) = {5, 1, 6};
lB3 = newl; Line(lB3) = {6, 7};
lB4 = newl; Line(lB4) = {7, 8};
lC1 = newl; Line(lC1) = {8, 9};
lD1 = newl; Line(lD1) = {9, 10};
lD2 = newl; Circle(lD2) = {10, 1, 11};
lD3 = newl; Line(lD3) = {11, 12};
lA3 = newl; Line(lA3) = {12, 13};
lA4 = newl; Circle(lA4) = {13, 2, 1};
lH = newl; Line(lH) = {6, 10};

Transfinite Curve {lB2, lD2} = 17;
Transfinite Curve {lH, lC1} = 65;
Transfinite Curve {lB3} = 41;
Transfinite Curve {lB4, lA2, lB1, lD3, lA3} = 9;
Transfinite Curve {lD1} = 49;
Transfinite Curve {lA1} = 25;
Transfinite Curve {lA4} = 9;

llHead = newll;
Curve Loop(llHead) = {lB2, lH, lD2, lD3, lA3, lA4, lA1, lA2, lB1};
sHead = news;
Plane Surface(sHead) = {llHead};
Transfinite Surface {sHead} = {5, 6, 10, 11};
Recombine Surface {sHead};

llHandle = newll;
Curve Loop(llHandle) = {lB3, lB4, lC1, lD1, -lH};
sHandle = news;
Plane Surface(sHandle) = {llHandle};
Transfinite Surface {sHandle} = {6, 8, 9, 10};
Recombine Surface {sHandle};

Physical Surface(WRENCH) = {sHead, sHandle};
Physical Curve(2) = {lB1, lD3};
Physical Curve(3) = {lC1};

Deflection = DefineNumber[
  -4 * AppliedForce * ((LLength - 0.018) / Width)^3 / (Young * Thickness) * 1e3,
  Name "Solution/Deflection (analytical) [mm]",
  ReadOnly 1
];
