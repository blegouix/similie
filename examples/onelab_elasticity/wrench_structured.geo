// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: AGPL-3.0-or-later

Include "original/wrench2D_common.pro";

Solver.AutoMesh = 2;

WRENCH = DefineNumber[1, Name "Input/90SimiLie/0Wrench material physical tag", Visible 0];
VOID = DefineNumber[4, Name "Input/90SimiLie/1Void physical tag", Visible 0];

Young = 1e9 * DefineNumber[200, Name "Material/Young modulus [GPa]"];
Poisson = DefineNumber[0.3, Name "Material/Poisson coefficient []"];
AppliedForce = DefineNumber[100, Name "Material/Applied force [N]"];

Nx = DefineNumber[96, Name "Geometry/90SimiLie structured cells x"];
Ny = DefineNumber[36, Name "Geometry/91SimiLie structured cells y"];

Ri = 0.494 * in;
Re = 0.8 * in;
L = LLength;
F = Width;
theta = Inclination;

xmin = -0.95 * Re;
xmax = L * Cos[theta] + 0.65 * F * Sin[theta];
ymin = -L * Sin[theta] - 0.85 * F;
ymax = 0.95 * Re;

p[] = {};
For j In {0:Ny}
  y = ymin + (ymax - ymin) * j / Ny;
  For i In {0:Nx}
    x = xmin + (xmax - xmin) * i / Nx;
    id = i + (Nx + 1) * j;
    p[id] = newp;
    Point(p[id]) = {x, y, 0, Refine};
  EndFor
EndFor

material_surfaces[] = {};
void_surfaces[] = {};

For j In {0:Ny-1}
  yc = ymin + (ymax - ymin) * (j + 0.5) / Ny;
  For i In {0:Nx-1}
    xc = xmin + (xmax - xmin) * (i + 0.5) / Nx;
    s = xc * Cos[theta] - yc * Sin[theta];
    t = xc * Sin[theta] + yc * Cos[theta];
    r = Sqrt[xc * xc + yc * yc];
    jaw_gap = (xc < -0.18 * Re && Abs[yc] < 0.32 * Re);
    handle = (s > 0 && s < L && Abs[t] < 0.5 * F);
    head = (r < Re && r > Ri && !jaw_gap);
    neck = (s > -0.18 * Re && s < 0.22 * Re && Abs[t] < 0.55 * F);

    l1 = newl; Line(l1) = {p[i + (Nx + 1) * j], p[i + 1 + (Nx + 1) * j]};
    l2 = newl; Line(l2) = {p[i + 1 + (Nx + 1) * j], p[i + 1 + (Nx + 1) * (j + 1)]};
    l3 = newl; Line(l3) = {p[i + 1 + (Nx + 1) * (j + 1)], p[i + (Nx + 1) * (j + 1)]};
    l4 = newl; Line(l4) = {p[i + (Nx + 1) * (j + 1)], p[i + (Nx + 1) * j]};
    ll = newll; Curve Loop(ll) = {l1, l2, l3, l4};
    surf = news; Plane Surface(surf) = {ll};
    Transfinite Curve {l1, l2, l3, l4} = 2;
    Transfinite Surface {surf};
    Recombine Surface {surf};

    If(handle || head || neck)
      material_surfaces[] += surf;
    Else
      void_surfaces[] += surf;
    EndIf
  EndFor
EndFor

Physical Surface(WRENCH) = {material_surfaces[]};
Physical Surface(VOID) = {void_surfaces[]};

Deflection = DefineNumber[
  -4 * AppliedForce * ((LLength - 0.018) / Width)^3 / (Young * Thickness) * 1e3,
  Name "Solution/Deflection (analytical) [mm]",
  ReadOnly 1
];
