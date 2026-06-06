// SPDX-FileCopyrightText: 1997-2026 C. Geuzaine, J.-F. Remacle
// SPDX-License-Identifier: GPL-2.0-or-later

cm = 1e-2;

If(!Exists(SimiLieInductorDataIncluded))
SimiLieInductorDataIncluded = 1;

DefineConstant[
  Flag_3Dmodel = {0, Choices{0="2D",1="3D"},
    Name "Input/00FE model", Highlight "Blue"},
  Flag_boolean = {0, Choices{0,1},
    Name "Input/00OpenCASCADE model?", Visible Flag_3Dmodel, Highlight "Blue"},
  Flag_Symmetry2D = {1, Choices{0="Full",1="Half"},
    Name "Input/00Symmetry type", Highlight "Blue", Visible (Flag_3Dmodel == 0)},
  Flag_Symmetry3D = {2, Choices{0="Full",1="Half",2="One fourth"},
    Name "Input/01Symmetry type", Highlight "Blue", Visible (Flag_3Dmodel == 1)},
  Flag_OpenCore = {1, Choices{0,1}, ReadOnly Flag_boolean,
    Name "Input/02Core with air gap"}
];

Flag_Symmetry = (Flag_3Dmodel == 0) ? Flag_Symmetry2D : Flag_Symmetry3D;
SymmetryFactor = Flag_Symmetry ? 2 * Flag_Symmetry : 1;
Printf("====> SymmetryFactor=%g", SymmetryFactor);

DefineConstant[
  wcoreE = {3 * cm, Name "Input/10Geometric dimensions/01E-core width of side legs [m]"},
  hcoil = {9 * cm, Name "Input/10Geometric dimensions/04Coil height [m]"},
  wcoil = {wcoreE, ReadOnly 1, Name "Input/10Geometric dimensions/03Coil width [m]"},
  hcoreE = {hcoil + wcoreE, ReadOnly 1, Name "Input/10Geometric dimensions/02E-core height of legs [m]"},
  ag = {0.33 * cm, Min 0.1 * cm, Max 4 * cm, Step 0.2 * cm,
    Visible Flag_OpenCore, Name "Input/10Geometric dimensions/05Air gap width [m]"},
  Lz = {9 * cm, Name "Input/10Geometric dimensions/00Length along z-axis [m]"},
  Rext = {28 * cm, Min 0.15, Max 1, Step 0.1,
    Name "Input/10Geometric dimensions/11Outer shell radius [m]"},
  md = {1., Name "Input/11Mesh control (Nbr of divisions)/0Mesh density"},
  nn_wcore = {Ceil[md * 2], ReadOnly 1,
    Name "Input/11Mesh control (Nbr of divisions)/0Core width"},
  nn_airgap = {Ceil[md], ReadOnly 1,
    Name "Input/11Mesh control (Nbr of divisions)/1Air gap width"},
  nn_ro = {Ceil[md * 6], ReadOnly 1,
    Name "Input/11Mesh control (Nbr of divisions)/3Outer shell"}
];

wcoreE_centralleg = 2 * wcoreE;
wcoreI = 2 * wcoreE + wcoreE_centralleg + 2 * wcoil;
hcoreI = wcoreE;
htot = hcoil + 2 * wcoreE + ag;

IA = 10;
Nw = 288;

ECORE = 1000;
ICORE = 1100;
COIL = 2000;
AIR = 3000;
AIRGAP = 3200;

EndIf
