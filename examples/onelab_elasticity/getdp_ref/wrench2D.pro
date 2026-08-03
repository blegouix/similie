// SPDX-FileCopyrightText: 1997-2026 C. Geuzaine, J.-F. Remacle
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Adapted from GetDP ONELAB example files for the SimiLie wrench geometry.

Include "../wrench2D_common.pro";

iP = 1;
WRENCH = 1;
CLAMP = 2;
LOAD = 3;

DefineConstant[
  GetDPOutputDir = {StrCat[CurrentDirectory, "res_elasticity"], Name "GetDP/0Output directory"},
  Young = {200e9, Name "Material/Young modulus [Pa]"},
  Poisson = {0.3, Name "Material/Poisson coefficient []"},
  AppliedForce = {100, Name "Material/Applied force [N]"}
];

Group {
  Vol_Elast~{iP} = Region[{WRENCH}];
  Vol_Force~{iP} = Region[{}];
  Sur_Force~{iP} = Region[{LOAD}];
  Sur_Clamp~{iP} = Region[{CLAMP}];
  Sur_Disp_x~{iP} = Region[{}];
  Sur_Disp_y~{iP} = Region[{}];
  Sur_Disp_z~{iP} = Region[{}];
  Domain_Dim_2~{iP} = Region[{WRENCH}];
}

Function {
  CoefJac~{iP}[] = Thickness;

  E[Vol_Elast~{iP}] = Young;
  nu[Vol_Elast~{iP}] = Poisson;

  pressure_x[Sur_Force~{iP}] = 0;
  pressure_y[Sur_Force~{iP}] = -AppliedForce / Width;
  pressure_z[Sur_Force~{iP}] = 0;

  force_x[Vol_Force~{iP}] = 0;
  force_y[Vol_Force~{iP}] = 0;
  force_z[Vol_Force~{iP}] = 0;

  displacement_x[] = 0;
  displacement_y[] = 0;
  displacement_z[] = 0;
}

Include "Lib_Elast_u.pro";

PostOperation Get_LocalFields UsingPost Elast_u {
  CreateDir[GetDPOutputDir];
  Print[u, OnElementsOf Vol_Elast~{iP}, File StrCat[GetDPOutputDir, "/u.pos"], LastTimeStepOnly];
  Print[sig_xx, OnElementsOf Vol_Elast~{iP}, File StrCat[GetDPOutputDir, "/sig_xx.pos"], LastTimeStepOnly];
  Print[sig_xy, OnElementsOf Vol_Elast~{iP}, File StrCat[GetDPOutputDir, "/sig_xy.pos"], LastTimeStepOnly];
  Print[sig_yy, OnElementsOf Vol_Elast~{iP}, File StrCat[GetDPOutputDir, "/sig_yy.pos"], LastTimeStepOnly];
}

PostOperation Get_Probe_Displacement UsingPost Elast_u {
  CreateDir[GetDPOutputDir];
  Print[u, OnPoint {probe_x, probe_y, 0}, Format Table,
    File StrCat[GetDPOutputDir, "/u_probe.txt"], LastTimeStepOnly];
}
