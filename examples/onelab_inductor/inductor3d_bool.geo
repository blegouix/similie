SetFactory("OpenCASCADE");

Mesh.Binary = 0;
Mesh.MshFileVersion = 2.2;
Mesh.RecombineAll = 1;
Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 0;
Mesh.Optimize = 0;

DefineConstant[
  Irms = {IA, Min 1, Max 4 * IA, Step 2,
    Name "Input/4Coil Parameters/0Current (rms) [A]", Highlight "AliceBlue"},
  NbWires = {Nw,
    Name "Input/4Coil Parameters/1Number of turns", Highlight "AliceBlue"},
  mur_fe = {2000., Min 100, Max 2000, Step 100,
    Name "Input/42Core relative permeability", Highlight "AliceBlue"}
];

jcoil = Sqrt[2] * Irms * NbWires / (wcoil * hcoil);
mu_fe = 4.e-7 * Pi * mur_fe;

DefineConstant[
  SimiLieJz = {jcoil,
    Name "Input/90SimiLie/0Coil current density magnitude z [A/m^2]", Highlight "Ivory"},
  SimiLieMuCore = {mu_fe,
    Name "Input/90SimiLie/1Core magnetic permeability [H/m]", Highlight "Ivory"},
  SimiLieECoreTag = {ECORE,
    Name "Input/90SimiLie/2E-core physical tag", Highlight "Ivory"},
  SimiLieICoreTag = {ICORE,
    Name "Input/90SimiLie/3I-core physical tag", Highlight "Ivory"},
  SimiLieCoilLeftTag = {COIL,
    Name "Input/90SimiLie/4Left coil physical tag", Highlight "Ivory"},
  SimiLieCoilRightTag = {COIL + 1,
    Name "Input/90SimiLie/5Right coil physical tag", Highlight "Ivory"},
  SimiLieAirGapTag = {AIRGAP,
    Name "Input/90SimiLie/6Air-gap physical tag", Highlight "Ivory"}
];

x1 = wcoreE;
x2 = wcoreE + wcoil;
x3 = 2 * wcoreE + wcoil;

y0 = htot / 2 - hcoreE - ag - hcoreI;
y1 = htot / 2 - hcoreE - ag;
y2 = htot / 2 - hcoreE;
y3 = htot / 2 - hcoreE + hcoil;
y4 = htot / 2 - hcoreE + hcoil + wcoreE;

lc0 = wcoil / nn_wcore;
refinement_factor = 8;
hcoil_div = refinement_factor * Ceil[hcoil / lc0];
nz_layers = refinement_factor * Ceil[Lz / lc0];

x_coords[] = {-Rext, -x3, -x2, -x1, 0, x1, x2, x3, Rext};
x_divisions[] = {refinement_factor * nn_ro,
                 refinement_factor * nn_wcore,
                 refinement_factor * nn_wcore,
                 refinement_factor * nn_wcore,
                 refinement_factor * nn_wcore,
                 refinement_factor * nn_wcore,
                 refinement_factor * nn_wcore,
                 refinement_factor * nn_ro};

y_coords[] = {-Rext, y0, y1, y2, y3, y4, Rext};
y_divisions[] = {refinement_factor * nn_ro,
                 refinement_factor * nn_wcore,
                 refinement_factor * nn_airgap,
                 hcoil_div,
                 refinement_factor * nn_wcore,
                 refinement_factor * nn_ro};

num_x = #x_coords[];
num_y = #y_coords[];

For j In {0:num_y - 1}
  For i In {0:num_x - 1}
    points[] += newp;
    Point(newp) = {x_coords[i], y_coords[j], -Lz / 2, lc0};
  EndFor
EndFor

For j In {0:num_y - 1}
  For i In {0:num_x - 2}
    line_id = newl;
    hlines[] += line_id;
    Line(line_id) = {points[i + num_x * j], points[i + 1 + num_x * j]};
    Transfinite Curve {line_id} = x_divisions[i] + 1;
  EndFor
EndFor

For j In {0:num_y - 2}
  For i In {0:num_x - 1}
    line_id = newl;
    vlines[] += line_id;
    Line(line_id) = {points[i + num_x * j], points[i + num_x * (j + 1)]};
    Transfinite Curve {line_id} = y_divisions[j] + 1;
  EndFor
EndFor

For j In {0:num_y - 2}
  For i In {0:num_x - 2}
    h0 = i + (num_x - 1) * j;
    h1 = i + (num_x - 1) * (j + 1);
    v0 = i + num_x * j;
    v1 = i + 1 + num_x * j;
    Line Loop(newll) = {hlines[h0], vlines[v1], -hlines[h1], -vlines[v0]};
    surface_id = news;
    surfaces[] += surface_id;
    Plane Surface(surface_id) = {newll - 1};
    Transfinite Surface {surface_id};
    Recombine Surface {surface_id};

    region_tag = AIR;
    If(i == 0 || i == num_x - 2 || j == 0 || j == num_y - 2)
      region_tag = AIR;
    ElseIf(j == 1)
      region_tag = ICORE;
    ElseIf(j == 2)
      If(i == 3 || i == 4)
        region_tag = AIRGAP;
      ElseIf(i == 1 || i == 6)
        region_tag = ECORE;
      Else
        region_tag = AIR;
      EndIf
    ElseIf(j == 3)
      If(i == 2)
        region_tag = COIL;
      ElseIf(i == 5)
        region_tag = COIL + 1;
      ElseIf(i == 1 || i == 3 || i == 4 || i == 6)
        region_tag = ECORE;
      Else
        region_tag = AIR;
      EndIf
    ElseIf(j == 4)
      region_tag = ECORE;
    EndIf
    surface_region_tags[] += region_tag;
  EndFor
EndFor

For s In {0:#surfaces[] - 1}
  out[] = Extrude {0, 0, Lz} {
    Surface{surfaces[s]};
    Layers{nz_layers};
    Recombine;
  };
  volumes[] += out[1];
  Transfinite Volume {out[1]};

  region_tag = surface_region_tags[s];
  If(region_tag == ECORE)
    ecore_volumes[] += out[1];
  ElseIf(region_tag == ICORE)
    icore_volumes[] += out[1];
  ElseIf(region_tag == AIRGAP)
    airgap_volumes[] += out[1];
  ElseIf(region_tag == COIL)
    coil_left_volumes[] += out[1];
  ElseIf(region_tag == COIL + 1)
    coil_right_volumes[] += out[1];
  Else
    air_volumes[] += out[1];
  EndIf
EndFor

Physical Volume(ECORE) = {ecore_volumes[]};
Physical Volume(ICORE) = {icore_volumes[]};
Physical Volume(AIRGAP) = {airgap_volumes[]};
Physical Volume(COIL) = {coil_left_volumes[]};
Physical Volume(COIL + 1) = {coil_right_volumes[]};
Physical Volume(AIR) = {air_volumes[]};
