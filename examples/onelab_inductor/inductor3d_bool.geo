SetFactory("OpenCASCADE");

If(!Exists(SimiLieInductorDataIncluded))
  Flag_3Dmodel = 1;
  Flag_boolean = 1;
  Include "inductor_data.geo";
EndIf

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
COIL_X_POS = COIL + 2;
COIL_X_NEG = COIL + 3;
lc0 = wcoil / nn_wcore;
refinement_factor = 8;
hcoil_div = refinement_factor * Ceil[hcoil / lc0];
nz_layers = refinement_factor * Ceil[Lz / lc0];
end_turn_layers = 1;
end_turn_depth = wcoil;
middle_layers = nz_layers;
middle_depth = Lz;

DefineConstant[
  SimiLieJz = {jcoil,
    Name "Input/90SimiLie/0Coil current density magnitude z [A/m^2]", Highlight "Ivory"},
  SimiLieJxEndTurn = {jcoil,
    Name "Input/90SimiLie/7Coil end-turn current density magnitude x [A/m^2]", Highlight "Ivory"},
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
    Name "Input/90SimiLie/6Air-gap physical tag", Highlight "Ivory"},
  SimiLieCoilXPositiveTag = {COIL_X_POS,
    Name "Input/90SimiLie/8Positive x end-turn physical tag", Highlight "Ivory"},
  SimiLieCoilXNegativeTag = {COIL_X_NEG,
    Name "Input/90SimiLie/9Negative x end-turn physical tag", Highlight "Ivory"}
];

x1 = wcoreE;
x2 = wcoreE + wcoil;
x3 = 2 * wcoreE + wcoil;

y0 = htot / 2 - hcoreE - ag - hcoreI;
y1 = htot / 2 - hcoreE - ag;
y2 = htot / 2 - hcoreE;
y3 = htot / 2 - hcoreE + hcoil;
y4 = htot / 2 - hcoreE + hcoil + wcoreE;

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
    Point(newp) = {x_coords[i], y_coords[j], -Lz / 2 - end_turn_depth, lc0};
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
    surface_i_indices[] += i;
    surface_j_indices[] += j;

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
  original_surface = surfaces[s];
  region_tag = surface_region_tags[s];
  surface_i = surface_i_indices[s];
  surface_j = surface_j_indices[s];
  is_end_turn_bridge = (surface_j == 3 && surface_i >= 2 && surface_i <= 5);

  out_front[] = Extrude {0, 0, end_turn_depth} {
    Surface{original_surface};
    Layers{end_turn_layers};
    Recombine;
  };
  front_volume = out_front[1];
  Transfinite Volume {front_volume};

  out_middle[] = Extrude {0, 0, middle_depth} {
    Surface{out_front[0]};
    Layers{middle_layers};
    Recombine;
  };
  middle_volume = out_middle[1];
  Transfinite Volume {middle_volume};

  out_back[] = Extrude {0, 0, end_turn_depth} {
    Surface{out_middle[0]};
    Layers{end_turn_layers};
    Recombine;
  };
  back_volume = out_back[1];
  Transfinite Volume {back_volume};

  volumes[] += front_volume;
  volumes[] += middle_volume;
  volumes[] += back_volume;

  If(is_end_turn_bridge)
    coil_x_negative_volumes[] += front_volume;
  Else
    air_volumes[] += front_volume;
  EndIf

  If(region_tag == ECORE)
    ecore_volumes[] += middle_volume;
  ElseIf(region_tag == ICORE)
    icore_volumes[] += middle_volume;
  ElseIf(region_tag == AIRGAP)
    airgap_volumes[] += middle_volume;
  ElseIf(region_tag == COIL)
    coil_left_volumes[] += middle_volume;
  ElseIf(region_tag == COIL + 1)
    coil_right_volumes[] += middle_volume;
  Else
    air_volumes[] += middle_volume;
  EndIf

  If(is_end_turn_bridge)
    coil_x_positive_volumes[] += back_volume;
  Else
    air_volumes[] += back_volume;
  EndIf
EndFor

Physical Volume(ECORE) = {ecore_volumes[]};
Physical Volume(ICORE) = {icore_volumes[]};
Physical Volume(AIRGAP) = {airgap_volumes[]};
Physical Volume(COIL) = {coil_left_volumes[]};
Physical Volume(COIL + 1) = {coil_right_volumes[]};
Physical Volume(COIL_X_POS) = {coil_x_positive_volumes[]};
Physical Volume(COIL_X_NEG) = {coil_x_negative_volumes[]};
Physical Volume(AIR) = {air_volumes[]};
