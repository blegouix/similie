Include "Inductor/inductor_data.geo";

SetFactory("Built-in");

Mesh.Binary = 0;
Mesh.MshFileVersion = 2.2;
Mesh.RecombineAll = 1;
Mesh.SubdivisionAlgorithm = 1;
Mesh.Smoothing = 0;
Mesh.Optimize = 0;

x1 = wcoreE;
x2 = wcoreE + wcoil;
x3 = 2 * wcoreE + wcoil;

y0 = htot / 2 - hcoreE - ag - hcoreI;
y1 = htot / 2 - hcoreE - ag;
y2 = htot / 2 - hcoreE;
y3 = htot / 2 - hcoreE + hcoil;
y4 = htot / 2 - hcoreE + hcoil + wcoreE;

lc0 = wcoil / nn_wcore;
hcoil_div = Ceil[hcoil / lc0];
nz_layers = Ceil[Lz / lc0];

x_coords[] = {-Rext, -x3, -x2, -x1, 0, x1, x2, x3, Rext};
x_divisions[] = {nn_ro, nn_wcore, nn_wcore, nn_wcore, nn_wcore, nn_wcore, nn_wcore, nn_ro};

y_coords[] = {-Rext, y0, y1, y2, y3, y4, Rext};
y_divisions[] = {nn_ro, nn_wcore, nn_airgap, hcoil_div, nn_wcore, nn_ro};

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
EndFor

Physical Volume(1) = {volumes[]};
