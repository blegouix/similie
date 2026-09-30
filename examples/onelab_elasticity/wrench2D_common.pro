// Some useful conversion coefficients
mm = 1.e-3;     // millimeters to meters
cm = 1.e-2;     // centimeters to meters
in = 0.0254;    // inches to meters
deg = Pi/180.;  // degrees to radians

Refine =
  mm*DefineNumber[ 2, Name "Geometry/2Main characteristic length"];
Recomb =
  DefineNumber[ 0, Name "Geometry/1Recombine", Choices{0,1}];
Thickness =
  mm*DefineNumber[ 10, Name "Geometry/4Thickness (mm)"];
Width =
  mm*DefineNumber[ 0.625*in/mm, Name "Geometry/5Arm Width (mm)"];
LLength =
  cm*DefineNumber[ 6.0*in/cm, Name "Geometry/6Arm Length (cm)"];

// Definition of the coordinates of the arm end
// at which the maximal deflection will be evaluated.
Inclination = 14*deg; // Arm angle with x-axis
eps = 1e-6;
probe_x = (LLength-eps)*Cos[Inclination];
probe_y =-(LLength-eps)*Sin[Inclination];

// a useful Print command to debug a model or communicate with the user
Printf("Maximal deflection calculated at point (%f,%f)", probe_x, probe_y);
