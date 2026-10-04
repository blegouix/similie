cm = 1e-2;
deg = Pi/180;
kmh = 10./36.;

Flag_Circ = 
  DefineNumber[0, Name "Model/Impose circulation", Choices {0, 1} ];
Flag_Object = 
  DefineNumber[1, Name "Model/Object",
               Choices {0="Cylinder", 1="Airfoil"} ];

BoxSize = 
  DefineNumber[7, Name "Model/Air box size [m]", Help "In flow direction."];
Velocity = kmh*
  DefineNumber[100, Name "Model/V", Label "Model/Flow velocity [km/h]"];
Incidence = -deg*
  DefineNumber[10, Name "Model/Angle of attack [deg]", Visible Flag_Object];

// Negative circulation for a positive lift. 
DeltaPhi = (-1)*
  DefineNumber[10, Name "Model/Circ", Visible Flag_Circ,
	       Min 0, Max 20, Step 2, 
	       Label "Circulation around foil [m^2/s]"];

MassFlowRate = 
  DefineNumber[-100, Name "Model/Dmdt", Visible !Flag_Circ,
	       Label "Mass flow rate [kg/s]"];

