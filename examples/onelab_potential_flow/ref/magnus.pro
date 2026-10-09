/* -------------------------------------------------------------------
   Tutorial 6 : Potential flow and Magnus effect

   Features:
   - Potential flow, irrotational flow
   - Multivalued scalar field
   - Lift and Magnus effect, stagnation points
   - Run-time variables
   - Elementary algorithms in the Resolution section
   - Non-linear iteration to achieve Kutta's condition

   To compute the solution in a terminal:
     getdp magnus.pro -solve PotentialFlow -pos PotentialFlow
     gmsh magnus.geo velocity.pos

   To compute the solution interactively from the Gmsh GUI:
       File > Open > magnus.pro
       Run (button at the bottom of the left panel)
   ------------------------------------------------------------------- */

/* This model solves a 2D potential flow around a cylinder or a naca airfoil
   placed in a uniform "V_infinity" flow.  Potential flows are defined by

   V = grad phi => curl V = 0

   where the multivalued scalar potential "phi" presents a discontinuity of
   magnitude "deltaPhi" across a cut "Sur_Cut".  In consequence, the velocity
   field "V" is curl-free over the wole domain of analysis "Vol_rho" but its
   circulation over any closed curve circling around the object, a quantity
   often noted "Gamma" in the literature, gives "deltaPhi" as a result.  The
   circulation of "V" over closed curves not circling around the object is zero.

   In the GetDP model, the discontinuity "deltaPhi" [m^2/s] is defined as a
   global quantity in the Functional space of "phi".  The associated (dual)
   global quantity is the mass flow density, noted "Dmdt", which has [kg/s] as a
   unit.

   Governing equations are

   div ( rho[] grad phi ) = 0         in Vol_rho
   rho[] grad phi . n = rho[] Vn = 0  on Sur_Neu

   whereas the uniform velocity field "V_infinity" is imposed by means of
   Dirichlet boundary conditions on the upstream and downstream surfaces
   "GammaUp" and "GammaDown".

   Momentum equation is decoupled in case of stationary potential flows.  The
   velocity filed can be solved first, and the pressure is obtained afterwards
   by invoking Bernoulli's theorem:

   p = -0.5 * rho[] * SquNorm [ {d phi} ]

   The "Lift" force in "Y" direction is then evaluated either by integrating "p
   * CompY[Normal[]]" over the contour of the object, or by the Kutta-Jukowski
   theorem

   Lift = - L_z rho[] V_infinity Gamma

   which is a first order approximation of the former.

   "Cylinder" case:

   Either the circulation "deltaPhi" or the mass flow rate "Dmdt" can be
   imposed, according to the "Flag_Circ" flag.  The Lift is evaluated by both
   the integration of pressure and the Kutta-Jukowski approximation.

   "Airfoil" case:

   - If "Flag_Circ == 1", the model works similar to the "Cylinder" case.

   - If "Flag_Circ == 0", the mass flow rate is not directly imposed in this
     case. It is determined so that the Kutta condition is verified.  This
     condition states that the actual value of "Dmdt" is the one for which the
     stagnation point (singularity of the velocity field) is located at the
     trailing edge of the airfoil,.  This is true when the function

     argVTrail[] = ( Atan2[CompY[$1], CompX[$1] ] - Incidence ) / deg ;

     returns zero when evaluated for the velocity at a point "P_edge" close to
     the trailing edge of the airfoil.  A non linear iteration is thus done by
     means of a pseudo-newton scheme, updating the value of the imposed "Dmdt"
     until the norm of the residual

     f(V) = argVTrail[ {d phi } ] OnPoint {1.0001,0,0}

     comes below a fixed tolerance. */

Include "../magnus_common.pro";

// Keep the reference solver's command-line parameters independent of the
// geometry's ONELAB database. GetDP's DefineNumber values are UI controls.
DefineConstant[
  RefObject = {1, Name "GetDP/Reference object"},
  RefImposeCirculation = {0, Name "GetDP/Reference impose circulation"},
  RefBoxSize = {7, Name "GetDP/Reference box size"},
  RefVelocity = {100./3.6, Name "GetDP/Reference velocity"},
  RefIncidence = {-10.*Pi/180., Name "GetDP/Reference incidence"},
  RefCirculation = {-10, Name "GetDP/Reference circulation"},
  RefMassFlowRate = {-100, Name "GetDP/Reference mass flow rate"}
];
Flag_Object = RefObject;
Flag_Circ = RefImposeCirculation;
BoxSize = RefBoxSize;
Velocity = RefVelocity;
Incidence = RefIncidence;
DeltaPhi = RefCirculation;
MassFlowRate = RefMassFlowRate;

Group{
  // Physical regions
  Fluid     = Region[ { 2, 3 } ];
  GammaUp   = Region[ 10 ];
  GammaDown = Region[ 11 ];
  GammaAirf = Region[ 12 ];
  GammaCut = Region[ 13 ];

  // Abstract regions
  Vol_rho = Region[ { Fluid } ];
  Sur_Dir = Region[ { GammaUp, GammaDown } ];
  Sur_Neu = Region[ { GammaAirf } ];
  Sur_Cut = Region[ { GammaCut } ];
}

Function{
  rho[] = 1.225; // kg/m^3
  MassFlowRate[] = MassFlowRate; // kg/s
  DeltaPhi[] = DeltaPhi; // m^2/s
  argVTrail[] = ( Atan2[CompY[$1], CompX[$1] ] - Incidence ) / deg ;
}

Jacobian {
  { Name Vol ;
    Case {
      { Region All ; Jacobian Vol ; }
    }
  }
  { Name Sur ;
    Case {
      { Region All ; Jacobian Sur ; }
    }
  }
}

Integration {
  { Name Int ;
    Case {
      { Type Gauss ;
        Case {
          { GeoElement Point       ; NumberOfPoints  1 ; }
          { GeoElement Line        ; NumberOfPoints  4 ; }
          { GeoElement Quadrangle  ; NumberOfPoints  7 ; }
        }
      }
    }
  }
}

Constraint{
  { Name Dirichlet; Type Assign;
    Case{
      // Boundary conditions for the V_infinity uniform flow
      { Region GammaDown; Value Velocity*BoxSize; }
      { Region GammaUp  ; Value 0; }
    }
  }
  { Name DeltaPhi; Type Assign;
    Case{
      { Region Sur_Cut; Value DeltaPhi[]; }
    }
  }
  { Name MassFlowRate; Type Assign;
    Case{
      { Region Sur_Cut; Value MassFlowRate[]; }
    }
  }
}

// Domains of definition used in the description of the function space
// These groups contain both volume and surface regions/elements.
Group{
  Dom_Vh =  Region[ { Vol_rho, Sur_Dir, Sur_Neu, Sur_Cut } ];
  Dom_Cut = ElementsOf[ Dom_Vh, OnPositiveSideOf Sur_Cut ];
}
FunctionSpace{
  { Name Vh; Type Form0;
    BasisFunction{
      { Name vn; NameOfCoef phin; Function BF_Node;
	Support Dom_Vh; Entity NodesOf[All]; }
      { Name vc; NameOfCoef dphi; Function BF_GroupOfNodes;
	Support Dom_Cut; Entity GroupsOfNodesOf[ Sur_Cut ];}
    }
    SubSpace {
      { Name phiCont ; NameOfBasisFunction { vn } ; }
      { Name phiDisc ; NameOfBasisFunction { vc } ; }
    }
    GlobalQuantity {
      { Name Circ ; Type AliasOf        ; NameOfCoef dphi ; }
      { Name Dmdt ; Type AssociatedWith ; NameOfCoef dphi ; }
    }
    Constraint{
      {NameOfCoef phin; EntityType NodesOf;
	NameOfConstraint Dirichlet;}
      If( Flag_Circ )
        {NameOfCoef Circ; EntityType GroupsOfNodesOf;
      	  NameOfConstraint DeltaPhi;}
      Else
        If( Flag_Object == 0 ) // if Cylinder only
      // In case of the Airfoil, the contraint on Dmdt is omitted
      // and replaced by an equation "Dmdt=$newDmdt"
      // See the "Resolution" section.
          {NameOfCoef Dmdt; EntityType GroupsOfNodesOf;
      	  NameOfConstraint MassFlowRate;}
        EndIf
      EndIf
    }
  }
}

Formulation{
  {Name PotentialFlow; Type FemEquation;
    Quantity{
      {Name phi; Type Local; NameOfSpace Vh;}
      { Name phiCont ; Type Local ; NameOfSpace Vh[phiCont] ; }
      { Name phiDisc ; Type Local ; NameOfSpace Vh[phiDisc] ; }
      { Name Circ ; Type Global ; NameOfSpace Vh[Circ] ; }
      { Name Dmdt ; Type Global ; NameOfSpace Vh[Dmdt] ; }
    }
    Equation{
      Integral{[ rho[] * Dof{d phi}, {d phi}];
        In Vol_rho; Jacobian Vol; Integration Int;}
      If( !Flag_Circ )
	GlobalTerm { [ -Dof{Dmdt} , {Circ} ] ; In Sur_Cut ; }
        If( Flag_Object == 1 )
          GlobalTerm { [ -Dof{Dmdt} , {Dmdt} ] ; In Sur_Cut ; }
	  GlobalTerm { [ $newDmdt , {Dmdt} ] ; In Sur_Cut ; }
        EndIf
      EndIf
    }
  }
}

Resolution{
  {Name PotentialFlow;
    System{
      { Name A; NameOfFormulation PotentialFlow; }
    }
    Operation{
      InitSolution[A];

      If( Flag_Circ || ( Flag_Object == 0 ) )

        Generate[A]; Solve[A]; SaveSolution[A];
        PostOperation[Trailing];

      Else
	// A resolution can contain elementary algorithms.
	// Available commands are:
	// Evaluate[] : affectation of a run-time variable
	// Test[]{}{} : logical test
	// While[]{} : iteration
        // Print[{}, Format ..., File ...] : formatted display

	// A pseudo-Newton iteration is implemented here to determine the value
	// of Dmdt (in the Airfoil case) that verifies Kutta's condition.  The
	// run-time variable $newDmdt is used in Generate[A] whereas $circ,
	// $dmdt, $argV and $phiTrailing are evaluated by the
	// PostOperation[Trailing].

	DeleteFile["KJiter.txt"];

        Evaluate[$newDmdt = MassFlowRate];
        Evaluate[ $syscount = 0 ];
        Generate[A]; Solve[A]; SaveSolution[A];

	Evaluate[$dmdtp = MassFlowRate];
	Evaluate[$argVp = DeltaPhi];
        PostOperation[Trailing];

        Print[{$syscount, $circ, $dmdt, $argV},
	    Format "iter = %3g Circ = %5.2f Dmdt = %5.2f argV = %5.2e"];

        While[ Norm[ $argV ] > 1e-3 && $syscount < 50] {
	  Test[ $syscount && Norm[$argV-$argVp] > 1e-3 ]
	    {Evaluate[$jac = Min[ ($dmdt-$dmdtp)/($argV-$argVp), 0.2 ] ] ;}
  	    {Evaluate[$jac = 0.2];}

	  Evaluate[$newDmdt = $dmdt - $jac * $argV];
	  Evaluate[ $syscount = $syscount + 1 ];
	  Generate[A]; Solve[A]; SaveSolution[A];

	  Evaluate[$dmdtp = $dmdt];
	  Evaluate[$argVp = $argV];
	  PostOperation[Trailing];

	  Print[{$syscount, $circ, $dmdt, $jac, $argV},
		Format "iter = %3g Circ = %5.2f Dmdt = %5.2f jac=%5.2f argV = %5.3e"];
	}
      EndIf
    }
  }
}

PostProcessing{
  { Name PotentialFlow; NameOfFormulation PotentialFlow;
    Quantity{
      { Name phi; Value {
	  Term{ [ {phi} ] ; In Dom_Vh; Jacobian Vol; }
        }
      }
      { Name phiCont ; Value {
	  Term { [ { phiCont } ] ; In Dom_Vh ; Jacobian Vol ; }
        }
      }
      { Name phiDisc ; Value {
 	  Term { [ { phiDisc } ] ; In Dom_Vh ; Jacobian Vol ; }
        }
      }
      { Name velocity; Value {
	  Term { [ {d phi} ]; In Dom_Vh; Jacobian Vol; }
        }
      }
      { Name normVelocity; Value {
	  Term { [ Norm[{d phi}] ]; In Dom_Vh; Jacobian Vol; }
        }
      }
      { Name pressure; Value {
	  Term { [-0.5*rho[]*SquNorm[ {d phi} ]]; In Dom_Vh; Jacobian Vol; }
        }
      }
      { Name Angle; Value {
          Term{ [ argVTrail[{d phi}] ]; In Dom_Vh; Jacobian Vol; }
        }
      }
      { Name Circ; Value {
          Term { [ {Circ} ]; In Sur_Cut; }
        }
      }
      { Name Dmdt; Value {
          Term { [ {Dmdt} ]; In Sur_Cut; }
        }
      }
      // Kutta-Jukowski approximation for Lift
      { Name LiftKJ; Value {
          Term { [ -rho[]*{Circ}*Velocity ]; In Sur_Cut; }
        }
      }
      // Lift computed with the real pressure field
      { Name Lift; Value {
          Integral { [ -0.5*rho[]*SquNorm[{d phi}]*CompY[Normal[]] ];
            In Dom_Vh ; Jacobian Sur ; Integration Int; }
        }
      }
      { Name circulation; Value {
          Integral { [ {d phi} * Tangent[] ] ;
            In Dom_Vh ; Jacobian Sur  ; Integration Int; }
        }
      }
    }
  }
}

PostOperation{
  {Name PotentialFlow; NameOfPostProcessing PotentialFlow;
    Operation{

      Print[Circ, OnRegion Sur_Cut, File > "output.txt", Color "Ivory",
	   Format Table, SendToServer "Output/Circ" ];

      Print[Dmdt, OnRegion Sur_Cut, File > "output.txt", Color "Ivory",
	    Format Table, SendToServer "Output/Dmdt" ];

      Print[LiftKJ, OnRegion Sur_Cut, File > "output.txt", Color "Ivory",
 	   Format Table, SendToServer "Output/LiftKJ" ];

      Print[Lift[GammaAirf], OnGlobal, Format Table,
	    File  > "output.txt", Color "Ivory",
	    SendToServer "Output/Lift"];

      // Uncomment these lines to have more color maps
      //Print[phi, OnElementsOf Vol_rho, File "phi.pos"];
      //Print[phiDisc, OnElementsOf Vol_rho, File "phiDisc.pos"];
      //Print[phiCont, OnElementsOf Vol_rho, File "phiCont.pos"];
      //Print[pressure, OnElementsOf Vol_rho, File "p.pos"];

      Print[ velocity, OnElementsOf Vol_rho, File "velocity.pos"];
      Echo[ Str["l=PostProcessing.NbViews-1;",
		"View[l].VectorType = 1;",
		"View[l].LineWidth = 2;",
		"View[l].ArrowSizeMax = 100;",
		"View[l].CenterGlyphs = 1;"],
	    File "tmp.geo", LastTimeStepOnly] ;

      // Stagnation points are points where Norm[{d phi}] is zero
      // This is better seen in log scale.
      Print[normVelocity, OnElementsOf Vol_rho, File "p.pos"];
      Echo[Str["l=PostProcessing.NbViews-1;",
	       "View[l].Name = 'stagnation points';",
	       "View[l].ScaleType = 2; // log scale"],
	   File "tmp.geo"] ;

      If( Flag_Object == 0 )
	Print[phi, OnElementsOf Vol_rho, File "phi.pos"];
      Else
	// Show the isovalue "phi=$PhiTrailing"
	// which is perpendicular to the airfoil
	// iff the Kutta condition is fulfilled
	Print[phi, OnElementsOf Vol_rho, File "KJ.pos"];
        Print[{$phiTrailing, 1.001*$phiTrailing}, Format
	      Str["l=PostProcessing.NbViews-1;",
		  "View[l].Name = 'isovalue phiTrailing';",
		  "View[l].IntervalsType = 3;",
		  "Mesh.SurfaceEdges = 0; // hide mesh",
		  "View[l].RangeType = 2; // custom range",
		  "View[l].CustomMin = %g;",
		  "View[l].CustomMax = %g;"],
	      File "tmp.geo"] ;
      EndIf
    }
  }
}

PostOperation{ // for the Airfoil model
  { Name Trailing; NameOfPostProcessing PotentialFlow;
    Operation{
       Print[Circ, OnRegion Sur_Cut, File > "KJiter.txt", Format Table,
     	     StoreInVariable $circ,
     	     SendToServer "Output/Circ" ];
       Print[Dmdt, OnRegion Sur_Cut, File > "KJiter.txt", Format Table,
     	     StoreInVariable $dmdt,
     	     SendToServer "Output/Dmdt" ];
       // P_edge = {1.0001,0,0}
	Print[phi, OnPoint {1.0001,0,0}, File > "KJiter.txt", Color "Ivory",
	      StoreInVariable $phiTrailing,
	      Format ValueOnly, SendToServer "Output/PhiTrailing"];

        Print[Angle, OnPoint{1.0001,0,0}, File > "KJiter.txt", Color "Ivory",
	      StoreInVariable $argV,
	      Format ValueOnly, SendToServer "Output/argVTrailing"];
    }
  }
}

DefineConstant[
  R_ = {"PotentialFlow", Name "GetDP/1ResolutionChoices", Visible 0},
  C_ = {"-solve -pos", Name "GetDP/9ComputeCommand", Visible 0},
  P_ = {"PotentialFlow", Name "GetDP/2PostOperationChoices", Visible 0}
];
