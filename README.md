# StiffODERungeKutta

This repo was meant to be used for the lab work at BSU.

This code is supposed to solve two stiff problems: Robertson's stiff system and Brunner's stiff system.

Both problems were supposed to be solved using the 3rd order explicit Runke-Kutta method and the 2 stage RadauIIA method (order 3).

However, due to an unreliable adaptive time stepping method and/or inefficient nonlinear system solving method, the RadauIIA method fails to solve Robertson's problem.

Note, that in a previous build, when the adaptive time stepping was not implemented, RadauIIA solved Robertson's problem just fine, although took long to do so.
