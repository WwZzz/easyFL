## Simulation with Client-State Machine
We construct a client-state machine to simulate arbitrary system heterogeneiry. In 
this state machine, a client's state will change as time goes by or some particular 
actions were taken. For example, a client will be available with a probability at each 
moment, and clients will be in state 'working' after they were selected if not dropping out.
The transfer rules across states are described in the figure below

![client_state_machine](../../img/overview_flgo_state.png)
We provide simple APIs for users to customize the system heterogeneity for simulation. Please see 
Tutorial 5.1 for details.
