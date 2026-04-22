SimState state;
state.allocate(hm.size);
simStateUpload(state, hm);

// simulation loop
for (int step = 0; step < nSteps; ++step) {
    launchWaterIncrement(state);   state.T1.swap();
    launchFluxUpdate(state);       state.T2.swap();
    launchWaterAndVelocity(state); state.T1.swap(); state.T3.swap();
    launchErosion(state);          state.T1.swap();
    launchAdvection(state);        state.T1.swap();
    launchEvaporation(state);      state.T1.swap();
}

// whenever you need the CPU-side heightmap back
simStateReadback(state, hm);
state.release();





# random things
- One unified kernel wouldn't work because we need to read data from neighbors that may not be in the same block.
    - __syncthreads() only synchronizes threads within a block
    - I will start from 7 different kernels and then unified the ones that are compatible (3&4, 5&6)