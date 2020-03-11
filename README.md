# drake-manip-station

This repo is meant to replicate the segmentaton fault error

To replicate the segmentation fault error:

```python

python3 iiwa_drake.py
```

The illustration can be found at the very bottom of the file, 
```python
if __name__ == '__main__':
  sim = ManipulationStationSimulator(visualizer='drake')
  sim.init_simulator()
  print('works fine before resetting the context')
  sim.sim_duration(2.)
  sim.incremental_move_in_world(increment=0.3, direction='y')
  print('now reset the context')
  sim.simulator.reset_context(sim.simulator_init_context)
  print('works fine for simple AdvanceTo with no commands')
  sim.sim_duration(2.0)
  print('segmentation fault if arm is commanded to move again')
  sim.incremental_move_in_world(direction='y')
```

note that 
```python
simulator.AdvanceTo(stop_time)
```
with NO command works fine after resetting the context

However, once the arm is commanded to move, the segmentation fault will appear.
