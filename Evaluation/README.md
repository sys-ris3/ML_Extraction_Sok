# Installing PCM

PCM provides several command-line utilities for real-time monitoring, including energy consumption, which we have utilized to evaluate power consumption.

1. Steps to Install [PCM](https://github.com/intel/pcm)
  - ```git clone --recursive https://github.com/opcm/pcm.git```
  - ```cd pcm```
  - ```mkdir build```
  - ```cd build```
  - ```cmake ..```
  - ```cmake --build . --parallel```
  
2. Executing PCM tools under Linux
 - ``` sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid' ```
 - ```export PCM_NO_MSR=1```
 - ```export PCM_KEEP_NMI_WATCHDOG=1```

3.  Run PCM in a parallel terminal to monitor the energy comsumption
 - ``` bin/pcm```
 
 Now you can monitor the energy consumption while running the attack/defense. The following shows an example of energy consumption, before and after running an ML extraction defense project.

 ![Alt text](PCM_evaluation.png?raw=true&sanitize=true "Optional Title")

**Note:  Refer to orginal PCM doumentation for me details [DOCUMENTATION](https://github.com/intel/pcm)**

