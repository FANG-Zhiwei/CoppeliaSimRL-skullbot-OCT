# test the action space (which is based on guassian distribution)

the value constantly go beyond the boundary, maybe using _clip

# wrap the observation and policy network better

look for visual only RL and Cnnpolicy wrap

# define reward function

define directional function

define stay penalty maybe

# image servoing w/o joint state

Sub modules: target image, edge detection. reward -- code pass

abnormal convergence in very few steps after running. Can fall in a desired path in some condition but mostly fall into unwanted condition.

actually not converge, but the env would be reset after some steps(around 100). The the coppeliaSim shows **abourt execution.** After a quite long time for reset, then the action goes to non-random.

**question**

* add reward and more state in obs?
* modify reward

* converge problem (sychronyze mode refer to original cart pole RL?)
* check action randomnizarion?

step time: usually 0.01x, no longer than 0.025