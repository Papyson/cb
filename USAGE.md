#CityBUilder Env

import citybuilder_env as cbe

env = cbe.make_env(seed=1234)
obs, mask, info = env.reset()

done = False
while not done:
    feasible = mask.nonzero()[0]
    action = int(feasible[0])       # your policy here
    step = env.step(action)
    obs, mask, done = step.observation, step.action_mask, step.done

summary = env.episode_summary()