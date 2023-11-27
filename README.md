# LibTorch_PPO
A minimal training program utilizing LibTorch for an implementation of the PPO algorithm. 

## Build it
To build the project, you need to download the official LibTorch (C++) libraries from https://pytorch.org/. In testing the CUDA platform 11.8 is used. 
In the current setup they are located at "C:/Program Files/libtorch/" by default, but you can change the path in the VS project. 
Make sure that the include paths are correct and the project should build. 

## Configure it
A variety of training parameters are configurable via the "trainingConfig.json". This eliminates the need to rebuild the executable. 
Some parameters are without functionality, due to cuts to the complexity of the program. The active parameters are: 
 - "learningRate": The learning rate used by the optimizer. 
 - "policyStepLength": After how many environment steps the agents take action. 
 - "trainingStepLength": After how many actions the agents are optimized.
 - "maxEpisodeLength": After how many environment steps the current episode is terminated.
 - "maxEpisodes": After how many episodes the training program is terminated.
 - "ppo_gamma": PPO gamma parameter.
 - "ppo_lambda": PPO lambda parameter.
 - "ppo_beta": PPO beta parameter.
 - "ppo_epochs": PPO epochs parameter.

##
If you encounter problems or have questions feel free to create an issue. 

This program was developed as part of a master thesis in the field of machine learning. 
