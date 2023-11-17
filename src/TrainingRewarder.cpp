#include "TrainingRewarder.h"

#include "Environment.h"

using namespace PLANS;

double TrainingRewarder::calculateReward(AGENT_ID agentID, bool didTakeAction, Environment* enviroment) {
	double reward = 0.0;
	
	//// Reward if action taken. 
	//if(didTakeAction) {
	//	reward += TrainingRewarder::REWARD_ACTION_TAKEN;
	//}

	reward += enviroment->rewardAgent(agentID);
	
    return reward;
}

const double TrainingRewarder::REWARD_ACTION_TAKEN = 10.0;
