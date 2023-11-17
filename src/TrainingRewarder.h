#pragma once

#include "TrainingConsts.h"

namespace PLANS {

	class Environment;

	class TrainingRewarder {
		public:
			static double calculateReward(AGENT_ID agentID, bool didTakeAction, Environment* enviroment);
		protected:
		private:
			static const double REWARD_ACTION_TAKEN;
	};

}
