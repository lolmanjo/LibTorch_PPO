#pragma once

#include "TrainingParameters.h"

namespace PLANS {

	/*
	*	Parses the configuration file. Returns the training parameters. 
	*/
	class TrainingParser {
		private:
		protected:
		public:
			static const std::string STD_CHECKPOINT_DIRECTORY_NAME;
			static const double STD_LEARNING_RATE;
			static const uint32_t STD_POLICY_STEP_LENGTH;	// How often the agents take action. 
			static const uint32_t STD_TRAINING_STEP_LENGTH;	// How many actions the agents have to take until the optimizer is executed. 
			static const uint32_t STD_MAX_EPISODE_LENGTH;	// Maximum lock steps until episode is terminated. 
			static const uint32_t STD_PPO_EPOCHS;			// How often the model is being optimized (via PPO optimization) per PPO call. 

			static void parseConfigFile(const std::string& configFileName, TrainingParameters* parameters);
	};

}
