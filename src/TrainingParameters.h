#pragma once

#include <string>
#include <stdint.h>

namespace PLANS {

	//############################ TrainingParams ############################

	struct TrainingParameters {
		std::string checkpointDirectoryName;
		std::string modelNameLoad;
		std::string modelNameSave;
		double learningRate;
		uint32_t policyStepLength;		// How often the agents take action. 
		uint32_t trainingStepLength;	// After how many agent actions the optimizer is executed. 
		uint32_t maxEpisodeLength;		// Maximum lock steps until episode is terminated. 
		//uint32_t episodesPerAgentSwap;	// After how many episodes the agents will be swapped. -1 for no swaps. 
		//uint32_t episodesPerCheckpoint;	// After how many episodes a checkpoint is being created. -1 for no checkpoints (not recommended). 
		uint32_t maxEpisodes;			// After how many episodes the training should be terminated. 0 for infinite. 
		bool continueLogFile;			// Wether a log file with matching name should be continued or a new log file should be created. 
		bool enableBackup;				// Whether the backup helper should upload files to the backup server. 
		double ppo_gamma;
		double ppo_lambda;
		double ppo_beta;
		uint32_t ppo_epochs;
		uint32_t ppo_miniBatchSize;		// Calculated automatically (trainingStepLength / ppo_epochs). 
	};

}
