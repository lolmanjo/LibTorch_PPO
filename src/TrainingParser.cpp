#include "TrainingParser.h"

#include <fstream>
#include <string>
#include <JSON/json.hpp>

#include "TrainingController.h"

using namespace PLANS;
using json = nlohmann::json;

const std::string TrainingParser::STD_CHECKPOINT_DIRECTORY_NAME = "checkpoints";
const double TrainingParser::STD_LEARNING_RATE = 0.0001;
const uint32_t TrainingParser::STD_POLICY_STEP_LENGTH = 5;
const uint32_t TrainingParser::STD_TRAINING_STEP_LENGTH = 120;
const uint32_t TrainingParser::STD_MAX_EPISODE_LENGTH = 30000;
const uint32_t TrainingParser::STD_PPO_EPOCHS = 4;

void TrainingParser::parseConfigFile(const std::string& configFileName, TrainingParameters* parameters) {
	// Read file. 
	std::ifstream f(configFileName);
	json data = json::parse(f);
	// Get parameters object. 
	json params = data["parameters"];
	// Get parameter values. 
	if(params.contains("checkpointDirectoryName")) {
		parameters->checkpointDirectoryName = params["checkpointDirectoryName"];
	} else {
		parameters->checkpointDirectoryName = STD_CHECKPOINT_DIRECTORY_NAME;
	}
	if(params.contains("modelNameLoad")) {
		parameters->modelNameLoad = params["modelNameLoad"];
	} else {
		parameters->modelNameLoad = "";	// No loading. 
	}
	if(params.contains("modelNameSave")) {
		parameters->modelNameSave = params["modelNameSave"];
	} else {
		parameters->modelNameSave = "";	// No saving. 
	}
	if(params.contains("learningRate")) {
		parameters->learningRate = params["learningRate"];
	} else {
		parameters->learningRate = STD_LEARNING_RATE;
	}
	if(params.contains("policyStepLength")) {
		parameters->policyStepLength = params["policyStepLength"];
	} else {
		parameters->policyStepLength = STD_POLICY_STEP_LENGTH;
	}
	if(params.contains("trainingStepLength")) {
		parameters->trainingStepLength = params["trainingStepLength"];
	} else {
		parameters->trainingStepLength = STD_TRAINING_STEP_LENGTH;
	}
	if(params.contains("maxEpisodeLength")) {
		parameters->maxEpisodeLength = params["maxEpisodeLength"];
	} else {
		parameters->maxEpisodeLength = STD_MAX_EPISODE_LENGTH;
	}
	//if(params.contains("episodesPerAgentSwap")) {
	//	parameters->episodesPerAgentSwap = Maths::max<uint32_t>(params["episodesPerAgentSwap"], 1);
	//} else {
	//	parameters->episodesPerAgentSwap = UINT32_MAX;
	//}
	//if(params.contains("episodesPerCheckpoint")) {
	//	parameters->episodesPerCheckpoint = Maths::max<uint32_t>(params["episodesPerCheckpoint"], 1);
	//} else {
	//	parameters->episodesPerCheckpoint = UINT32_MAX;
	//}
	if(params.contains("maxEpisodes")) {
		parameters->maxEpisodes = params["maxEpisodes"];
	} else {
		parameters->maxEpisodes = 0;	// Infinite. 
	}
	if(params.contains("continueLogFile")) {
		parameters->continueLogFile = params["continueLogFile"];
	} else {
		parameters->continueLogFile = false;
	}
	if(params.contains("enableBackup")) {
		parameters->enableBackup = params["enableBackup"];
	} else {
		parameters->enableBackup = false;
	}
	if(params.contains("ppo_gamma")) {
		parameters->ppo_gamma = params["ppo_gamma"];
	} else {
		parameters->ppo_gamma = 0.99;
	}
	if(params.contains("ppo_lambda")) {
		parameters->ppo_lambda = params["ppo_lambda"];
	} else {
		parameters->ppo_lambda = 0.95;
	}
	if(params.contains("ppo_beta")) {
		parameters->ppo_beta = params["ppo_beta"];
	} else {
		parameters->ppo_beta = 0.001;
	}
	if(params.contains("ppo_epochs")) {
		parameters->ppo_epochs = params["ppo_epochs"];
		if(parameters->trainingStepLength % parameters->ppo_epochs != 0) {
			TrainingController::consoleOut("TrainingParser::parseConfigFile: trainingStepLength (" + std::to_string(parameters->trainingStepLength) + ") is not a multiples of ppo_epochs (" + std::to_string(parameters->ppo_epochs) + ").");
			abort();
		}
	} else {
		parameters->ppo_epochs = STD_PPO_EPOCHS;
	}
	parameters->ppo_miniBatchSize = parameters->trainingStepLength / parameters->ppo_epochs;
}
