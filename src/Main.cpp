#include "TrainingConsts.h"
#include "Environment.h"
#include "TrainingParser.h"
#include "TrainingController.h"
#include "TrainingEncoder.h"

using namespace PLANS;

int main(int argc, const char** argv) {

	at::globalContext().setAllowTF32CuDNN(true);
	at::globalContext().setDeterministicCuDNN(true);

	// Parse training parameters. 
	TrainingParameters* parameters = new TrainingParameters();
	TrainingParser::parseConfigFile("./trainingConfig.json", parameters);

	// Init environment. 
	//Environment* enviroment = new EnvironmentBinary();
	Environment* enviroment = new EnvironmentFloat();
	
	// Init training controller. 
	if(!TrainingController::init(parameters, enviroment)) {
		return 1;	// Somehow failed. 
	}

	// Prepare first scenario. 
	TrainingController::onNextScenarioRequired(true);

	// Train until telled to stop. 
	bool stop = false;
	bool actionTaken;
	float action = 0.0F;
	std::vector<torch::Tensor> output;
	while(!stop) {
		// Update environment. 
		enviroment->update();

		// Update agents. 
		for(uint32_t agentID = 0; agentID < NUM_OF_AGENTS; agentID++) {
			// Get action to take. 
			output.clear();
			actionTaken = TrainingController::onActionRequired(agentID, output);

			if(actionTaken) {
				action = TrainingEncoder::decodeAction(agentID, output[0]);

				// Execute action. 
				enviroment->onAction(agentID, action);

				// Reward agent. 
				TrainingController::onAgentExecuted(agentID);
			}
		}

		// Finish tick. 
		if(TrainingController::onGameTickPassed()) {
			if(TrainingController::onNextScenarioRequired(false)) {
				stop = true;
			}
		}
	}

	TrainingController::cleanUp();
}
