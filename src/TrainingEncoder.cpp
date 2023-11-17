#include "TrainingEncoder.h"

#include "Environment.h"

using namespace PLANS;

StateData* TrainingEncoder::buildInputTensor(AGENT_ID agentID, Environment* environment) {
    // Put all data into a vector. Put this into a tensor with the dimensions { TrainingController::LSTM_SEQUENCE_LENGTH, TrainingController::LSTM_BATCH_SIZE, TrainingController::LSTM_INPUT_SIZE } ("inputTensor"). 
    // The final tensor is located on the correct device (CPU / CUDA). 
    // The final tensor is packed into a "StateData" object, along with the IDs of the encoded spaceships. 

    // Prepare state data. 
    StateData* stateData = new StateData(TrainingController::getStepsInThisEpisode());

    // Prepare data vector. 
    std::vector<float> data = std::vector<float>(LSTM_INPUT_SIZE);

    environment->getInputData(agentID, data);

#ifdef _DEBUG
    // Check if any value is NaN. 
    for(uint32_t i = 0; i < LSTM_INPUT_SIZE; i++) {
        if(std::isnan(data[i])) {
            TrainingController::consoleOut("TrainingEncoder::buildInputTensor: Value at index " + std::to_string(i) + " is NaN.");
            abort();
        }
    }
#endif

    torch::Tensor dataTensor = torch::from_blob(data.data(), LSTM_INPUT_SIZE, TrainingController::getTensorOptionsCPU());
    //dataTensor.size(0);

    //double d_0 = dataTensor[LSTM_INPUT_SIZE - 1].item().toDouble();

    // Softmax input data. 
    dataTensor = dataTensor.softmax(0);

    //double d_1 = dataTensor[LSTM_INPUT_SIZE - 1].item().toDouble();

    //// Build tensor. 
    //stateData->inputTensor = torch::empty({ TrainingController::LSTM_SEQUENCE_LENGTH, TrainingController::LSTM_INPUT_SIZE }, TrainingController::getTensorOptionsCPU());
    //for(int64_t i = 0; i < TrainingController::LSTM_SEQUENCE_LENGTH; i++) {
    //    //torch::Tensor tensorC = torch::from_blob(data.data(), TrainingController::LSTM_INPUT_SIZE, TrainingController::getTensorOptionsCPU());
    //    //batchTensor.index_put_({ k }, dataTensor);
    //    stateData->inputTensor[i] = dataTensor;
    //}
    stateData->inputTensor = torch::empty({ 1, LSTM_INPUT_SIZE }, TrainingController::getTensorOptionsCPU());
    stateData->inputTensor[0] = dataTensor;
    //std::cout << inputTensor << std::endl;

#ifdef USE_CUDA
    // Copy tensor to GPU. 
    stateData->inputTensorDevice = stateData->inputTensor.to(torch::kCUDA);
#else
    stateData->inputTensorDevice = stateData->inputTensor.to(torch::kCPU);
#endif
    
    return stateData;
}

float TrainingEncoder::decodeAction(AGENT_ID agentID, const torch::Tensor& actorOutput) {
    // Check that actorOutput has the correct dimensions of 1. 
    if(actorOutput.dim() != 1) {
        TrainingController::consoleOut("TrainingEncoder::decodeAction: \"actorOutput\" has invalid dimension: " + std::to_string(actorOutput.dim()));
        abort();
    }

    float output = actorOutput[0].item<float>();
    //std::cout << std::to_string(TrainingController::getStepsInThisEpisode()) + ": " + std::to_string(output) << std::endl;

    //for(int64_t i = 0; i < TrainingController::LSTM_OUTPUT_SIZE; i++) {
    //    std::cout << std::to_string(i) + ": " + std::to_string(actorOutput[i].item<float>()) << std::endl;
    //}

    return output;
}
