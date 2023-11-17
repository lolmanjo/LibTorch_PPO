#include "Models.h"

#include "TrainingController.h"

using namespace PLANS;

//############################ ActorCriticImpl ############################

ActorCriticImpl::ActorCriticImpl(double std)
    : // Actor.
    a_lin1_(torch::nn::Linear(LSTM_INPUT_SIZE, 16)),
    a_lin2_(torch::nn::Linear(16, 32)),
    a_lin3_(torch::nn::Linear(32, LSTM_OUTPUT_SIZE)),
    mu_(torch::full(LSTM_OUTPUT_SIZE, 0.)),
    log_std_(torch::full(LSTM_OUTPUT_SIZE, std)),

    // Critic
    c_lin1_(torch::nn::Linear(LSTM_INPUT_SIZE, 16)),
    c_lin2_(torch::nn::Linear(16, 32)),
    c_lin3_(torch::nn::Linear(32, LSTM_OUTPUT_SIZE)),
    c_val_(torch::nn::Linear(LSTM_OUTPUT_SIZE, 1)) {
    // Register the modules.
    register_module("a_lin1", a_lin1_);
    register_module("a_lin2", a_lin2_);
    register_module("a_lin3", a_lin3_);
    register_parameter("log_std", log_std_);

    register_module("c_lin1", c_lin1_);
    register_module("c_lin2", c_lin2_);
    register_module("c_lin3", c_lin3_);
    register_module("c_val", c_val_);
}

std::tuple<torch::Tensor, torch::Tensor> ActorCriticImpl::forward(const torch::Tensor& inputTensor, bool b) {

    // Actor.
    mu_ = torch::relu(a_lin1_->forward(inputTensor));
    mu_ = torch::relu(a_lin2_->forward(mu_));
    mu_ = torch::tanh(a_lin3_->forward(mu_));

    // Critic.
    torch::Tensor val = torch::relu(c_lin1_->forward(inputTensor));
    val = torch::relu(c_lin2_->forward(val));
    val = torch::tanh(c_lin3_->forward(val));
    val = c_val_->forward(val);

    torch::NoGradGuard no_grad;

    torch::Tensor action = at::normal(mu_, log_std_.exp().expand_as(mu_));
    return std::make_tuple(action, val);
}

void ActorCriticImpl::normal(double mu, double std) {
    torch::NoGradGuard no_grad;

    for(auto& p : this->parameters()) {
        p.normal_(mu, std);
    }
}

torch::Tensor ActorCriticImpl::entropy() {
    // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
    return 0.5 + 0.5 * log(2 * M_PI) + log_std_;
}

torch::Tensor ActorCriticImpl::logProb(const torch::Tensor& action) {
    // Logarithmic probability of taken action, given the current distribution.
    torch::Tensor var = (log_std_ + log_std_).exp();

    return -((action - mu_) * (action - mu_)) / (2 * var) - log_std_ - log(sqrt(2 * M_PI));
}

void ActorCriticImpl::toDevice(torch::DeviceType device) {
    to(device);
}

//############################ ActorCriticOpenAIFiveImpl ############################

ActorCriticOpenAIFiveImpl::ActorCriticOpenAIFiveImpl(double std) :
    torch::nn::Module(),
    std(std),
    lstm(nullptr),
    actor_0(torch::nn::Linear(LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE)),
    actor_1(torch::nn::Linear(LSTM_HIDDEN_SIZE, LSTM_OUTPUT_SIZE)),
    critic_0(torch::nn::Linear(LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE)),
    critic_1(torch::nn::Linear(LSTM_HIDDEN_SIZE, 1)) {
    // Create lstm options. 
    torch::nn::LSTMOptions lstmOptions =
        torch::nn::LSTMOptions(LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE)
        .proj_size(0)   // Same as hidden size. 
        .num_layers(LSTM_NUM_LAYERS)
        .bidirectional(false)
        .batch_first(true);
    // Create lstm model. 
    lstm = torch::nn::LSTM(lstmOptions);
    // Register modules. 
    register_module("lstm", lstm);
    register_module("actor_0", actor_0);
    register_module("actor_1", actor_1);
    register_module("critic_0", critic_0);
    register_module("critic_1", critic_1);

    // Build hx_options (filled with zeros is standard). 
    std::get<0>(hx_options) = torch::zeros({ LSTM_NUM_LAYERS, LSTM_BATCH_SIZE, LSTM_HIDDEN_SIZE }, TrainingController::getTensorOptions());
    std::get<1>(hx_options) = torch::zeros({ LSTM_NUM_LAYERS, LSTM_BATCH_SIZE, LSTM_HIDDEN_SIZE }, TrainingController::getTensorOptions());

    // Register hx_options. 
    register_parameter("hiddenStates", std::get<0>(hx_options));
    register_parameter("cellStates", std::get<1>(hx_options));

    // Create and register log_std_. 
    log_std_ = torch::full(LSTM_OUTPUT_SIZE, std, TrainingController::getTensorOptions());
    //register_parameter("log_std_", log_std_);

    lastActorOutput = torch::zeros(LSTM_OUTPUT_SIZE, TrainingController::getTensorOptions());
}

std::tuple<torch::Tensor, torch::Tensor> ActorCriticOpenAIFiveImpl::forward(const torch::Tensor& inputTensor, bool updateHxOptions) {

    // NOTE: inputTensor has size { 1, LSTM_INPUT_SIZE }. The LSTM requires an three-dimensional tensor, so inputTensor is being put into a temporary 3D tensor. 
    torch::Tensor lstmInput = torch::empty({ 1, inputTensor.size(0), inputTensor.size(1) }, TrainingController::getTensorOptions());
    lstmInput[0] = inputTensor;

    lstmInput.set_requires_grad(false);

    // Prepare the hidden states based on "updateHxOptions". 
    std::tuple<torch::Tensor, torch::Tensor> hidden;
    if(updateHxOptions) {
        hidden = hx_options;
    } else {
        // Detach to avoid an update of the hidden states (source: https://stackoverflow.com/questions/75842061/i-dont-think-there-is-an-inplace-operation-but-an-inplace-operation-error-occu). 
        hidden = std::make_tuple(std::get<0>(hx_options).detach(), std::get<1>(hx_options).detach());
        std::get<0>(hx_options).set_requires_grad(false);
        std::get<1>(hx_options).set_requires_grad(false);
    }

    // Pass input tensor and the prepared hidden states into LSTM. 
    std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> lstmOutput = lstm->forward(lstmInput, hidden);

    if(updateHxOptions) {
        // Save hx_options for next time. 
        hx_options = std::get<1>(lstmOutput);
    }

    // Extract actual lstm output (eliminate sequence and batch indices). 
    torch::Tensor in = std::get<0>(lstmOutput)[0];
    in.set_requires_grad(false);

    //// Debug output variables. 
    //in.dim();
    //double d_0 = in[0][0].item().toDouble();
    //double d_1 = in[0][1].item().toDouble();
    //double d_2 = in[0][2].item().toDouble();
    //double d_3 = in[0][3].item().toDouble();
    //double d_4 = in[0][4].item().toDouble();
    //double d_5 = in[0][5].item().toDouble();

    // Pass LSTM output into actor. 
    torch::Tensor tmp = actor_0->forward(in);
    lastActorOutput = torch::relu(actor_1->forward(tmp));

    // Pass LSTM output into critic. 
    torch::Tensor criticOutput = critic_0->forward(in);
    criticOutput = torch::relu(critic_1->forward(criticOutput));

    torch::NoGradGuard no_grad; // Disables change of the gradients of the linear layers. 

    log_std_ = torch::full(LSTM_OUTPUT_SIZE, std, TrainingController::getTensorOptions());

    torch::Tensor action = at::normal(lastActorOutput, log_std_.exp());

    return std::make_tuple(action, criticOutput);
}

void ActorCriticOpenAIFiveImpl::normal(double mu, double std) {
    torch::NoGradGuard no_grad;

    for(auto& p : this->parameters()) {
        p.normal_(mu, std);
    }
}

torch::Tensor ActorCriticOpenAIFiveImpl::entropy() const {
    // Differential entropy of normal distribution. For reference https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal
    return 0.5 + 0.5 * log(2 * M_PI) + log_std_;
}

torch::Tensor ActorCriticOpenAIFiveImpl::logProb(const torch::Tensor& action) const {
    // Logarithmic probability of taken action, given the current distribution.
    torch::Tensor var = (log_std_ + log_std_).exp();

    return -((action - lastActorOutput) * (action - lastActorOutput)) / (2 * var) - log_std_ - log(sqrt(2 * M_PI));
}

void ActorCriticOpenAIFiveImpl::toDevice(torch::DeviceType device) {
    to(device);
    lstm->to(device);
    actor_0->to(device);
    actor_1->to(device);
    critic_0->to(device);
    critic_1->to(device);
}

void ActorCriticOpenAIFiveImpl::reset() {
    std::get<0>(hx_options) = torch::zeros({ LSTM_NUM_LAYERS, LSTM_BATCH_SIZE, LSTM_HIDDEN_SIZE }, TrainingController::getTensorOptions());
    std::get<1>(hx_options) = torch::zeros({ LSTM_NUM_LAYERS, LSTM_BATCH_SIZE, LSTM_HIDDEN_SIZE }, TrainingController::getTensorOptions());
    lastActorOutput = torch::zeros(LSTM_OUTPUT_SIZE, TrainingController::getTensorOptions());
}
