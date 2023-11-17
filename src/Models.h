#pragma once

#include "TrainingConsts.h"

namespace PLANS {

    //############################ ActorCriticImpl ############################
    
    // Network model for Proximal Policy Optimization on Incy Wincy.
    struct ActorCriticImpl : public torch::nn::Module {
        // Actor.
        torch::nn::Linear a_lin1_, a_lin2_, a_lin3_;
        torch::Tensor mu_;
        torch::Tensor log_std_;
    
        // Critic.
        torch::nn::Linear c_lin1_, c_lin2_, c_lin3_, c_val_;
    
        ActorCriticImpl(double std);

        // Returned tuple: [0]: Actor output of size 1, [1]: Critic output of size 1. 
        std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& inputTensor, bool b);
        void normal(double mu, double std);
        torch::Tensor entropy();
        torch::Tensor logProb(const torch::Tensor& action);
        void toDevice(torch::DeviceType device);
    };
    
    TORCH_MODULE(ActorCritic);

    //############################ ActorCriticOpenAIFiveImpl ############################

    // A model based on the structure of OpenAI Five. A single LSTM whichs outputs are passed into linear projections to produce the action and value outputs. 
    struct ActorCriticOpenAIFiveImpl : torch::nn::Module {
        double std;

        torch::nn::LSTM lstm;
        torch::nn::Linear actor_0, actor_1;
        torch::nn::Linear critic_0, critic_1;

        std::tuple<torch::Tensor, torch::Tensor> hx_options;    // Hidden states of LSTM. 
        torch::Tensor log_std_;
        torch::Tensor lastActorOutput;

        ActorCriticOpenAIFiveImpl(double std);

        // Returned tuple: [0]: Actor output of size LSTM_OUTPUT_SIZE, [1]: Critic output of size 1. 
        std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& inputTensor, bool updateHxOptions);
        void normal(double mu, double std);
        torch::Tensor entropy() const;
        torch::Tensor logProb(const torch::Tensor& action) const;
        void toDevice(torch::DeviceType device);
        void reset();
    };
    TORCH_MODULE(ActorCriticOpenAIFive);

}
