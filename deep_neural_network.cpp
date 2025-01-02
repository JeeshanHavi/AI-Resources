#include <torch/torch.h>
#include <iostream>
#include <vector>

// Define the model
struct DeepNeuralNet : torch::nn::Module {
    torch::nn::Linear l1, l2, l3, l4, l5;
    torch::nn::BatchNorm1d bn1, bn2, bn3, bn4;
    torch::nn::ReLU relu;
    torch::nn::Dropout dropout;

    DeepNeuralNet(int64_t input_size, int64_t hidden_size, int64_t num_classes)
        : l1(torch::nn::LinearOptions(input_size, hidden_size)),
          l2(torch::nn::LinearOptions(hidden_size, hidden_size)),
          l3(torch::nn::LinearOptions(hidden_size, hidden_size)),
          l4(torch::nn::LinearOptions(hidden_size, hidden_size)),
          l5(torch::nn::LinearOptions(hidden_size, num_classes)),
          bn1(hidden_size), bn2(hidden_size), bn3(hidden_size), bn4(hidden_size),
          relu(torch::nn::ReLU()), dropout(torch::nn::Dropout(0.5)) {

        // Register modules (required for parameters)
        register_module("l1", l1);
        register_module("l2", l2);
        register_module("l3", l3);
        register_module("l4", l4);
        register_module("l5", l5);
        register_module("bn1", bn1);
        register_module("bn2", bn2);
        register_module("bn3", bn3);
        register_module("bn4", bn4);
        register_module("relu", relu);
        register_module("dropout", dropout);

        // Weight initialization using Kaiming Normal and Xavier Normal
        torch::nn::init::kaiming_normal_(l1->weight, /*nonlinearity=*/"relu");
        torch::nn::init::kaiming_normal_(l2->weight, /*nonlinearity=*/"relu");
        torch::nn::init::kaiming_normal_(l3->weight, /*nonlinearity=*/"relu");
        torch::nn::init::kaiming_normal_(l4->weight, /*nonlinearity=*/"relu");
        torch::nn::init::xavier_normal_(l5->weight);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = l1(x);
        x = bn1(x);
        x = relu(x);
        x = dropout(x);

        x = l2(x);
        x = bn2(x);
        x = relu(x);
        x = dropout(x);

        x = l3(x);
        x = bn3(x);
        x = relu(x);
        x = dropout(x);

        x = l4(x);
        x = bn4(x);
        x = relu(x);
        x = dropout(x);

        x = l5(x);
        return x;  // Output raw logits
    }
};

int main() {
    // Hyperparameters
    int64_t input_size = 784; // For MNIST
    int64_t hidden_size = 128;
    int64_t num_classes = 10;
    int64_t batch_size = 128;

    // Setup the model
    DeepNeuralNet model(input_size, hidden_size, num_classes);

    // Load a pre-trained model (for inference)
    try {
        torch::load(model, "model.pt");  // Load the model from a file
        std::cout << "Model loaded successfully!" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return -1;
    }

    // Create a dummy tensor for the forward pass (simulate the MNIST input)
    torch::Tensor input = torch::randn({batch_size, input_size});  // Random input tensor (batch of 128)

    // Perform inference (forward pass)
    model.eval();  // Set the model to evaluation mode
    torch::Tensor output = model.forward(input);

    // Print output shape (Logits for each class)
    std::cout << "Output shape: " << output.sizes() << std::endl;

    // Optionally, print the class predictions (argmax across the class dimension)
    auto predictions = output.argmax(1);  // Get predicted class indices (argmax over output logits)
    std::cout << "Predicted classes: " << predictions << std::endl;

    return 0;
}
