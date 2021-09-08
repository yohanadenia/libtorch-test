/*
 * Yohana Denia Gaotama
 * Tes Calon Staff AI Researcher
 * PT Delameta Bilano
 * Jakarta Timur
 * ============================================================================
 */

include <model.h>
include <opencv2/core/core.hpp>
include <opencv2/highgui.hpp>
include <opencv2/opencv.hpp>
include <sys/stat.h>
include <torch/cuda.h>
include <torch/optim.h>
include <unistd.h>
include <vector>


train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=50, shuffle=False)
testset = torch.utils.data.DataLoader(test, batch_size=50, shuffle=True)

struct LeNet : torch::nn::Module {
  LeNet() {
    // Construct and register two Linear submodules.
    LeNet::conv1 = register_module("conv1", torch::nn::Conv2d(1, 6, 5));
    LeNet::conv2 = register_module("conv2", torch::nn::Conv2d(6, 16, 5));
    LeNet::fc1 = register_module("fc1", torch::nn::Linear(16 * 4 * 4, 120));
    LeNet::fc2 = register_module("fc2", torch::nn::Linear(120, 84));
    LeNet::fc3 = register_module("fc3", torch::nn::Linear(84, 10));

    LeNet::pool = register_module("pool", torch::nn::MaxPool2d(2));
    LeNet::relu = register_module("relu", torch::nn::ReLU());
  }

  torch::Tensor forward(torch::Tensor x) {
    x = LeNet::conv1->forward(x);
    x = LeNet::relu->forward(x);
    x = LeNet::pool->forward(x);

    x = LeNet::conv2->forward(x);
    x = LeNet::relu->forward(x);
    x = LeNet::pool->forward(x);

    x = x.view({x.size(0), 16 * 4 * 4});
    x = LeNet::fc1->forward(x);
    x = LeNet::relu->forward(x);
    x = LeNet::fc2->forward(x);
    x = LeNet::relu->forward(x);
    x = LeNet::fc3->forward(x);
    x = torch::log_softmax(x, -1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::MaxPool2d pool{nullptr};
  torch::nn::ReLU relu{nullptr};
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

double train(size_t epoch,
             std::shared_ptr<LeNet> model,
             torch::Device device,
             DataLoader &data_loader,
             torch::optim::Optimizer &optimizer,
             size_t dataset_size) {
  // set train mode
  model->train();
  size_t batch_index = 1;
  // Iterate data loader to yield batches from the dataset
  for (auto &batch : data_loader) {
    auto images = batch.data.to(device);
    auto targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model->forward(images);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));

    // Compute gradients
    loss.backward();
    // Update the parameters
    optimizer.step();

    if (++batch_index % 10 == 0) {
      std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
                  epoch,
                  batch_index * batch.data.size(0),
                  dataset_size,
                  loss.template item<float>());
    }
  }
}

template<typename DataLoader>
std::vector<double> evaluate(std::shared_ptr<LeNet> model,
                             torch::Device device,
                             DataLoader &data_loader,
                             size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model->eval();

  // define value.
  double loss = 0.;
  double accuracy;
  size_t correct = 0;
  std::vector<double> result;

  for (const auto &batch : data_loader) {
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);
    auto output = model->forward(data);
    loss += torch::nll_loss(output, targets, {}, torch::Reduction::Sum).template item<double>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }
  loss /= dataset_size;
  accuracy = static_cast<double>(correct) / dataset_size;

  std::printf("\nTest set: Average loss: %.4f | Accuracy: %.3f\n", loss, accuracy);

  result.push_back(loss);
  result.push_back(accuracy);

  return result;
}
