import torch
import argparse


def evaluate(data_path, model_checkpoint):

    print("Evaluating until hitting the ceiling")

    # print(model_checkpoint)
    test_set = torch.load(data_path)
    model = torch.load(model_checkpoint)
    with torch.no_grad():
        model.eval()

    criterion = torch.nn.NLLLoss()
    test_losses = []
    test_loss = 0
    images = test_set["images"]
    labels = test_set["labels"]
    # for images, labels in test_set:
    log_ps = model(images.float())
    loss = criterion(log_ps, labels)
    test_loss += loss.item()
    test_losses.append(loss.item())

    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f"Accuracy: {accuracy.item()*100}%")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_checkpoint", type=str, help="model_checkpoint")
    parser.add_argument("data_path", type=str, help="location of test data")
    res = parser.parse_args()

    evaluate(data_path=res.data_path, model_checkpoint=res.model_checkpoint)
