import numpy as np
import torch
import cv2
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch import nn

CLASS_NAMES = {0: 'coast',
               1: 'forest',
               2: 'highway',
               3: 'insidecity',
               4: 'mountain',
               5: 'opencountry',
               6: 'street',
               7: 'tallbuilding'}


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def deprocess_img(perturbed):
    image = perturbed.data.cpu().numpy()[0]
    image = image.transpose(1, 2, 0)
    image = (image * std) + mean
    image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    # for data, target in tqdm(test_loader):
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = criterion(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = deprocess_img(perturbed_data)
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = deprocess_img(perturbed_data)
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

device = 'cpu'

# epsilons = [0, .0025, .005, .0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
epsilons = [0, .005, 0.01,  0.015, 0.02, 0.025, 0.03]
# epsilons = [0, .1, .2, .3, .4, .5]

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

image_directory = "./output/val"
dataset = datasets.ImageFolder(image_directory, data_transforms)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

pretrained_model = "./models/finetune_8classes64_100norand.pth"
resnet = torch.load(pretrained_model)
resnet.eval()

criterion = nn.CrossEntropyLoss()

accuracies = []
examples = []

for eps in epsilons:
    print('Attack with epsilon={}'.format(eps))
    acc, ex = test(resnet, device, loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(epsilons)
plt.title("Perturbation FGSM")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8, 10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, ex = examples[i][j]
        plt.title("{} -> {}".format(CLASS_NAMES[orig][0], CLASS_NAMES[adv][0]))
        plt.imshow(ex)
plt.tight_layout()
plt.show()