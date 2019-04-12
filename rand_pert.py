import numpy as np
import torch
import cv2
from torchvision import datasets, transforms, models
from matplotlib import pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm

CLASS_NAMES = {0: 'coast',
               1: 'forest',
               2: 'highway',
               3: 'insidecity',
               4: 'mountain',
               5: 'opencountry',
               6: 'street',
               7: 'tallbuilding'}


def random_vector_surface(shape=(256, 256, 3)):
    # generates a random vector on the surface of hypersphere
    mat = np.random.normal(size=shape)
    norm = np.linalg.norm(mat)
    return mat/norm


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


# Random attack code
def random_attack(image, epsilon):
    pert = epsilon * np.sign(random_vector_surface())
    perturbed_image = image.clone()

    # add perturbation to image
    perturbed_image.data = image.data + torch.from_numpy(pert.transpose(2, 0, 1)).float().unsqueeze(0)

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

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Call Attack
        perturbed_data = random_attack(data, epsilon)

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

# epsilons = [0, .05, .1, .15, .2, .25, .3]
epsilons = [0, .1, .2, .3, .4, .5]

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
plt.xticks(np.arange(0, .6, step=0.1))
plt.title("Perturbation aléatoire")
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