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


def deprocess_img(perturbed, original):
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

        # Call Attack
        perturbed_data = random_attack(data, epsilon)

        # Re-classify the perturbed image
        output = model(perturbed_data)


        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        adv_ex = deprocess_img(perturbed_data, data)
        adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Return modified image example
    return adv_examples

device = 'cpu'

epsilons = [0, .05, .1, .2, .3, .4, .5, .6]

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

image_directory = "./images"
dataset = datasets.ImageFolder(image_directory, data_transforms)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

pretrained_model = "./models/finetune_8classes64_100norand.pth"
resnet = torch.load(pretrained_model)
resnet.eval()

accuracies = []
examples = []

for eps in epsilons:
    print('Generating for epsilon={}'.format(eps))
    ex = test(resnet, device, loader, eps)
    examples.append(ex)

# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(15, 40))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0])*2,cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=60)
        orig, adv, ex = examples[i][j]
        # plt.title("{} -> {}".format(CLASS_NAMES[orig][0], CLASS_NAMES[adv][0]))
        plt.imshow(ex)
plt.tight_layout()
plt.show()