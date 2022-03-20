from oxford_pet import SimpleOxfordPetDataset
import segmentation_models_pytorch as smp
from catalyst import utils
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Setting up GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "" - CPU, "0" - 1 GPU, "0,1" - MultiGPU
SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

# Datasets
root = "."
SimpleOxfordPetDataset.download(root)

# init train, val, test sets
train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")
test_dataset = SimpleOxfordPetDataset(root, "test")

# It is a good practice to check datasets don`t intersects with each other
assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

# lets look at some samples
sample = train_dataset[0]
plt.subplot(1, 2, 1)
plt.imshow(sample["image"].transpose(1, 2, 0))  # for visualization we have to transpose back to HWC
plt.subplot(1, 2, 2)
plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
# plt.show()

sample = valid_dataset[0]
plt.subplot(1, 2, 1)
plt.imshow(sample["image"].transpose(1, 2, 0))  # for visualization we have to transpose back to HWC
plt.subplot(1, 2, 2)
plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
# plt.show()

sample = test_dataset[0]
plt.subplot(1, 2, 1)
plt.imshow(sample["image"].transpose(1, 2, 0))  # for visualization we have to transpose back to HWC
plt.subplot(1, 2, 2)
plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
# plt.show()

# Create segmentation model
model = smp.Unet("resnet34", encoder_weights="imagenet", classes=1, activation=None)

batch = next(iter(test_dataloader))
with torch.no_grad():
    model.eval()
    logits = model(batch["image"].float())

pr_masks = logits.sigmoid()

for image, gt_mask, pr_mask in zip(batch["image"], batch["mask"], pr_masks):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pr_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
    plt.title("Prediction")
    plt.axis("off")
    plt.show()
