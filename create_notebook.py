import json
import os

def generate_notebook():
    cells = []

    # Cell 1: Description and Imports
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# KLA Hackathon: Super-Resolution and Denoising\n",
            "\n",
            "This notebook builds, trains, and runs inference for restoring high-resolution clean images from noisy low-resolution inputs. \n",
            "\n",
            "**Data Summary:**\n",
            "- **NoisyLR**: `(128,128)` Input Images with noise\n",
            "- **GT**: `(256,256)` Target Clean Images (Scale Factor: 2.0x)\n",
            "\n",
            "**Model Architecture:**\n",
            "A Residual Network based approach with `PixelShuffle` for upsampling to `(256, 256)`. This performs Joint Super-Resolution and Denoising."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import glob\n",
            "import numpy as np\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "import base64\n",
            "import io\n",
            "\n",
            "print(\"Torch Available:\", torch.__version__)\n",
            "print(\"MPS Available (Mac GPU):\", torch.backends.mps.is_available())\n",
            "device = torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")\n",
            "print(f\"Using device: {device}\")"
        ]
    })

    # Cell 2: Hyperparameters
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Hyperparameters\n",
            "BATCH_SIZE = 16\n",
            "EPOCHS = 50\n",
            "LEARNING_RATE = 1e-4\n",
            "TRAIN_GT_DIR = 'train/GT'\n",
            "TRAIN_LR_DIR = 'train/NoisyLR'\n",
            "TEST_LR_DIR = 'test/NoisyLR'\n",
            "WEIGHTS_PATH = 'best_model.pth'\n"
        ]
    })

    # Cell 3: Dataset Definition
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class KLADataset(Dataset):\n",
            "    def __init__(self, lr_dir, gt_dir=None):\n",
            "        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, \"*.npy\")))\n",
            "        self.gt_dir = gt_dir\n",
            "        self.gt_files = None\n",
            "        if gt_dir:\n",
            "            self.gt_files = sorted(glob.glob(os.path.join(gt_dir, \"*.npy\")))\n",
            "            assert len(self.lr_files) == len(self.gt_files), \"LR and GT counts mismatch!\"\n",
            "\n",
            "    def __len__(self):\n",
            "        return len(self.lr_files)\n",
            "\n",
            "    def __getitem__(self, idx):\n",
            "        lr_path = self.lr_files[idx]\n",
            "        lr_img = np.load(lr_path).astype(np.float32)\n",
            "        # Reshape to (C, H, W) where C=1\n",
            "        lr_img = np.expand_dims(lr_img, axis=0)\n",
            "        \n",
            "        if self.gt_files:\n",
            "            gt_path = self.gt_files[idx]\n",
            "            gt_img = np.load(gt_path).astype(np.float32)\n",
            "            gt_img = np.expand_dims(gt_img, axis=0)\n",
            "            return torch.from_numpy(lr_img), torch.from_numpy(gt_img)\n",
            "        \n",
            "        return torch.from_numpy(lr_img), os.path.basename(lr_path)\n",
            "\n",
            "print(\"Loading dataset...\")\n",
            "train_dataset = KLADataset(TRAIN_LR_DIR, TRAIN_GT_DIR)\n",
            "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n",
            "print(f\"Total batches: {len(train_loader)}\")"
        ]
    })

    # Cell 4: Model Architecture
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class ResBlock(nn.Module):\n",
            "    def __init__(self, channels):\n",
            "        super().__init__()\n",
            "        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)\n",
            "        self.relu = nn.ReLU(inplace=True)\n",
            "        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)\n",
            "\n",
            "    def forward(self, x):\n",
            "        residual = x\n",
            "        out = self.conv1(x)\n",
            "        out = self.relu(out)\n",
            "        out = self.conv2(out)\n",
            "        return out + residual\n",
            "\n",
            "class SuperResNet(nn.Module):\n",
            "    def __init__(self, num_blocks=16, num_features=64):\n",
            "        super().__init__()\n",
            "        self.head = nn.Conv2d(1, num_features, kernel_size=3, padding=1)\n",
            "        \n",
            "        blocks = [ResBlock(num_features) for _ in range(num_blocks)]\n",
            "        self.body = nn.Sequential(*blocks)\n",
            "        \n",
            "        self.upsample = nn.Sequential(\n",
            "            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),\n",
            "            nn.PixelShuffle(2), # Upscale 2x -> (256, 256)\n",
            "            nn.ReLU(inplace=True)\n",
            "        )\n",
            "        \n",
            "        self.tail = nn.Conv2d(num_features, 1, kernel_size=3, padding=1)\n",
            "\n",
            "    def forward(self, x):\n",
            "        out = self.head(x)\n",
            "        res = self.body(out)\n",
            "        out = out + res\n",
            "        out = self.upsample(out)\n",
            "        out = self.tail(out)\n",
            "        return out\n",
            "\n",
            "model = SuperResNet().to(device)\n",
            "print(model)"
        ]
    })

    # Cell 5: Training Loop
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "criterion = nn.L1Loss() # MAE Loss is preferred for SR\n",
            "optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)\n",
            "\n",
            "best_loss = float('inf')\n",
            "\n",
            "print(\"Starting training...\")\n",
            "for epoch in range(EPOCHS):\n",
            "    model.train()\n",
            "    epoch_loss = 0\n",
            "    \n",
            "    for i, (lr, gt) in enumerate(train_loader):\n",
            "        lr, gt = lr.to(device), gt.to(device)\n",
            "        \n",
            "        optimizer.zero_grad()\n",
            "        output = model(lr)\n",
            "        loss = criterion(output, gt)\n",
            "        \n",
            "        loss.backward()\n",
            "        optimizer.step()\n",
            "        \n",
            "        epoch_loss += loss.item()\n",
            "        \n",
            "    avg_loss = epoch_loss / len(train_loader)\n",
            "    print(f\"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}\")\n",
            "    \n",
            "    if avg_loss < best_loss:\n",
            "        best_loss = avg_loss\n",
            "        torch.save(model.state_with_dict() if hasattr(model, 'state_with_dict') else model.state_dict(), WEIGHTS_PATH)\n",
            "        print(f\" Saved best model weights to {WEIGHTS_PATH}\")"
        ]
    })

    # Cell 6: Prediction & Submission
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def encode_npy_base64(npy_array):\n",
            "    buffer = io.BytesIO()\n",
            "    np.save(buffer, npy_array)\n",
            "    buffer.seek(0)\n",
            "    encoded = base64.b64encode(buffer.read())\n",
            "    return encoded.decode('utf-8')\n",
            "\n",
            "print(\"Loading Best Weights...\")\n",
            "model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))\n",
            "model.eval()\n",
            "\n",
            "test_dataset = KLADataset(TEST_LR_DIR)\n",
            "test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)\n",
            "\n",
            "submission_data = []\n",
            "print(\"Running inference on test files...\")\n",
            "\n",
            "with torch.no_grad():\n",
            "    for lr_batch, filename_batch in test_loader:\n",
            "        lr = lr_batch.to(device)\n",
            "        filename = filename_batch[0]\n",
            "        output = model(lr)\n",
            "        \n",
            "        pred = output.cpu().numpy()[0, 0] # shape (256, 256)\n",
            "        # Clip to valid range if needed, depending on data normalization\n",
            "        pred = np.clip(pred, 0.0, 1.0)\n",
            "        \n",
            "        b64_str = encode_npy_base64(pred)\n",
            "        img_id = int(filename.split('.')[0]) + 1 # e.g., '000000.npy' -> 0 + 1 = 1\n",
            "        \n",
            "        submission_data.append((img_id, b64_str))\n",
            "\n",
            "submission_data = sorted(submission_data, key=lambda x: x[0])\n",
            "\n",
            "import pandas as pd\n",
            "df = pd.DataFrame(submission_data, columns=['id', 'npy_base64'])\n",
            "df.to_csv('submission_new.csv', index=False)\n",
            "print(\"Saved new submission to submission_new.csv!\")\n"
        ]
    })

    # Save to JSON structure
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open('/Users/ashutoshpatidar/Documents/KLA Hackathon/KLA AI - HACK/check.ipynb', 'w') as f:
        json.dump(nb_dict, f, indent=2)
    print("Notebook populated!")

generate_notebook()
