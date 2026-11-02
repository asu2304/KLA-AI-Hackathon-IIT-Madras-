import json

def generate_viz_notebook():
    cells = []

    # Cell 1: Comparison plot code
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 🖼️ Side-by-Side Comparison: Noisy LR vs Predicted HR\n",
            "\n",
            "This cell decodes predictions from `submission.csv` and loads the corresponding original noisy images from `test/NoisyLR/` to display them side-by-side using Matplotlib."
        ]
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import base64\n",
            "import io\n",
            "import os\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "def decode_npy_base64(b64_str):\n",
            "    \"\"\"Decode base64 string back to NumPy array.\"\"\"\n",
            "    buffer = io.BytesIO(base64.b64decode(b64_str))\n",
            "    return np.load(buffer)\n",
            "\n",
            "def plot_comparison(submission_path='submission.csv', noisy_dir='test/NoisyLR', num_images=5, start_index=0):\n",
            "    # Load submission\n",
            "    df = pd.read_csv(submission_path)\n",
            "    \n",
            "    fig, axes = plt.subplots(num_images, 2, figsize=(10, 4 * num_images))\n",
            "    if num_images == 1:\n",
            "        axes = np.expand_dims(axes, axis=0)\n",
            "        \n",
            "    for i in range(num_images):\n",
            "        idx = start_index + i\n",
            "        if idx >= len(df):\n",
            "            break\n",
            "            \n",
            "        row = df.iloc[idx]\n",
            "        img_id = int(row['id'])\n",
            "        b64_str = row['npy_base64']\n",
            "        \n",
            "        # Decode prediction\n",
            "        try:\n",
            "            pred_img = decode_npy_base64(b64_str)\n",
            "        except Exception as e:\n",
            "            print(f\"Error decoding id {img_id}: {e}\")\n",
            "            continue\n",
            "            \n",
            "        # Map 1-based submission ID to 0-based filename filename\n",
            "        # (Match: ID 1 -> 000000.npy)\n",
            "        noisy_filename = os.path.join(noisy_dir, f\"{img_id-1:06d}.npy\")\n",
            "        \n",
            "        # Load noisy image\n",
            "        if os.path.exists(noisy_filename):\n",
            "            noisy_img = np.load(noisy_filename)\n",
            "        else:\n",
            "            print(f\"File not found: {noisy_filename}\")\n",
            "            noisy_img = np.zeros_like(pred_img) # placeholder\n",
            "            \n",
            "        # Plotting\n",
            "        axes[i, 0].imshow(noisy_img, cmap='gray')\n",
            "        axes[i, 0].set_title(f\"Noisy LR (File: {img_id-1:06d}.npy)\")\n",
            "        axes[i, 0].axis('off')\n",
            "        \n",
            "        axes[i, 1].imshow(pred_img, cmap='gray')\n",
            "        axes[i, 1].set_title(f\"Submission Prediction (ID: {img_id})\")\n",
            "        axes[i, 1].axis('off')\n",
            "        \n",
            "    plt.tight_layout()\n",
            "    plt.show()\n",
            "\n",
            "# Usage: Plot first 5 comparisons\n",
            "plot_comparison(num_images=3, start_index=0)"
        ]
    })

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

generate_viz_notebook()
print("Visualization Code populated.")
