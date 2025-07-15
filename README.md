# 🧠 Skin Disease Detection using Hugging Face Transformers

This project is a Python-based image classification tool that uses a fine-tuned Hugging Face model to detect potential skin diseases from image inputs.

## 🔍 Overview

This tool loads a pretrained model (`Jayanth2002/dinov2-base-finetuned-SkinDisease`) from the Hugging Face Hub and predicts the most likely skin condition from an input image.

### ✅ Features
- Predicts the **top 5 possible skin diseases** with their confidence scores
- Built using `transformers`, `torch`, and `Pillow`
- Simple, standalone script for image-based predictions
- Works with any RGB skin image input

## 📁 File Structure

```
Skin Disease Model/
├── Skin_model.py           # Main inference script
├── sample_img2.jpg         # Sample input image 1
├── sample_img3.jpg         # Sample input image 2
```

## 🧪 Requirements

Install the dependencies using pip:

```bash
pip install torch transformers Pillow
```

## 🚀 How to Run

1. Clone this repository or download the files.
2. Place your image in the same directory and update `image_path` in `Skin_model.py`.
3. Run the script:

```bash
python Skin_model.py
```

The script will output the top 5 predicted skin conditions and their confidence scores.

## 🖼️ Sample Output

```
Top Predictions:
Psoriasis: 92.12%
Eczema: 4.63%
Melanoma: 1.77%
...
Based on top probability, predicted skin disease might be Psoriasis
```

## 🔗 Model Details

- Model: [`Jayanth2002/dinov2-base-finetuned-SkinDisease`](https://huggingface.co/Jayanth2002/dinov2-base-finetuned-SkinDisease)
- Source: Hugging Face Hub

## 📌 Notes

- This tool is built for educational purposes and should **not be used for actual medical diagnosis**.
- Accuracy may vary depending on image quality and conditions.

## 📫 Connect

Feel free to raise an issue or connect with me on [LinkedIn](https://www.linkedin.com) if you have questions or suggestions!

---

Made with ❤️ using Hugging Face Transformers and a passion for learning.
