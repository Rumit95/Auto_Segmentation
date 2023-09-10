# Semantic Segmentation with MobileNetV2 U-Net

This project demonstrates semantic image segmentation using a custom architecture that combines MobileNetV2 and U-Net. The goal is to accurately segment objects in images, showcasing a robust approach to image analysis.

## Project Overview

- The core functionality of this project is implemented in the `main.py` and `webapi.py` files. `main.py` contains the code for running the semantic segmentation locally, while `webapi.py` is a Streamlit web application for performing semantic segmentation through a user-friendly interface.

![Web App demo](sample.mp4 "Web App")

## Prerequisites

- Python (>=3.6)
- PyTorch (>=1.0)
- torchvision
- tqdm
- matplotlib
- numpy
- PIL
- Streamlit (for the web application)

## Project Structure

- `main.py`: Python script for running semantic segmentation locally.
- `webapi.py`: Streamlit web application for semantic segmentation.
- `MobileNetV2_Unet_wts.pth`: Saved model weights.
- `MobileNetV2_Unet_model.pth`: Saved entire model (architecture and weights).

## Getting Started

1. Clone this repository: `git clone https://github.com/yourusername/semantic-segmentation.git`
2. Navigate to the project directory: `cd semantic-segmentation`

### Running Locally

3. To perform semantic segmentation locally, run the following command:
`python main.py`

This will start the segmentation process, and you can input your images for analysis.

### Running the Streamlit Web Application

4. To use the web application, run the following command:
`streamlit run webapi.py`

This will open a browser window with the Streamlit interface. You can upload images and perform semantic segmentation interactively.

## Usage

- When running locally with `main.py`, follow the prompts to input your images for segmentation. The results will be displayed on your console.

- When using the Streamlit web application, open it in your web browser, upload images, and click the "Segment" button. The segmented images will be displayed on the web interface, and you can download them as well.

- You can modify hyperparameters, the number of training epochs, or experiment with different architectures in the `main.py` file to customize the segmentation process.

## Credits

- The MobileNetV2 architecture is based on the original paper by Sandler et al. [https://arxiv.org/abs/1801.04381].
- U-Net architecture reference: Olaf Ronneberger, Philipp Fischer, Thomas Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation."

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to reach out if you have any questions or suggestions!
