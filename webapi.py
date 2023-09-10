import os
import io
import cv2
import torch
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
#from starlette.responses import FileResponse

from model import Modified_Unet

app = FastAPI()

# Create necessary directories
os.makedirs("static/Results", exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model

model = Modified_Unet()
model = model.to(device)
#model_weights_path = "weights/MobileNetV2_Unet_wts (1).pth"

model_weights_path = "weights/MobileNetV2_Unet 30 epoch.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.eval()

# Preprocessing transformations
preprocess_X = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0,0,0),std=(1,1,1))])

# preprocess_y = transforms.Compose([
#             transforms.Resize((258, 258)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0),std=(1))])

def apply_mask(input_image, mask_image):

    mask_3channel = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    mask_3channel_resized = cv2.resize(mask_3channel, (input_image.shape[1], input_image.shape[0]))
    output_white = cv2.addWeighted(input_image, 1, 255-mask_3channel_resized, 0.5, 0)
    output_black = cv2.bitwise_and(input_image, mask_3channel_resized)

    return output_white,output_black

# @app.get("/")
# async def start():
#     return {"message": "Welcome to the image processing API"}

@app.post("/upload")
async def image_process(file: UploadFile):
    try:

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        image.save("static/Results/input.png", format='PNG')

        input_image = Image.open("static/Results/input.png")
        input_tensor = preprocess_X(input_image)

        with torch.no_grad():
            input_tensor = input_tensor.unsqueeze(0).to(device)
            output_image = model(input_tensor)
            output_mask_image = transforms.ToPILImage()(output_image.squeeze(0).cpu())

        output_mask_image.save("static/Results/mask.png", format="PNG")

        input_image_cv = cv2.imread("static/Results/input.png")
        mask_cv = cv2.imread("static/Results/mask.png", cv2.IMREAD_GRAYSCALE)

        binary_mask = cv2.adaptiveThreshold(mask_cv,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
  
        output_image_white, output_image_black = apply_mask(input_image_cv, binary_mask)

        cv2.imwrite("static/Results/output_b.png", output_image_black)
        cv2.imwrite("static/Results/output_w.png", output_image_white)

        return {"message": "Image uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn webapi:app --reload