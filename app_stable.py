import cv2
import numpy as np
import gradio as gr
import onnxruntime as rt
import os
from PIL import Image

#full path to the onnx model
model_path = r"D:\Anime aesthetic skynt\model.onnx"
model = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def safe_read_image(path):
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))
# singel image input, returns a single score.
def predict(img):
    img = img.astype(np.float32) / 255
    s = 768
    h, w = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    pred = model.run(None, {"img": img_input})[0].item()
    return pred
# Batch image input, returns a list of scores.
def batch_predict(folder_path):
    supported_exts = ['.jpg', '.png', '.jpeg']
    results = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext)for ext in supported_exts):
            img_path = os.path.join(folder_path, filename)
            img = safe_read_image(img_path)
            if img is not None:
                score = predict(img)
                results.append(f"{filename}: {score:.4f}")
    return "\n".join(results) if results else "No valid images found."
    
if __name__ == "__main__":
    examples = [[]]
    examples = [[f"examples/{x:02d}.jpg"] for x in range(0, 2)]
## Gradio interface
single_image_interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(label="input image"),
        outputs=gr.Number(label="score"),
        title="Anime Aesthetic Score Predictor",
        allow_flagging="never",
        examples=examples,
        cache_examples=False,
    ) 

folder_interface = gr.Interface(
        fn=batch_predict,
        inputs=gr.Textbox(label="Folder Path", placeholder="./your_folder_path"),
        outputs=gr.Textbox(label="Scores"),
        title="Batch Anime Aesthetic Score Predictor",
        description="Enter the path to a folder containing anime images and get their aesthetic scores.",
        allow_flagging="never",
)
#use tabbed interface to switch between single image and batch processing
app = gr.TabbedInterface(
    interface_list=[single_image_interface, folder_interface],
    tab_names=["single_image_score", "batch_score"],
)
app.launch()