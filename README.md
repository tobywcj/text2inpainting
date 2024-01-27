# Text-prompt Inpainting WebApp

This application uses the ClipSeg and Stable Diffusion models to identify and transform elements in images.

---

## Demo



## Features

- Identify elements in images using ClipSeg
- Transform identified elements using Stable Diffusion
- Display the transformed images in the application

---

## How to Use

1. Load an image into the application.
2. Enter the prompts for the elements you want to identify in the image.
3. The application will use ClipSeg to identify the elements in the image and display the masks for each element.
4. Enter the prompts for how you want to transform the identified elements.
5. The application will use Stable Diffusion to transform the identified elements and display the transformed images.

---

## Functions

- `process_with_clipseg(clip_model, tensor_image, target_prompts)`: This function uses the ClipSeg model to identify elements in the image based on the target prompts. It returns the masks for the identified elements.

- `process_with_stable_diffusion(diffusion_pipe, source_image, stable_diffusion_masks, target_prompts, inpainting_prompts)`: This function uses the Stable Diffusion model to transform the identified elements based on the inpainting prompts. It returns the transformed images.

---

## Installation

1. Clone the repository.
2. `` git clone https://github.com/timojl/clipseg.git ``
2. Install the required packages using pip: `pip install -r requirements.txt`
3. Run the application: `streamlit run main.py`

---

## Note

This application is intended for educational and research purposes. Please use responsibly.