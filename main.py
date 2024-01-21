import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np

# Import Clip, ClipSeg, and Stable Diffusion models
import clip
from clipseg.models.clipseg import CLIPDensePredT
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

use_gpu = True


# Load models
@st.cache_resource
def load_models():
    # load clipseg model
    clip_model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    clip_model.eval()
    clip_model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False) # non-strict mode: decoder weights only (no CLIP weights)

    # load stable diffusion model
    model_dir="stabilityai/stable-diffusion-2-inpainting"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler") # The scheduler determine the algorithm used to produce new samples during the denoising process
    if use_gpu:
        diffusion_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir,
                                                        scheduler=scheduler,
                                                        revision="fp16",
                                                        torch_dtype=torch.float16)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
        diffusion_pipe = diffusion_pipe.to(device)
        if torch.cuda.is_available():
            diffusion_pipe.enable_xformers_memory_efficient_attention()
    else:
        diffusion_pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir,
                                                        scheduler=scheduler,
                                                        revision="fp16")
        device = torch.device('cpu')
        diffusion_pipe = diffusion_pipe.to(device)

    return clip_model, diffusion_pipe, device



# System configuration
def system_configuration():
    if 'clip_model' not in st.session_state and 'diffusion_pipe' not in st.session_state and 'device' not in st.session_state:
        st.session_state.clip_model, st.session_state.diffusion_pipe, st.session_state.device = load_models()
    if 'uploaded_image_unprocessed' not in st.session_state:
        st.session_state.uploaded_image_unprocessed  = None
    if 'source_image' not in st.session_state:
        st.session_state.source_image = None
    if 'tensor_image' not in st.session_state:
        st.session_state.tensor_image = None
    if 'target_prompts' not in st.session_state:
        st.session_state.target_prompts = []
    if 'inpainting_prompts' not in st.session_state:
        st.session_state.inpainting_prompts = []
    if 'target_element1' not in st.session_state:
        st.session_state.target_element1 = None
    if 'target_element2' not in st.session_state:
        st.session_state.target_element2 = None
    if 'inpaint1' not in st.session_state:
        st.session_state.inpaint1 = None
    if 'inpaint2' not in st.session_state:
        st.session_state.inpaint2 = None
    if 'stable_diffusion_masks' not in st.session_state:
        st.session_state.stable_diffusion_masks = None
    if 'transformed_images' not in st.session_state:
        st.session_state.transformed_images = None


# Reset session state
def reset_session_state():
    st.session_state.clip_model, st.session_state.diffusion_pipe, st.session_state.device = load_models()
    st.session_state.uploaded_image_unprocessed  = None
    st.session_state.source_image = None
    st.session_state.tensor_image = None
    st.session_state.target_prompts = []
    st.session_state.inpainting_prompts = []
    st.session_state.target_element1 = None
    st.session_state.target_element2 = None
    st.session_state.inpaint1 = None
    st.session_state.inpaint2 = None
    st.session_state.stable_diffusion_masks = None
    st.session_state.transformed_images = None


# Preprocess the image
@st.cache_data
def process_image(uploaded_image):

    target_width, target_height = 512,512
    source_image = Image.open(uploaded_image)

    width, height = source_image.size
    print(f"Source image size: {source_image.size}")

    source_image = source_image.crop((0, height-width , width , height))  # box=(left, upper, right, lower)
    source_image = source_image.resize((target_width, target_height), Image.LANCZOS)
    print(f"Target image size: {source_image.size}")

    # Setup transformations to be applied to the image aligning the input requirement of the clipseg model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor_image = transform(source_image).unsqueeze(0)

    return source_image, tensor_image


# ClipSeg model
def process_with_clipseg(clip_model, tensor_image, target_prompts):

    if use_gpu == False:
        tensor_image = tensor_image.float()
    # st.text(f"tensor_image shape: {tensor_image.shape}")
    # st.text(f"Target prompts: {target_prompts}")

    # Use ClipSeg to identify elements in picture
    with torch.no_grad():
        tensor_images_masks = clip_model(tensor_image.repeat(len(target_prompts),1,1,1), target_prompts)[0]

    # st.text(f"tensor_images_masks shape: {tensor_images_masks.shape}")

    processed_masks = []
    stable_diffusion_masks = []

    # Normalize mask values by computing the area under Gaussan probability density function, calculating the cumulative distribution with ndtr
    for i in range(len(tensor_images_masks)):
        processed_masks.append(torch.special.ndtr(tensor_images_masks[i][0]))
        stable_diffusion_masks.append(transforms.ToPILImage()(processed_masks[i]))
        st.image(stable_diffusion_masks[i], caption=f'Mask of element {i+1}', use_column_width=True)

    return stable_diffusion_masks


# Stable Diffusion model
def process_with_stable_diffusion(diffusion_pipe, source_image, stable_diffusion_masks, target_prompts, inpainting_prompts):

    generator = torch.Generator(device=st.session_state.device).manual_seed(77) # 155, 77, 

    # Run Stable Diffusion pipeline in inpainting mode
    transformed_images = []
    for i in range(len(stable_diffusion_masks)):
            with st.spinner(f"Transforming the {target_prompts[i]} ..."):
                image = diffusion_pipe(prompt=inpainting_prompts[i], guidance_scale=7.5, num_inference_steps=60, generator=generator, image=source_image, mask_image=stable_diffusion_masks[i]).images[0]
                transformed_images.append(image)
                st.image(image, caption=target_prompts[i], use_column_width=True)

    return transformed_images



def main():

    # set page configuration
    st.set_page_config(page_title='Virtual Try On', page_icon='👗', layout='wide')

    st.title('Virtual Try On')

    col1, col2 = st.columns(2)

    with col1:

        # Upload image
        st.session_state.uploaded_image_unprocessed = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

        # set system configuration and load models
        system_configuration()

        if st.session_state.uploaded_image_unprocessed is not None:
            st.session_state.source_image, st.session_state.tensor_image = process_image(st.session_state.uploaded_image_unprocessed)
            st.image(st.session_state.source_image, caption='Uploaded Image', use_column_width=True)

            # Text input for target element and transformation prompt
            st.session_state.target_element1 = st.text_input("Target Element 1").strip().lower()
            st.session_state.inpaint1 = st.text_input("Inpainting Prompt 1").strip().lower()
            st.session_state.target_element2 = st.text_input("Target Element 2").strip().lower()
            st.session_state.inpaint2 = st.text_input("Inpainting Prompt 2").strip().lower()
    

    with col2:

        # Reset session state
        if st.button('Reset'):
            reset_session_state()
            # st.experimental_rerun()

        if st.button('Start Transformation'):

            if st.session_state.source_image is None:
                st.error('Please upload an image.')
        
            if (st.session_state.target_element1 and st.session_state.inpaint1) or (st.session_state.target_element2 and st.session_state.inpaint2):
                with st.spinner('Magic happening ...'): 

                    for target in [st.session_state.target_element1, st.session_state.target_element2]:
                        if target:
                            st.session_state.target_prompts.append(target) 

                    for inpaint in [st.session_state.inpaint1, st.session_state.inpaint2]:
                        if inpaint:
                            st.session_state.inpainting_prompts.append(inpaint)

                    st.info(f"Target prompts: {st.session_state.target_prompts}")
                    st.info(f"Inpainting prompts: {st.session_state.inpainting_prompts}")

                    # Run ClipSeg model
                    with st.spinner('Finding and Locating the target element(s) ...'):
                        st.session_state.stable_diffusion_masks = process_with_clipseg(st.session_state.clip_model, st.session_state.tensor_image, st.session_state.target_prompts)

                    # Run Stable Diffusion model
                    with st.spinner('Generating new element ...'):
                        st.session_state.transformed_images = process_with_stable_diffusion(st.session_state.diffusion_pipe, st.session_state.source_image, st.session_state.stable_diffusion_masks, st.session_state.target_prompts, st.session_state.inpainting_prompts)

                    st.success('Transformation Complete !')

                    # # Save images
                    # if st.button('Save Images'):
                    #     with st.spinner('Saving images...'):
                    #         for i, image in enumerate(st.session_state.transformed_images):
                    #             image_name = f"image_{i}.png"
                    #             image.save(image_name)
                        
                    #         st.balloons() # display a balloon when the image is saved successfully
                    #         st.success(f"Saved {image_name}")
                    #         os.remove(image_name) # remove the image file after saving it to the streamlit app

            else:
                st.error('Please fill in at least one target element and its inpainting prompt.')
    



if __name__ == '__main__':
    main()
