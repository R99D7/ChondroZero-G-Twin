import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np

class FutureStateGenerator:
    """
    Predictive Diffusion Model for generating synthetic, future-state SEM images
    of cartilage based on current degradation vector conditions.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Diffusion Pipeline on {self.device}...")
        
        # In a real deployed environment, we might use a specifically fine-tuned LoRA for SEM images.
        # We load a lightweight or standard diffusers pipeline for img2img.
        # Note: Using float16 for memory efficiency if on CUDA
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                safety_checker=None # Disabled for medical imagery
            )
            self.pipe = self.pipe.to(self.device)
            # Reduce memory footprint
            self.pipe.enable_attention_slicing()
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            self.pipe = None

    def generate_6_month_prediction(self, init_image_path: str, degradation_vector: dict, output_path: str = "future_sem.png"):
        """
        Takes the current structural SEM image and degradation vector,
        and generates the 6-month future state.
        
        Args:
            init_image_path: Path to the current SEM image.
            degradation_vector: Dict containing metrics like {'matrix_integrity': 0.6, 'drug_concentration': 0.8}
            output_path: Where to save the generated image.
        """
        if self.pipe is None:
            raise RuntimeError("Diffusion pipeline is not initialized.")
            
        # 1. Load and preprocess the initial SEM image
        init_image = Image.open(init_image_path).convert("RGB")
        init_image = init_image.resize((512, 512))
        
        # 2. Formulate the conditional prompt based on the Neural ODE degradation vector
        integrity = degradation_vector.get('matrix_integrity', 0.5)
        # Lower integrity -> more degraded, higher porosity, fibrotic tissue.
        if integrity > 0.7:
            condition_desc = "healthy, dense cartilage extracellular matrix, smooth surface, organized collagen network"
        elif integrity > 0.4:
            condition_desc = "mild osteoarthritis, some surface fibrillation, slight loss of proteoglycans, moderate degradation"
        else:
            condition_desc = "severe cartilage degradation, deep fissures, exposed subchondral bone, high porosity, extensive matrix breakdown"
            
        prompt = f"High resolution Scanning Electron Microscope (SEM) image of cartilage tissue, {condition_desc}, grayscale, highly detailed"
        negative_prompt = "color, drawing, painting, blurry, low resolution, artifacts, cells out of focus"
        
        # 3. Strength defines how much to transform the original image (0.0 to 1.0)
        # Higher degradation means we want to transform it more.
        strength = 1.0 - integrity 
        # Clamp strength to reasonable bounds so we don't completely lose original structure
        strength = max(0.3, min(0.8, strength))
        
        print(f"Generating future state with prompt: '{prompt}' and strength: {strength}")
        
        # 4. Generate the image
        with torch.no_grad():
            generated_images = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=7.5
            ).images
            
        future_image = generated_images[0]
        
        # 5. Save the output
        future_image.save(output_path)
        print(f"Generated 6-month future state saved to {output_path}")
        return future_image

if __name__ == "__main__":
    # Example Usage (Requires an actual image to test)
    # generator = FutureStateGenerator()
    # To test properly, you need an initial image at 'data/raw/base_sem.png'
    print("FutureStateGenerator module initialized.")
