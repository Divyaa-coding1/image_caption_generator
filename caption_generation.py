import io
import base64
from PIL import Image, ImageDraw, ImageFont

# API Clients
import openai
import google.generativeai as genai
from groq import Groq


class MultiModalCaptionGenerator:
    def __init__(self) -> None:
        self.openai_client = None
        self.groq_client = None
        self.gemini_configured = False

    def configure_apis(self, openai_key:str|None=None, groq_key:str|None=None, gemini_key:str|None=None):
        if openai_key:
            self.openai_client = openai.OpenAI(api_key=openai_key)
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
        if gemini_key:
            self.gemini_configured = True
            genai.configure(api_key=gemini_key)

    def encode_image_base64(self, image:Image.Image)->str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def generate_caption_openai(self, image:Image.Image, model:str="gpt-5-nano")->str:
        if not self.openai_client:
            raise ValueError("OpenAI API key is not configured!")
        
        base64_image = self.encode_image_base64(image)

        response = self.openai_client.chat.completions.create(
            model=model, 
            messages=[
               { "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Generate an engaging caption for this image. Be concise in your choice of words. Maximum word limit: 20."
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }]}
            ],
            max_completion_tokens=20000
        )
        return response.choices[0].message.content
    
    def generate_caption_groq(self, image:Image.Image, model:str="meta-llama/llama-4-scout-17b-16e-instruct")->str:
        if not self.groq_client:
            raise ValueError("GROQ API key is not configured!")
        
        base64_image = self.encode_image_base64(image)

        completion = self.groq_client.chat.completions.create(
            model = model, 
            messages = [
                {
                    "role": "user", 
                    "content": [{
                        "type": "text",
                        "text": "Generate an engaging caption for this image. Be concise in your choice of words. Maximum word limit: 20."
                    },
                    {
                        "type":"image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        return completion.choices[0].message.content
    
    def generate_caption_gemini(self, image:Image.Image, model:str="gemini-2.5-flash-lite")->str:
        if not self.gemini_configured:
            raise ValueError("Gemini API key is not configured!")
        
        model_instance = genai.GenerativeModel(model_name=model)
        prompt = "Generate an engaging caption for this image. Be concise in your choice of words. Maximum word limit: 20."

        response = model_instance.generate_content([prompt, image])
        return response.text