import os 
import json
import datetime
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage 
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory 
from typing import Dict, List

class CaptionHistory:
    def __init__(self, use_file_history:bool=True) -> None:
        self.history_file = "caption_history.json"
        self.metadata_file = "caption_metadata.json"

        if use_file_history:
            self.chat_history = FileChatMessageHistory(file_path=self.history_file)

        else:
            # Use in-memory history with manual persistance 
            self.chat_history = InMemoryChatMessageHistory()
            self.load_history()

    def add_interaction(self, image_name:str, model:str, caption:str, timestamp:str=None):
        if not timestamp:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

            # Create message with metadata 
            human_msg = HumanMessage(content=f"Generate caption for {image_name} using {model}.", 
            additional_kwargs={
                "image_name": image_name, 
                "model": model, 
                "timestamp": timestamp
            })

            ai_msg = AIMessage(content=caption, 
            additional_kwargs={
                    "image_name": image_name,
                    "model": model,
                    "timestamp": timestamp
            })

            # Add to Langchain chat history 
            self.chat_history.add_message(human_msg)
            self.chat_history.add_message(ai_msg)

            # Save metadata separately for easy querying 
            self.save_metadata({
                "timestamp": timestamp,
                "image_name": image_name,
                "model": model,
                "caption": caption
            })

    def get_messages(self)->List[BaseMessage]:
        return self.chat_history.messages
    
    def get_history(self)->List[Dict]:
        try:
            with open(self.metadata_file, mode="r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        
    def save_metadata(self, interaction:Dict):
        history = self.get_history()
        history.append(interaction)

        with open(self.metadata_file, mode="w") as f:
            json.dump(history, f, indent=2)

    def load_history(self):
        if isinstance(self.chat_history, FileChatMessageHistory):
            return
        
        history = self.get_history()
        for item in history:
            human_msg = HumanMessage(
                content=f"Generate caption for {item['image_name']} using {item['model']}",
                additional_kwargs={
                    "image_name": item["image_name"], 
                    "model": item["model"],
                    "timestamp": item["timestamp"]
                }
            )
            ai_msg = AIMessage(
                content=item["caption"], 
                additional_kwargs= {
                    "image_name": item["image_name"], 
                    "model": item["model"],
                    "timestamp": item["timestamp"]
                }
            )
            self.chat_history.add_message(human_msg)
            self.chat_history.add_message(ai_msg)

    def clear_history(self):
        self.chat_history.clear()

        # Remove metadata file 
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)

        # Remove history file if using manual persistance 
        if isinstance(self.chat_history, FileChatMessageHistory):
            if os.path.exists(self.history_file):
                os.remove(self.history_file) 

    def get_recent_interactions(self, n:int=10)->List[Dict]:
        history = self.get_history()
        return history[-n:] if len(history) > n else history
    
    def search_by_model(self, model:str)->List[Dict]:
        history = self.get_history()
        return [item for item in history if item.get("model", None) == model]
    
    def search_by_image(self, image_name:str)->List[Dict]:
        history = self.get_history()
        return [item for item in history if item.get("image_name", None) == image_name]
    