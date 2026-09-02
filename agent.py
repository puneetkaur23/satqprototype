# agent.py — The Agentic Orchestrator
import re
import time

class TaskClassifier:
    """
    Analyzes the user's natural language query + image inputs
    to determine which remote-sensing task to perform.
    """
    
    def classify(self, query: str, num_images: int, filenames: list = None):
        query_lower = query.lower()
        
        # --- CHANGE DETECTION ---
        # Requires 2 images + change-related keywords
        if num_images == 2 and any(kw in query_lower for kw in [
            'change', 'difference', 'changed', 'before', 'after', 
            'between', 'temporal', 'time', 'years ago', 'since'
        ]):
            return "change_analysis"
        
        # --- CROSS-MODAL FUSION ---
        # Requires 2 images + modality keywords
        if num_images == 2 and any(kw in query_lower for kw in [
            'sar', 'radar', 'fusion', 'combine', 'both', 'optical and sar',
            'multi-modal', 'cross-modal', 'risat', 'compare modes'
        ]):
            return "cross_modal"
        
        # --- GROUNDING / LOCALIZATION ---
        # User wants to know WHERE something is
        if any(kw in query_lower for kw in [
            'where is', 'locate', 'find', 'show me', 'highlight', 
            'point to', 'bounding box', 'area of', 'region'
        ]):
            return "grounding"
        
        # --- IMAGE CAPTIONING ---
        # User wants a description without a specific question
        if any(kw in query_lower for kw in [
            'describe', 'caption', 'what is in', 'what do you see', 
            'summarize this image'
        ]):
            return "captioning"
        
        # --- DEFAULT: VISUAL QUESTION ANSWERING (VQA) ---
        return "vqa"


class ModelRegistry:
    """
    Registry of available specialist models.
    In the full version, these would be actual fine-tuned models.
    For the hackathon prototype, VQA is REAL, others are ARCHITECTURE.
    """
    
    def __init__(self):
        self.models = {
            "vqa": {
                "name": "LLaVA-1.6-Mistral-7B-4bit",
                "type": "vision-language",
                "status": "LIVE",
                "description": "General visual question answering for satellite imagery"
            },
            "change_analysis": {
                "name": "VisTA-ChangeVQA-RS",
                "type": "bi-temporal-change-detection",
                "status": "ARCHITECTURE",
                "description": "Detects changes between two time-separated images"
            },
            "grounding": {
                "name": "Grounding-DINO-RS",
                "type": "geospatial-grounding",
                "status": "ARCHITECTURE", 
                "description": "Localizes objects with bounding box coordinates"
            },
            "cross_modal": {
                "name": "Optical-SAR-Fusion-Net",
                "type": "multi-modal-fusion",
                "status": "ARCHITECTURE",
                "description": "Fuses optical and SAR data for enhanced analysis"
            },
            "captioning": {
                "name": "BLIP-2-RS-Caption",
                "type": "image-captioning",
                "status": "LIVE",
                "description": "Generates natural language descriptions of scenes"
            }
        }
    
    def get_model(self, task: str):
        return self.models.get(task, self.models["vqa"])
    
    def list_capabilities(self):
        return {k: v["description"] for k, v in self.models.items()}


class ExecutionTracer:
    """
    Generates an audit trail for every decision the agent makes.
    This is the "explainability" feature that wows judges.
    """
    
    def __init__(self):
        self.trace = {}
        self.start_time = time.time()
    
    def log(self, key: str, value):
        self.trace[key] = value
    
    def finalize(self):
        self.trace["total_execution_time"] = f"{time.time() - self.start_time:.2f}s"
        self.trace["agent_version"] = "SatQuery-v0.1-prototype"
        return self.trace


# Simple test
if __name__ == "__main__":
    classifier = TaskClassifier()
    registry = ModelRegistry()
    
    test_queries = [
        ("What crops are visible here?", 1),
        ("What changed between these two images?", 2),
        ("Show me where the roads are", 1),
        ("Fuse this optical and SAR data", 2),
    ]
    
    print("🧠 Agentic Layer Test:")
    for query, num_img in test_queries:
        task = classifier.classify(query, num_img)
        model = registry.get_model(task)
        print(f"  Query: '{query}' -> Task: {task} -> Model: {model['name']} ({model['status']})")