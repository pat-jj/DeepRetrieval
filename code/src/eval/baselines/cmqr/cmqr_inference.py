import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import logging

# Set up simple logging
logging.basicConfig(level=logging.WARNING)  # Suppress info logs for cleaner output

class CMQR:
    """Simple API for CMQR query rewriting: query in, rewritten query out"""
    
    def __init__(self, model_path='/shared/eng/pj20/cmqr_model', beam_width=10):
        """
        Initialize CMQR model
        
        Args:
            model_path (str): Path to trained CMQR model
            beam_width (int): Beam width for generation
        """
        self.beam_width = beam_width
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        if os.path.exists(model_path):
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            # Fallback to base T5 if trained model not found
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
            self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        
        self.model.to(self.device)
        self.model.eval()
    
    def rewrite(self, query, context="", num_results=1):
        """
        Rewrite a query
        
        Args:
            query (str): Input query to rewrite
            context (str): Optional conversation context
            num_results (int): Number of rewrites to return (default 1)
            
        Returns:
            str or list: If num_results=1, returns single rewritten query string
                        If num_results>1, returns list of rewritten queries
        """
        # Prepare input
        if context.strip():
            input_text = f"{context.strip()} {query}"
        else:
            input_text = query
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=self.beam_width,
                num_return_sequences=min(num_results, self.beam_width),
                return_dict_in_generate=True,
                output_scores=True,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode results
        sequences = outputs.sequences
        scores = outputs.sequences_scores
        
        results = []
        for seq, score in zip(sequences, scores):
            rewritten_query = self.tokenizer.decode(seq, skip_special_tokens=True)
            results.append(rewritten_query)
        
        # Return single string if only one result requested, otherwise list
        if num_results == 1:
            return results[0] if results else query
        else:
            return results
    
    def __call__(self, query, context=""):
        """Allow calling the model directly: rewritten = cmqr(query)"""
        return self.rewrite(query, context)


# Example usage
if __name__ == "__main__":
    # Initialize the model
    cmqr = CMQR()
    
    # Simple usage - query in, rewritten query out
    query = "What is it?"
    context = "We were talking about machine learning"
    
    rewritten = cmqr.rewrite(query, context)
    print(f"Original: {query}")
    print(f"Rewritten: {rewritten}")
    
    # Or use it as a callable
    rewritten2 = cmqr(query, context)
    print(f"Callable result: {rewritten2}")
    
    # Get multiple rewrites
    multiple_rewrites = cmqr.rewrite(query, context, num_results=3)
    print(f"Multiple rewrites: {multiple_rewrites}")
    
    # Without context
    simple_rewrite = cmqr("What are the benefits?")
    print(f"Without context: {simple_rewrite}")