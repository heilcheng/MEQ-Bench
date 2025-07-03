"""
Prompt templates for audience-adaptive medical explanations
"""

class AudienceAdaptivePrompt:
    """
    Standardized prompt template for generating audience-adaptive medical explanations
    """
    
    base_template = """Medical Information: {medical_content}

Transform this explanation for four distinct audiences. Ensure each explanation is self-contained and clearly labeled:

- For a Physician: Provide a technical, evidence-based explanation using precise medical terminology.

- For a Nurse: Focus on practical care implications, monitoring parameters, and patient education points.

- For a Patient: Use simple, jargon-free, and empathetic language. Focus on what the condition means and the next steps.

- For a Caregiver: Provide concrete tasks, symptoms to watch for, and when to seek help.

Generated Explanations:"""

    @classmethod
    def format_prompt(cls, medical_content: str) -> str:
        """
        Format the prompt template with medical content
        
        Args:
            medical_content: The medical information to be adapted
            
        Returns:
            Formatted prompt string
        """
        return cls.base_template.format(medical_content=medical_content)
    
    @staticmethod
    def parse_response(response: str) -> dict:
        """
        Parse the model response to extract audience-specific explanations
        
        Args:
            response: Model response containing all audience explanations
            
        Returns:
            Dictionary with audience-specific explanations
        """
        audiences = ["physician", "nurse", "patient", "caregiver"]
        explanations = {}
        
        # Simple parsing logic - can be enhanced with more sophisticated methods
        current_audience = None
        current_text = []
        
        for line in response.split('\n'):
            line = line.strip()
            if any(aud.lower() in line.lower() for aud in audiences):
                if current_audience and current_text:
                    explanations[current_audience] = '\n'.join(current_text).strip()
                current_audience = next((aud for aud in audiences if aud.lower() in line.lower()), None)
                current_text = []
            elif current_audience and line:
                current_text.append(line)
        
        # Add the last audience explanation
        if current_audience and current_text:
            explanations[current_audience] = '\n'.join(current_text).strip()
            
        return explanations