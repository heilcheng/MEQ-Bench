"""
Tests for the robust LLM response parser
"""

import pytest
from src.prompt_templates import AudienceAdaptivePrompt


class TestRobustParser:
    """Test the robust LLM response parser with various input formats"""
    
    def test_standard_format(self):
        """Test parsing with standard format: 'For a Physician:'"""
        response = """
        For a Physician: This is a technical explanation with medical terminology and evidence-based recommendations for clinical practice.
        
        For a Nurse: This focuses on practical care implications, monitoring parameters, and patient education points for nursing staff.
        
        For a Patient: This uses simple, jargon-free language to explain what the condition means and next steps in an empathetic way.
        
        For a Caregiver: This provides concrete tasks, symptoms to watch for, and clear guidance on when to seek help.
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert len(result) == 4
        assert 'physician' in result
        assert 'nurse' in result
        assert 'patient' in result
        assert 'caregiver' in result
        
        assert 'technical explanation' in result['physician']
        assert 'practical care implications' in result['nurse']
        assert 'simple, jargon-free' in result['patient']
        assert 'concrete tasks' in result['caregiver']
    
    def test_colon_format(self):
        """Test parsing with simple colon format: 'Physician:'"""
        response = """
        Physician: Technical medical explanation for healthcare professionals.
        
        Nurse: Practical nursing care guidelines and monitoring.
        
        Patient: Easy-to-understand explanation for patients.
        
        Caregiver: Clear instructions for family caregivers.
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert len(result) == 4
        assert 'Technical medical explanation' in result['physician']
        assert 'Practical nursing care' in result['nurse']
        assert 'Easy-to-understand' in result['patient']
        assert 'Clear instructions' in result['caregiver']
    
    def test_numbered_format(self):
        """Test parsing with numbered format: '1. Physician:'"""
        response = """
        1. Physician: Advanced medical explanation with clinical terminology.
        
        2. Nurse: Care plan and monitoring responsibilities.
        
        3. Patient: Simple explanation of the condition.
        
        4. Caregiver: Supportive care instructions.
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert len(result) == 4
        assert 'Advanced medical explanation' in result['physician']
        assert 'Care plan and monitoring' in result['nurse']
        assert 'Simple explanation' in result['patient']
        assert 'Supportive care instructions' in result['caregiver']
    
    def test_markdown_header_format(self):
        """Test parsing with markdown headers: '## Physician'"""
        response = """
        ## Physician
        Detailed pathophysiology and treatment protocols.
        
        ## Nurse  
        Nursing assessment and intervention guidelines.
        
        ## Patient
        Patient-friendly explanation of the medical condition.
        
        ## Caregiver
        Family support and care coordination.
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert len(result) == 4
        assert 'pathophysiology' in result['physician']
        assert 'assessment and intervention' in result['nurse']
        assert 'Patient-friendly' in result['patient']
        assert 'Family support' in result['caregiver']
    
    def test_bold_format(self):
        """Test parsing with bold formatting: '**Physician:**'"""
        response = """
        **Physician:** Medical explanation with clinical focus.
        
        **Nurse:** Nursing care priorities and protocols.
        
        **Patient:** Accessible explanation for patient understanding.
        
        **Caregiver:** Practical guidance for family members.
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert len(result) == 4
        assert 'clinical focus' in result['physician']
        assert 'care priorities' in result['nurse']
        assert 'Accessible explanation' in result['patient']
        assert 'Practical guidance' in result['caregiver']
    
    def test_case_insensitive(self):
        """Test that parsing is case-insensitive"""
        response = """
        FOR A PHYSICIAN: Upper case format explanation.
        
        for a nurse: Lower case format explanation.
        
        For A PATIENT: Mixed case format explanation.
        
        for a Caregiver: Mixed case format explanation.
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert len(result) == 4
        assert 'Upper case format' in result['physician']
        assert 'Lower case format' in result['nurse']
        assert 'Mixed case format' in result['patient']
        assert 'Mixed case format' in result['caregiver']
    
    def test_whitespace_variations(self):
        """Test parsing with various whitespace variations"""
        response = """
        For a Physician   :   Explanation with extra spaces.
        
        
        For a Nurse:
        Explanation on new line.
        
        For a Patient     : Explanation with tabs and spaces.
        
        For a Caregiver:
        
        Explanation with blank line.
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert len(result) == 4
        assert 'extra spaces' in result['physician']
        assert 'new line' in result['nurse']
        assert 'tabs and spaces' in result['patient']
        assert 'blank line' in result['caregiver']
    
    def test_multiline_explanations(self):
        """Test parsing with multi-line explanations"""
        response = """
        For a Physician: 
        This is the first line of the physician explanation.
        This is the second line with more details.
        And this is the third line with even more information.
        
        For a Nurse:
        First line for nurses.
        Second line with care instructions.
        
        For a Patient:
        Simple first line.
        Simple second line.
        
        For a Caregiver:
        Caregiver instruction line one.
        Caregiver instruction line two.
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert len(result) == 4
        assert 'first line' in result['physician']
        assert 'second line' in result['physician'] 
        assert 'third line' in result['physician']
        assert 'care instructions' in result['nurse']
        assert 'Simple first line' in result['patient']
        assert 'instruction line one' in result['caregiver']
    
    def test_fallback_parsing(self):
        """Test fallback parsing when regex fails"""
        response = """
        Some text about physicians and their role.
        
        physician explanation here
        
        nurse content follows
        
        patient information provided
        
        caregiver details included
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        # Should still extract some content even with poor formatting
        assert isinstance(result, dict)
    
    def test_empty_response(self):
        """Test parsing with empty response"""
        response = ""
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_malformed_response(self):
        """Test parsing with malformed response"""
        response = "This is just random text without any audience markers."
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert isinstance(result, dict)
        # Should return empty dict if no audiences found
    
    def test_partial_audiences(self):
        """Test parsing when only some audiences are present"""
        response = """
        For a Physician: Technical explanation only for doctors.
        
        For a Patient: Simple explanation only for patients.
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        assert len(result) == 2
        assert 'physician' in result
        assert 'patient' in result
        assert 'nurse' not in result
        assert 'caregiver' not in result
    
    def test_text_cleaning(self):
        """Test that extracted text is properly cleaned"""
        response = """
        For a Physician:   **Bold text** and *italic text* with `code` formatting.
        
        
        Multiple blank lines should be normalized.
        
        
        For a Patient: - Leading dash should be removed
        """
        
        result = AudienceAdaptivePrompt.parse_response(response)
        
        # Check that markdown formatting is removed
        assert '**' not in result['physician']
        assert '*' not in result['physician'] 
        assert '`' not in result['physician']
        assert 'Bold text' in result['physician']
        assert 'italic text' in result['physician']
        assert 'code' in result['physician']
        
        # Check that leading dash is removed
        assert not result['patient'].startswith('-')
        assert 'Leading dash should be removed' in result['patient']