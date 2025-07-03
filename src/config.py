"""Configuration management system for MEQ-Bench.

This module provides a centralized configuration management system that loads settings
from YAML files and environment variables. It implements a singleton pattern to ensure
consistent configuration access across the application.

The configuration system supports:
    - YAML-based configuration files
    - Environment variable overrides for sensitive data (API keys)
    - Validation of required configuration sections
    - Logging setup and management
    - Multiple configuration environments
"""

import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional


class ConfigurationError(Exception):
    """Raised when there's an error in configuration.
    
    This exception is raised when:
        - Configuration files are missing or invalid
        - Required configuration sections are missing
        - YAML parsing fails
        - Required environment variables are not set
    """
    pass


class Config:
    """Singleton configuration manager for MEQ-Bench.
    
    This class manages all configuration settings for the MEQ-Bench application.
    It loads configuration from YAML files and provides methods to access 
    configuration values using dot notation.
    
    The class implements the singleton pattern to ensure that configuration
    is loaded once and shared across the entire application.
    
    Attributes:
        _instance: Singleton instance of the Config class.
        _config: Dictionary containing the loaded configuration data.
    
    Example:
        ```python
        from config import config
        
        # Get configuration values
        model_name = config.get('llm_judge.default_model')
        audiences = config.get_audiences()
        
        # Set up logging
        config.setup_logging()
        ```
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file.
        
        Loads configuration settings from a YAML file and validates required sections.
        If no path is provided, looks for config.yaml in the project root directory.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
                in the project root directory.
                
        Raises:
            ConfigurationError: If configuration file is not found or contains invalid YAML.
        """
        if config_path is None:
            # Look for config.yaml in the project root
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            
            # Validate required sections
            self._validate_config()
            
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
    
    def _validate_config(self) -> None:
        """Validate that required configuration sections exist"""
        required_sections = [
            'app', 'audiences', 'complexity_levels', 'llm_judge', 
            'evaluation', 'scoring', 'logging'
        ]
        
        for section in required_sections:
            if section not in self._config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Retrieves configuration values using a dot-separated key path. For example,
        'llm_judge.default_model' would access config['llm_judge']['default_model'].
        
        Args:
            key: Configuration key using dot notation (e.g., 'app.name', 
                'llm_judge.default_model').
            default: Default value to return if key is not found.
            
        Returns:
            The configuration value at the specified key path, or the default value
            if the key is not found.
            
        Raises:
            ConfigurationError: If key is not found and no default value is provided.
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise ConfigurationError(f"Configuration key not found: {key}")
    
    def get_audiences(self) -> List[str]:
        """Get list of target audiences"""
        return self.get('audiences')
    
    def get_complexity_levels(self) -> List[str]:
        """Get list of complexity levels"""
        return self.get('complexity_levels')
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM judge configuration"""
        return self.get('llm_judge')
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.get('evaluation')
    
    def get_scoring_config(self) -> Dict[str, Any]:
        """Get scoring configuration"""
        return self.get('scoring')
    
    def get_api_config(self, provider: str) -> Dict[str, Any]:
        """
        Get API configuration for a specific provider
        
        Args:
            provider: API provider name (e.g., 'openai', 'anthropic')
        """
        return self.get(f'api.{provider}')
    
    def get_data_path(self) -> str:
        """Get data directory path"""
        return self.get('app.data_path', 'data/')
    
    def get_output_path(self) -> str:
        """Get output directory path"""
        return self.get('app.output_path', 'results/')
    
    def setup_logging(self) -> None:
        """Set up logging based on configuration"""
        import logging.config
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Get logging configuration
        logging_config = self.get('logging')
        
        try:
            logging.config.dictConfig(logging_config)
        except Exception as e:
            # Fallback to basic logging if config fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
            logging.error(f"Failed to configure logging from config: {e}")
    
    def get_environment_variable(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable with optional default
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)
    
    def get_api_key(self, provider: str) -> str:
        """
        Get API key for provider from environment variables
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            API key
            
        Raises:
            ConfigurationError: If API key not found
        """
        env_var_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY'
        }
        
        env_var = env_var_map.get(provider.lower())
        if not env_var:
            raise ConfigurationError(f"Unknown API provider: {provider}")
        
        api_key = self.get_environment_variable(env_var)
        if not api_key:
            raise ConfigurationError(
                f"API key not found. Please set {env_var} environment variable."
            )
        
        return api_key
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self._config = None
        self.load_config()


# Global config instance
config = Config()