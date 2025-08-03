import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'agent': {
        'name': 'default_agent',
        'type': 'default_type',
        'params': {
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 10
        }
    },
    'environment': {
        'name': 'default_environment',
        'type': 'default_type',
        'params': {
            'num_agents': 2,
            'num_episodes': 100
        }
    }
}

class ConfigError(Exception):
    """Configuration error"""
    pass

class ConfigType(Enum):
    """Configuration type"""
    DEFAULT = 'default'
    CUSTOM = 'custom'

class Config(ABC):
    """Base configuration class"""
    def __init__(self, name: str, type: ConfigType, params: Dict):
        self.name = name
        self.type = type
        self.params = params

    @abstractmethod
    def validate(self):
        """Validate configuration"""
        pass

    @abstractmethod
    def load(self, config: Dict):
        """Load configuration from dictionary"""
        pass

class AgentConfig(Config):
    """Agent configuration"""
    def __init__(self, name: str, type: ConfigType, params: Dict):
        super().__init__(name, type, params)
        self.validate()

    def validate(self):
        """Validate agent configuration"""
        required_params = ['learning_rate', 'batch_size', 'epochs']
        for param in required_params:
            if param not in self.params:
                raise ConfigError(f"Missing required parameter: {param}")

    def load(self, config: Dict):
        """Load agent configuration from dictionary"""
        self.params['learning_rate'] = config.get('learning_rate', 0.01)
        self.params['batch_size'] = config.get('batch_size', 32)
        self.params['epochs'] = config.get('epochs', 10)

class EnvironmentConfig(Config):
    """Environment configuration"""
    def __init__(self, name: str, type: ConfigType, params: Dict):
        super().__init__(name, type, params)
        self.validate()

    def validate(self):
        """Validate environment configuration"""
        required_params = ['num_agents', 'num_episodes']
        for param in required_params:
            if param not in self.params:
                raise ConfigError(f"Missing required parameter: {param}")

    def load(self, config: Dict):
        """Load environment configuration from dictionary"""
        self.params['num_agents'] = config.get('num_agents', 2)
        self.params['num_episodes'] = config.get('num_episodes', 100)

class ConfigManager:
    """Configuration manager"""
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_file}")
            return DEFAULT_CONFIG

    def save_config(self, config: Dict):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def get_config(self) -> Dict:
        """Get configuration"""
        return self.config

    def update_config(self, config: Dict):
        """Update configuration"""
        self.config.update(config)
        self.save_config(self.config)

    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration"""
        return AgentConfig(self.config['agent']['name'], self.config['agent']['type'], self.config['agent']['params'])

    def get_environment_config(self) -> EnvironmentConfig:
        """Get environment configuration"""
        return EnvironmentConfig(self.config['environment']['name'], self.config['environment']['type'], self.config['environment']['params'])

@contextmanager
def config_context(config_manager: ConfigManager):
    """Context manager for configuration"""
    try:
        yield config_manager.get_config()
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        raise
    finally:
        config_manager.save_config(config_manager.get_config())

def main():
    config_manager = ConfigManager()
    with config_context(config_manager) as config:
        agent_config = config_manager.get_agent_config()
        environment_config = config_manager.get_environment_config()
        logger.info(f"Agent config: {agent_config.params}")
        logger.info(f"Environment config: {environment_config.params}")

if __name__ == '__main__':
    main()