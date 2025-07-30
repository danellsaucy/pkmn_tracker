# config.py - Configuration file for Pokemon Card Manager

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = PROJECT_ROOT / "card_images"

# Database Configuration
DATABASE_CONFIG = {
    # SQLite configuration (default)
    'sqlite': {
        'type': 'sqlite',
        'path': str(DATA_DIR / 'pokemon_cards.db')
    },
    
    # PostgreSQL configuration (for when you upgrade)
    'postgresql': {
        'type': 'postgresql',
        'host': '35.95.111.21',
        'database': 'pokemoncards',
        'user': 'pokemon',
        'password': 'pokemon',
        'port': 5432
    }
}

# Card Scanner Configuration
CARD_SCANNER_CONFIG = {
    'config_file': r'C:\Users\daforbes\Desktop\projects\models\mask\pointmask_transforms\my_config.py',
    'checkpoint_file': r'C:\Users\daforbes\Desktop\projects\models\mask\pointmask_transforms\best_coco_segm_mAP_epoch_99.pth',
    'hash_dict_path': r"C:\Users\daforbes\Desktop\projects\tcg_scanner\raw\card_hashes.json",
    'reference_cards_path': "downloaded_cards"
}

# Application Settings
APP_SETTINGS = {
    'price_refresh_hours': 24,  # Refresh prices older than 24 hours
    'max_concurrent_scans': 5,  # For batch processing
    'backup_enabled': True,
    'backup_interval_days': 7,
    'log_level': 'INFO'
}

# File paths
PATHS = {
    'temp_dir': DATA_DIR / 'temp',
    'exports_dir': DATA_DIR / 'exports',
    'logs_dir': DATA_DIR / 'logs',
    'backups_dir': DATA_DIR / 'backups'
}

# Create directories if they don't exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Which database to use (change this to switch between SQLite and PostgreSQL)
ACTIVE_DATABASE = 'postgresql'  # Change to 'postgresql' when ready

def get_active_db_config():
    """Get the active database configuration"""
    return DATABASE_CONFIG[ACTIVE_DATABASE]

def get_export_path(filename: str = None) -> str:
    """Get path for export files"""
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pokemon_cards_export_{timestamp}.csv"
    
    return str(PATHS['exports_dir'] / filename)

def get_log_path() -> str:
    """Get path for log files"""
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    return str(PATHS['logs_dir'] / f"pokemon_cards_{date_str}.log")

# Environment-specific overrides
if os.getenv('POKEMON_ENV') == 'production':
    # Production overrides
    DATABASE_CONFIG['postgresql']['host'] = os.getenv('DB_HOST', 'localhost')
    DATABASE_CONFIG['postgresql']['database'] = os.getenv('DB_NAME', 'pokemon_cards')
    DATABASE_CONFIG['postgresql']['user'] = os.getenv('DB_USER', 'postgres')
    DATABASE_CONFIG['postgresql']['password'] = os.getenv('DB_PASSWORD', '')
    ACTIVE_DATABASE = 'postgresql'
    APP_SETTINGS['log_level'] = 'WARNING'

elif os.getenv('POKEMON_ENV') == 'development':
    # Development overrides
    APP_SETTINGS['log_level'] = 'DEBUG'
    APP_SETTINGS['price_refresh_hours'] = 1  # More frequent refreshes for testing