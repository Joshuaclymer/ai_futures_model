"""
Flask API for the AI Futures Simulator.

This API provides endpoints for running simulations and retrieving configuration.
"""

import sys
from pathlib import Path

# Add directories to path for imports
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent))

from flask import Flask
from flask_cors import CORS
import logging

from endpoints import register_all_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Register all routes
register_all_routes(app)


if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5329)
    args = parser.parse_args()
    # Use PORT env var if set (for Render), otherwise use CLI arg or default
    port = int(os.environ.get('PORT', args.port))
    # Bind to 0.0.0.0 for production, enable debug only in development
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
