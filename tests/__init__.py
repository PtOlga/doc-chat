import os
import sys

# Add project root to Python path for imports in tests
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)