"""
JSON schema validation utilities.

Validates all artifact files against their schemas.
"""

import json
import jsonschema
from pathlib import Path
from typing import Dict


def load_schema(schema_name: str) -> Dict:
    """
    Load JSON schema from schemas directory.
    
    Args:
        schema_name: Schema filename (e.g., "config.schema.json")
    
    Returns:
        Schema dictionary
    """
    schema_path = Path(__file__).parent.parent / "schemas" / schema_name
    with open(schema_path, 'r') as f:
        return json.load(f)


def validate_json(data: Dict, schema_name: str):
    """
    Validate JSON data against schema.
    
    Args:
        data: Data to validate
        schema_name: Schema filename
    
    Raises:
        jsonschema.ValidationError: If validation fails
    """
    schema = load_schema(schema_name)
    jsonschema.validate(instance=data, schema=schema)


def validate_config(config_path: str):
    """Validate config.json against schema"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    validate_json(config, "config.schema.json")


def validate_splits(splits_path: str):
    """Validate splits.json against schema"""
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    validate_json(splits, "splits.schema.json")


def validate_thresholds(thresholds_path: str):
    """Validate thresholds.json against schema"""
    with open(thresholds_path, 'r') as f:
        thresholds = json.load(f)
    validate_json(thresholds, "thresholds.schema.json")


def validate_gateparams(gateparams_path: str):
    """Validate gateparams.json against schema"""
    with open(gateparams_path, 'r') as f:
        gateparams = json.load(f)
    validate_json(gateparams, "gateparams.schema.json")


def validate_scrcparams(scrcparams_path: str):
    """Validate scrcparams.json against schema"""
    with open(scrcparams_path, 'r') as f:
        scrcparams = json.load(f)
    validate_json(scrcparams, "scrcparams.schema.json")


def validate_bundle(bundle_path: str):
    """Validate bundle.json against schema"""
    with open(bundle_path, 'r') as f:
        bundle = json.load(f)
    validate_json(bundle, "bundle.schema.json")
