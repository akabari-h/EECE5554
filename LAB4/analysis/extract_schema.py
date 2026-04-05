#!/usr/bin/env python3
"""
Extract message schemas from MCAP file to understand
the exact field structure of lab4_pkg messages.
"""
from mcap.reader import make_reader
from pathlib import Path

bag_path = Path.home() / 'EECE5554/LAB4/data/data_going_in_circles'
mcap_files = list(bag_path.glob('*.mcap'))

print(f"Reading: {mcap_files[0]}")

with open(mcap_files[0], 'rb') as f:
    reader = make_reader(f)
    summary = reader.get_summary()

    print("\n=== SCHEMAS ===")
    for schema_id, schema in summary.schemas.items():
        print(f"\nSchema ID: {schema_id}")
        print(f"Name: {schema.name}")
        print(f"Encoding: {schema.encoding}")
        print(f"Data:\n{schema.data.decode('utf-8', errors='ignore')}")
