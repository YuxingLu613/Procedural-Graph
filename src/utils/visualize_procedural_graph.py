#!/usr/bin/env python3
"""Helper script to visualize procedural graphs using Mermaid.js."""

import argparse
import json
import os
import sys

def json_to_mermaid(json_data, direction='TB'):
  """Converts procedural graph JSON to Mermaid flowchart."""
  nodes = json_data.get('nodes', [])
  edges = json_data.get('edges', [])

  mermaid_lines = []
  mermaid_lines.append(f'flowchart {direction}')

  # Define styles with a clean, professional palette
  mermaid_lines.append('  classDef state fill:#CFD8DC,stroke:#37474F,stroke-width:2px,color:#263238,font-weight:bold;')
  mermaid_lines.append('  classDef action fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1,font-weight:bold;')
  mermaid_lines.append('  classDef reasoning fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20,font-weight:bold;')

  # Add nodes
  for node in nodes:
    node_id = node.get('id')
    node_type = node.get('type', 'ACTION')
    
    # Format display name (replace underscores with spaces for readability)
    display_name = node_id.replace('_', ' ')
    
    # Determine shape based on type
    if node_type == 'STATE':
      shape_open, shape_close = '([', '])'
      node_class = 'state'
    elif node_type == 'REASONING':
      shape_open, shape_close = '{{', '}}'
      node_class = 'reasoning'
    else: # ACTION
      shape_open, shape_close = '[', ']'
      node_class = 'action'
      
    mermaid_lines.append(f'  {node_id}{shape_open}"{display_name}"{shape_close}:::{node_class}')

  mermaid_lines.append('')

  # Add edges
  for edge in edges:
    source = edge.get('source')
    target = edge.get('target')
    relation = edge.get('relation')
    condition = edge.get('condition')
    
    # Determine edge label (keep it minimal)
    label_parts = []
    
    # We omit common/generic relations to reduce noise
    if relation and relation not in ['LEADS_TO', 'PROVIDES_INPUT_FOR']:
      # Convert RELATION_NAME to friendly text if needed, or just keep it
      label_parts.append(relation.replace('_', ' '))
      
    if condition:
      label_parts.append(f'{condition}')
      
    if label_parts:
      label = ' | '.join(label_parts)
      mermaid_lines.append(f'  {source} -->|"{label}"| {target}')
    else:
      mermaid_lines.append(f'  {source} --> {target}')

  return '\n'.join(mermaid_lines)

def main():
  parser = argparse.ArgumentParser(description='Convert Procedural Graph JSON to Mermaid.')
  parser.add_argument('input_json', help='Path to the procedural graph JSON file')
  parser.add_argument('--output', '-o', help='Path to output file (markdown or mermaid)')
  parser.add_argument('--direction', '-d', default='TB', choices=['TB', 'BT', 'LR', 'RL'],
                      help='Direction of the flow chart (default: TB)')
  
  args = parser.parse_args()
  
  try:
    with open(args.input_json, 'r') as f:
      data = json.load(f)
  except Exception as e:
    print(f"Error reading JSON file: {e}", file=sys.stderr)
    sys.exit(1)
    
  mermaid_code = json_to_mermaid(data, direction=args.direction)
  
  if args.output:
    try:
      with open(args.output, 'w') as f:
        if args.output.endswith('.md'):
          f.write(f"```mermaid\n{mermaid_code}\n```\n")
        else:
          f.write(mermaid_code)
      print(f"Successfully wrote visualization to {args.output}")
    except Exception as e:
      print(f"Error writing output file: {e}", file=sys.stderr)
      sys.exit(1)
  else:
    print(mermaid_code)

if __name__ == '__main__':
  main()
