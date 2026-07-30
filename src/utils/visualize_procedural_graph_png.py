#!/usr/bin/env python3
"""Helper script to visualize procedural graphs using Graphviz (dot) and export to PNG."""

import argparse
import json
import os
import subprocess
import sys

def json_to_dot(json_data, direction='TB', show_all_relations=False, max_cond_len=25, hide_conditions=False):
  """Converts procedural graph JSON to Graphviz DOT format."""
  nodes = json_data.get('nodes', [])
  edges = json_data.get('edges', [])

  dot_lines = []
  dot_lines.append('digraph G {')
  # Graph attributes for clean, modern look
  dot_lines.append(f'  rankdir={direction};')
  dot_lines.append('  bgcolor="#FFFFFF";')
  dot_lines.append('  pad="0.5";')
  dot_lines.append('  nodesep="0.5";')
  dot_lines.append('  ranksep="0.6";')
  dot_lines.append('  fontname="Helvetica,Arial,sans-serif";')
  
  # Default node attributes
  dot_lines.append('  node [fontname="Helvetica,Arial,sans-serif", fontsize=11, penwidth=2, margin="0.2,0.1"];')
  # Default edge attributes
  dot_lines.append('  edge [fontname="Helvetica,Arial,sans-serif", fontsize=9, color="#455A64", fontcolor="#37474F", penwidth=1.5, arrowsize=0.8];')

  # Define styles
  state_style = 'shape=ellipse, style=filled, fillcolor="#CFD8DC", color="#37474F", fontcolor="#263238"'
  action_style = 'shape=box, style="filled,rounded", fillcolor="#E3F2FD", color="#1565C0", fontcolor="#0D47A1"'
  reasoning_style = 'shape=hexagon, style=filled, fillcolor="#E8F5E9", color="#2E7D32", fontcolor="#1B5E20"'

  # Add nodes
  for node in nodes:
    node_id = node.get('id')
    node_type = node.get('type', 'ACTION')
    display_name = node_id.replace('_', ' ')
    
    if node_type == 'STATE':
      style = state_style
    elif node_type == 'REASONING':
      style = reasoning_style
    else:
      style = action_style
      
    dot_lines.append(f'  "{node_id}" [label="{display_name}", {style}];')

  dot_lines.append('')

  # Add edges
  for edge in edges:
    source = edge.get('source')
    target = edge.get('target')
    relation = edge.get('relation')
    condition = edge.get('condition')
    
    label_parts = []
    edge_attrs = []
    
    # If show_all_relations is True, we show even generic ones like LEADS_TO
    if relation:
      if show_all_relations or relation not in ['LEADS_TO', 'PROVIDES_INPUT_FOR']:
        label_parts.append(relation.replace('_', ' '))
      
    if condition:
      edge_attrs.append('style=dashed')
      if not hide_conditions:
        if len(condition) > max_cond_len:
          # Try to truncate at word boundary if possible, otherwise hard truncate
          truncated = condition[:max_cond_len-3]
          last_space = truncated.rfind(' ')
          if last_space > max_cond_len * 0.6: # Only split at space if it's reasonably long
            truncated = truncated[:last_space]
          label_parts.append(f'[{truncated}...]')
        else:
          label_parts.append(f'[{condition}]')
      
    if label_parts:
      label = '\\n'.join(label_parts)
      edge_attrs.append(f'label="{label}"')
      
    attrs_str = f' [{", ".join(edge_attrs)}]' if edge_attrs else ''
    dot_lines.append(f'  "{source}" -> "{target}"{attrs_str};')

  dot_lines.append('}')
  return '\n'.join(dot_lines)

def main():
  parser = argparse.ArgumentParser(description='Convert Procedural Graph JSON to PNG using Graphviz.')
  parser.add_argument('input_json', help='Path to the procedural graph JSON file')
  parser.add_argument('--output-png', '-o', help='Path to output PNG file')
  parser.add_argument('--direction', '-d', default='TB', choices=['TB', 'BT', 'LR', 'RL'],
                      help='Direction of the flow chart (default: TB)')
  parser.add_argument('--show-all-relations', action='store_true',
                      help='Show all relations on edges (including LEADS_TO)')
  parser.add_argument('--max-cond-len', type=int, default=25,
                      help='Maximum length of condition text before truncation')
  parser.add_argument('--hide-conditions', action='store_true',
                      help='Hide condition text entirely (conditional edges remain dashed)')
  
  args = parser.parse_args()
  
  try:
    with open(args.input_json, 'r') as f:
      data = json.load(f)
  except Exception as e:
    print(f"Error reading JSON file: {e}", file=sys.stderr)
    sys.exit(1)
    
  dot_code = json_to_dot(
      data, 
      direction=args.direction, 
      show_all_relations=args.show_all_relations,
      max_cond_len=args.max_cond_len,
      hide_conditions=args.hide_conditions
  )
  
  # Write temporary dot file
  dot_file_path = args.input_json + '.dot'
  try:
    with open(dot_file_path, 'w') as f:
      f.write(dot_code)
  except Exception as e:
    print(f"Error writing temporary DOT file: {e}", file=sys.stderr)
    sys.exit(1)
    
  # Determine output PNG path
  output_png = args.output_png
  if not output_png:
    output_png = os.path.splitext(args.input_json)[0] + '.png'
    
  # Run dot command
  try:
    subprocess.run(['dot', '-Tpng', dot_file_path, '-o', output_png], check=True)
    print(f"Successfully generated PNG: {output_png}")
  except subprocess.CalledProcessError as e:
    print(f"Error running Graphviz 'dot': {e}", file=sys.stderr)
    sys.exit(1)
  finally:
    # Clean up temporary dot file
    if os.path.exists(dot_file_path):
      os.remove(dot_file_path)

if __name__ == '__main__':
  main()
