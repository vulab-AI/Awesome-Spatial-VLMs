import re
import csv
from typing import List, Dict

def parse_markdown_table(md_content: str) -> List[Dict[str, str]]:
    """
    Parse Markdown table content
    """
    # Extract table section
    table_match = re.search(r'<table.*?>(.*?)</table>', md_content, re.DOTALL)
    if not table_match:
        raise ValueError("Table content not found")
    
    table_html = table_match.group(1)
    
    # Extract table body
    tbody_match = re.search(r'<tbody>(.*?)</tbody>', table_html, re.DOTALL)
    if not tbody_match:
        raise ValueError("Table body not found")
    
    tbody_html = tbody_match.group(1)
    rows = re.findall(r'<tr.*?>(.*?)</tr>', tbody_html, re.DOTALL)
    
    data = []
    for row in rows:
        cells = re.findall(r'<td.*?>(.*?)</td>', row, re.DOTALL)
        
        if len(cells) >= 9:
            row_dict = parse_row(cells)
            data.append(row_dict)
    
    return data

def parse_row(cells: List[str]) -> Dict[str, str]:
    """
    Parse a single row of data
    """
    # Parse first column: Dataset name and link
    dataset_name, dataset_url = extract_link(cells[0])
    
    # Parse cognitive levels (can contain multiple tags)
    cognitive_levels = []
    if '✓' in cells[2]:
        cognitive_levels.append('Perception')
    if '✓' in cells[3]:
        cognitive_levels.append('Understanding')
    if '✓' in cells[4]:
        cognitive_levels.append('Extrapolation')
    
    # Join multiple levels with semicolon
    cognitive_level_str = '; '.join(cognitive_levels) if cognitive_levels else ''
    
    row_dict = {
        'Dataset': dataset_name,
        'Venue': clean_text(cells[1]),
        'Cognitive Level': cognitive_level_str,
        'Fundamental Task': clean_text(cells[5]),
        'Size': clean_text(cells[6]),
        'Image Source': clean_text(cells[7]),
        'Modality': clean_text(cells[8]),
        'Link': dataset_url,
    }
    
    return row_dict

def extract_link(cell: str) -> tuple:
    """
    Extract link text and URL
    """
    link_match = re.search(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', cell, re.DOTALL)
    if link_match:
        url = link_match.group(1)
        text = clean_text(link_match.group(2))
        return text, url
    else:
        return clean_text(cell), ''

def clean_text(text: str) -> str:
    """
    Clean text content by removing HTML tags and extra whitespace
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

def save_to_csv(data: List[Dict[str, str]], output_file: str):
    """
    Save data to CSV file
    """
    if not data:
        print("No data to save")
        return
    
    # Define column order
    headers = [
        'Dataset',
        'Venue',
        'Cognitive Level',
        'Fundamental Task',
        'Size',
        'Image Source',
        'Modality',
        'Link',
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Successfully saved {len(data)} records to {output_file}")

def print_statistics(data: List[Dict[str, str]]):
    """
    Print statistics about the dataset
    """
    print(f"\nTotal datasets: {len(data)}")
    
    # Count cognitive level distribution
    perception_count = sum(1 for d in data if 'Perception' in d['Cognitive Level'])
    understanding_count = sum(1 for d in data if 'Understanding' in d['Cognitive Level'])
    extrapolation_count = sum(1 for d in data if 'Extrapolation' in d['Cognitive Level'])
    
    # Count datasets with multiple levels
    multi_level = sum(1 for d in data if d['Cognitive Level'].count(';') > 0)
    
    print(f"\nCognitive Level Distribution:")
    print(f"  Perception: {perception_count} datasets")
    print(f"  Understanding: {understanding_count} datasets")
    print(f"  Extrapolation: {extrapolation_count} datasets")
    print(f"  Multi-level: {multi_level} datasets")
    
    # Count datasets with links
    with_url = sum(1 for d in data if d['Link'])
    without_url = len(data) - with_url
    print(f"\nLink Statistics:")
    print(f"  With link: {with_url} datasets")
    print(f"  Without link: {without_url} datasets")

def main(input_file: str = None, output_file: str = None):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Parse table
        data = parse_markdown_table(md_content)
        
        # Save to CSV
        save_to_csv(data, output_file)
        
        # Print statistics
        print_statistics(data)
        
        # Display first 5 examples
        print("\nFirst 5 datasets:")
        for i, row in enumerate(data[:5], 1):
            print(f"\n--- Dataset {i} ---")
            print(f"Name: {row['Dataset']}")
            print(f"Link: {row['Link'][:50] + '...' if len(row['Link']) > 50 else row['Link']}")
            print(f"Venue: {row['Venue']}")
            print(f"Cognitive Level: {row['Cognitive Level']}")
            print(f"Task: {row['Fundamental Task'][:50] + '...' if len(row['Fundamental Task']) > 50 else row['Fundamental Task']}")
        
    except FileNotFoundError:
        print(f"Error: File not found - {input_file}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    input_file = '../data_benchmark/Benchmark_SVQA.md'
    output_file = 'spatial_benchmarks.csv'
    main(input_file, output_file)
    
    input_file = '../data_benchmark/Dataset_SVQA.md'
    output_file = 'spatial_datasets.csv'
    main(input_file, output_file)