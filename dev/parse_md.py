"""
Optimized Parser for Awesome Spatial VLMs README

Features:
1. Clean institution names (removes underscores)
2. Fetch actual publication dates from arXiv API
3. Extract year from venue strings
4. Export to CSV with detailed metadata

Usage:
    python parse_papers_final.py [--no-arxiv]
    
    --no-arxiv: Skip arXiv API calls for faster parsing
"""
import os
import re
import csv
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict
import time
import argparse
import pandas as pd
def arxiv_times_from_pdf_url(pdf_url: str) -> Optional[Dict]:
    """
    Extract publication dates from arXiv PDF URL using arXiv API.
    
    Args:
        pdf_url: URL to arXiv PDF
        
    Returns:
        Dictionary with arxiv_id, published and updated dates, or None if failed
    """
    try:
        # Extract arXiv id from PDF URL (handles version numbers like v2, v3)
        m = re.search(r"(\d{4}\.\d{5})(v\d+)?", pdf_url)
        if not m:
            return None
        arxiv_id = m.group(1)

        # Call arXiv Atom API
        api = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(api, timeout=15)
        xml_text = response.text

        # Parse Atom XML: entry/published/updated
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        published = entry.findtext("atom:published", default="", namespaces=ns)
        updated = entry.findtext("atom:updated", default="", namespaces=ns)

        # Convert to datetime (ISO8601)
        def parse_dt(s: str) -> datetime:
            s = s.strip().replace("Z", "+00:00")
            return datetime.fromisoformat(s).astimezone(timezone.utc)

        published_dt = parse_dt(published)
        updated_dt = parse_dt(updated)

        return {
            "arxiv_id": arxiv_id,
            "published_v1_utc": published_dt,
            "updated_latest_utc": updated_dt,
            "published_year": published_dt.year,
            "published_date": published_dt.strftime("%Y-%m-%d"),
            "updated_date": updated_dt.strftime("%Y-%m-%d")
        }
    except Exception as e:
        print(f"  ⚠ Warning: Failed to fetch arXiv info from {pdf_url[:50]}...: {str(e)[:80]}")
        return None

def extract_year_from_venue(venue_str: str) -> str:
    """
    Extract year from venue string (e.g., 'CVPR2024' -> '2024').
    """
    year_match = re.search(r'20\d{2}', venue_str)
    if year_match:
        return year_match.group(0)
    return ""

def clean_institution(institution: str) -> str:
    """
    Clean institution name by removing underscores and extra spaces.
    
    Args:
        institution: Raw institution string like '_Peking University_'
        
    Returns:
        Cleaned string like 'Peking University'
    """
    # Remove leading/trailing underscores
    cleaned = institution.strip('_').strip()
    return cleaned

def parse_readme_papers(readme_path, fetch_arxiv_dates=True, df_title = None):
    """
    Parse the Awesome Papers section from README.md and extract paper information.
    
    Args:
        readme_path: Path to README.md file
        fetch_arxiv_dates: Whether to fetch dates from arXiv API (default: True)
    
    Returns:
        List of dictionaries containing paper information
    """
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the Awesome Papers section
    papers_section = re.search(
        r'## 🚀 Awesome Papers(.*?)(?=## 📚 Datasets and Benchmarks|$)',
        content,
        re.DOTALL
    )
    
    if not papers_section:
        print("❌ Could not find Awesome Papers section")
        return []
    
    papers_text = papers_section.group(1)
    
    # Find all paper entries
    # Pattern: [Venue/Year] Title (Institution) [[paper]](url) [[code]](url) [[checkpoint]](url)
    paper_pattern = r'\s*-\s*\[([^\]]+)\]\s+(.+?)\s+\(((?:[^()]+|\([^()]*\))*)\)\s+\[\[paper\]\]\(([^)]+)\)(?:\s+\[\[code\]\]\(([^)]+)\))?(?:\s+\[\[checkpoint\]\]\(([^)]+)\))?'
    papers = []
    matches = re.finditer(paper_pattern, papers_text)
    
    # Find the category for each paper by tracking section headers
    category_pattern = r'<summary><b>([^<]+)</b></summary>'
    categories = list(re.finditer(category_pattern, papers_text))
    
    # Main section pattern
    main_section_pattern = r'###\s+([^\n]+)'
    main_sections = list(re.finditer(main_section_pattern, papers_text))
    
    total_papers = 0
    arxiv_papers = 0
    arxiv_success = 0

    new_paper = []
    for match in matches:
        title = match.group(2).strip()
        if title not in list(df_title):
            new_paper.append(title)
    print("new_paper:", new_paper)
    matches = re.finditer(paper_pattern, papers_text)
    for match in matches:
        total_papers += 1
        venue_year = match.group(1).strip()
        title = match.group(2).strip()
        institution = match.group(3).strip()
        paper_url = match.group(4).strip()
        code_url = match.group(5).strip() if match.group(5) else ""
        checkpoint_url = match.group(6).strip() if match.group(6) else ""
        
        # Clean institution name (remove underscores)
        institution = clean_institution(institution)
        
        # Find the position of this paper
        paper_pos = match.start()
        
        # Find which main section this paper belongs to
        main_section = "Unknown"
        for i, ms in enumerate(main_sections):
            if ms.start() < paper_pos:
                if i + 1 < len(main_sections):
                    if main_sections[i + 1].start() > paper_pos:
                        main_section = ms.group(1).strip()
                        break
                else:
                    main_section = ms.group(1).strip()
        
        # Find which subsection this paper belongs to
        subsection = "Unknown"
        for i, cat in enumerate(categories):
            if cat.start() < paper_pos:
                if i + 1 < len(categories):
                    if categories[i + 1].start() > paper_pos:
                        subsection = cat.group(1).strip()
                        break
                else:
                    subsection = cat.group(1).strip()
        
        # Extract venue from venue_year string
        venue = venue_year
        
        # Try to extract year from venue string first
        year = extract_year_from_venue(venue_year)
        published_date = f"{year}"
        updated_date = f"{year}"
        arxiv_id = ""
        
        # If paper is from arXiv and we want to fetch dates, get the actual publication date
        if fetch_arxiv_dates and 'arxiv.org' in paper_url.lower():
            arxiv_papers += 1
            print(f"📄 [{total_papers}/{len(list(re.finditer(paper_pattern, papers_text)))}] Fetching arXiv info: {title[:50]}...")
            arxiv_info = arxiv_times_from_pdf_url(paper_url)
            if arxiv_info:
                arxiv_success += 1
                year = str(arxiv_info['published_year'])
                published_date = arxiv_info['published_date']
                updated_date = arxiv_info['updated_date']
                arxiv_id = arxiv_info['arxiv_id']
                print(f"  ✓ Got dates: Published {published_date}, Updated {updated_date}")
            
            # Rate limiting: wait a bit to avoid overwhelming arXiv API
            time.sleep(0.3)
        
        papers.append({
            'Main_Section': main_section,
            'Subsection': subsection,
            'Venue': venue,
            'Year': year,
            'Published_Date': published_date,
            'Updated_Date': updated_date,
            'ArXiv_ID': arxiv_id,
            'Title': title,
            'Institution': institution,
            'Paper_URL': paper_url,
            'Code_URL': code_url,
            'Checkpoint_URL': checkpoint_url
        })
    
    print("\n" + "=" * 60)
    print(f"📊 Summary:")
    print(f"  Total papers processed: {total_papers}")
    if fetch_arxiv_dates:
        print(f"  ArXiv papers found: {arxiv_papers}")
        print(f"  ArXiv dates fetched successfully: {arxiv_success}/{arxiv_papers}")
    print("=" * 60)
    
    return papers

def save_to_csv(papers, output_path):
    """
    Save parsed papers to CSV file.
    """
    if not papers:
        print("❌ No papers to save")
        return
    
    fieldnames = ['Main_Section', 'Subsection', 'Venue', 'Year', 
                  'Published_Date', 'Updated_Date', 'ArXiv_ID',
                  'Title', 'Institution', 'Paper_URL', 'Code_URL', 'Checkpoint_URL']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(papers)
    
    print(f"✅ Successfully saved {len(papers)} papers to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Parse Awesome Spatial VLMs README and extract paper information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parse_papers_final.py                    # Full parsing with arXiv API
  python parse_papers_final.py --no-arxiv        # Fast parsing without arXiv API
  python parse_papers_final.py -i custom.md      # Parse custom README file
        """
    )
    parser.add_argument('--no-arxiv', action='store_true',
                       help='Skip arXiv API calls for faster parsing')
    parser.add_argument('-i', '--input', default='README.md',
                       help='Input README file path (default: README.md)')
    parser.add_argument('-o', '--output', default='./dev/awesome_papers.csv',
                       help='Output CSV file path (default: awesome_papers.csv)')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    readme_path = Path(args.input)
    output_path = Path(args.output)
    
    if not readme_path.exists():
        print(f"❌ Error: Input file '{readme_path}' not found!")
        return
    df = pd.read_csv(output_path)
    df_title = df.Title
    print("🚀 Awesome Spatial VLMs Paper Parser")
    print("=" * 60)
    print(f"📂 Input:  {readme_path}")
    print(f"💾 Output: {output_path}")
    print(f"🌐 ArXiv API: {'DISABLED (fast mode)' if args.no_arxiv else 'ENABLED'}")
    print("=" * 60)
    print()
    
    # Parse papers
    papers = parse_readme_papers(readme_path, fetch_arxiv_dates=not args.no_arxiv, df_title=df_title)
    
    if papers:
        print("\n📝 Sample papers:")
        for i, paper in enumerate(papers[:3], 1):
            print(f"\n  {i}. [{paper['Venue']}] {paper['Title'][:60]}...")
            print(f"     🏫 Institution: {paper['Institution']}")
            print(f"     📅 Year: {paper['Year']}")
            if paper['Published_Date']:
                print(f"     📆 Published: {paper['Published_Date']}")
                print(f"     🔄 Updated: {paper['Updated_Date']}")
            if paper['Code_URL']:
                print(f"     💻 Code: Available")
            if paper['Checkpoint_URL']:
                print(f"     🔗 Checkpoint: Available")
    
    print()
    # Save to CSV
    save_to_csv(papers, output_path)
    print()

if __name__ == "__main__":
    main()