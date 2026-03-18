import csv
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import hashlib


# Language ID mapping for model training
LANGUAGE_TO_ID = {
    'en': 0,
    'de': 1,
    'sv': 2,
    'unknown': 3
}

ID_TO_LANGUAGE = {v: k for k, v in LANGUAGE_TO_ID.items()}


@dataclass
class ParallelSample:
    """Represents a parallel translation sample."""
    source_text: str          # Original runic transliteration
    target_translation: str   # Translation (Swedish, English, German, etc.)
    source_name: str          # Identifier/name of the inscription
    source_file: str          # Original source file
    language: str             # Target language (sv, en, de, etc.)
    language_id: int          # Numeric language ID for model training
    metadata: Dict            # Additional metadata


def get_file_hash(text: str) -> str:
    """Generate hash for deduplication."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def normalize_text(text: Optional[str]) -> str:
    """Normalize text by stripping whitespace and handling None values."""
    if text is None:
        return ""
    return str(text).strip()


def detect_language(translation: str) -> str:
    """Detect language based on common words."""
    if not translation:
        return "unknown"
    
    translation_lower = translation.lower()
    
    # Swedish indicators
    swedish_words = ['och', 'som', 'den', 'det', 'hans', 'här', 'efter', 'sten', 'gud', 'lät', 'göra', 'minne']
    # English indicators
    english_words = ['and', 'the', 'this', 'his', 'her', 'stone', 'god', 'made', 'memory', 'in', 'of', 'for']
    # German indicators
    german_words = ['und', 'der', 'die', 'das', 'sein', 'hier', 'nach', 'stein', 'gott', 'gemacht']
    
    swedish_count = sum(1 for word in swedish_words if word in translation_lower.split())
    english_count = sum(1 for word in english_words if word in translation_lower.split())
    german_count = sum(1 for word in german_words if word in german_words if word in translation_lower.split())
    
    if swedish_count >= english_count and swedish_count >= german_count and swedish_count > 0:
        return "sv"
    elif english_count >= german_count and english_count > 0:
        return "en"
    elif german_count > 0:
        return "de"
    
    return "unknown"


def get_language_id(language: str) -> int:
    """Get numeric ID for language."""
    return LANGUAGE_TO_ID.get(language, LANGUAGE_TO_ID['unknown'])


def process_christerhamp(filepath: str) -> List[ParallelSample]:
    """Process christerhamp_gamla_runor.csv format."""
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            transliteration = normalize_text(row.get('transliteration', ''))
            translation = normalize_text(row.get('translation', ''))
            
            # Skip if no transliteration or translation
            if not transliteration or not translation or translation == '-':
                continue
            
            language = detect_language(translation)
            sample = ParallelSample(
                source_text=transliteration,
                target_translation=translation,
                source_name=normalize_text(row.get('name', '')),
                source_file='christerhamp_gamla_runor.csv',
                language=language,
                language_id=get_language_id(language),
                metadata={
                    'stone_id': normalize_text(row.get('stone_id', '')),
                    'page_url': normalize_text(row.get('page_url', '')),
                    'image_url': normalize_text(row.get('image_url', ''))
                }
            )
            samples.append(sample)
    
    return samples


def process_srd_parallel(filepath: str) -> List[ParallelSample]:
    """Process srd_parallel.csv format."""
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            transliteration = normalize_text(row.get('transliteration', ''))
            translation = normalize_text(row.get('english', ''))
            
            # Skip if no transliteration or translation
            if not transliteration or not translation:
                continue
            
            language = 'en'
            sample = ParallelSample(
                source_text=transliteration,
                target_translation=translation,
                source_name=normalize_text(row.get('signum', '')),
                source_file='srd_parallel.csv',
                language=language,
                language_id=get_language_id(language),
                metadata={
                    'signum': normalize_text(row.get('signum', ''))
                }
            )
            samples.append(sample)
    
    return samples


def process_runesdb(filepath: str) -> List[ParallelSample]:
    """Process runes_parsed_parallel_1_10k.csv format (RunesDB)."""
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            transliteration = normalize_text(row.get('transliteration', ''))
            translation = normalize_text(row.get('translation', ''))
            
            # Skip if no transliteration or translation
            if not transliteration or not translation or translation == '-':
                continue
            
            # RunesDB translations are typically in German
            language = 'de'
            sample = ParallelSample(
                source_text=transliteration,
                target_translation=translation,
                source_name=normalize_text(row.get('find_id', '')) or normalize_text(row.get('url', '')),
                source_file='runes_parsed_parallel_1_10k.csv',
                language=language,
                language_id=get_language_id(language),
                metadata={
                    'url': normalize_text(row.get('url', '')),
                    'find_id': normalize_text(row.get('find_id', '')),
                    'runerow': normalize_text(row.get('runerow', '')),
                    'findplace': normalize_text(row.get('findplace', '')),
                    'country': normalize_text(row.get('country', '')),
                    'object_class': normalize_text(row.get('object_class', '')),
                    'dating': normalize_text(row.get('dating', '')),
                    'inscription': normalize_text(row.get('inscription', '')),
                    'images_urls': normalize_text(row.get('images_urls', ''))
                }
            )
            samples.append(sample)
    
    return samples


def deduplicate_samples(samples: List[ParallelSample]) -> List[ParallelSample]:
    """Remove duplicate samples based on source text hash."""
    seen_hashes = set()
    unique_samples = []
    
    for sample in samples:
        # Create hash from source text + translation for deduplication
        hash_input = f"{sample.source_text}|||{sample.target_translation}"
        hash_value = get_file_hash(hash_input)
        
        if hash_value not in seen_hashes:
            seen_hashes.add(hash_value)
            unique_samples.append(sample)
    
    return unique_samples


def save_to_csv(samples: List[ParallelSample], output_path: str):
    """Save samples to CSV file."""
    if not samples:
        print("No samples to save.")
        return
    
    fieldnames = ['source_text', 'target_translation', 'source_name', 'source_file', 'language', 'language_id']
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        for sample in samples:
            writer.writerow({
                'source_text': sample.source_text,
                'target_translation': sample.target_translation,
                'source_name': sample.source_name,
                'source_file': sample.source_file,
                'language': sample.language,
                'language_id': sample.language_id
            })
    
    print(f"Saved {len(samples)} samples to {output_path}")


def save_detailed_json(samples: List[ParallelSample], output_path: str):
    """Save samples with full metadata to JSON file."""
    import json
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(s) for s in samples], f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(samples)} detailed samples to {output_path}")


def collect_parallel_data(
    parsers_dir: str = "parsers",
    output_csv: str = "parallel_corpus.csv",
    output_json: str = "parallel_corpus_detailed.json",
    deduplicate: bool = True
) -> List[ParallelSample]:
    """
    Main function to collect parallel translation data from all parser outputs.
    
    Args:
        parsers_dir: Directory containing parsed CSV files
        output_csv: Path for output CSV file (simple format)
        output_json: Path for output JSON file (with metadata)
        deduplicate: Whether to remove duplicate entries
    
    Returns:
        List of collected ParallelSample objects
    """
    all_samples = []
    parsers_path = Path(parsers_dir)
    
    # Process each known file format
    file_processors = {
        'christerhamp_gamla_runor.csv': process_christerhamp,
        'srd_parallel.csv': process_srd_parallel,
        'runes_parsed_parallel_1_10k.csv': process_runesdb,
    }
    
    for filename, processor in file_processors.items():
        filepath = parsers_path / filename
        if filepath.exists():
            print(f"Processing {filename}...")
            samples = processor(str(filepath))
            print(f"  Found {len(samples)} valid samples")
            all_samples.extend(samples)
        else:
            print(f"  Warning: {filename} not found")
    
    # Deduplicate if requested
    if deduplicate:
        print(f"\nDeduplicating samples...")
        original_count = len(all_samples)
        all_samples = deduplicate_samples(all_samples)
        removed_count = original_count - len(all_samples)
        print(f"  Removed {removed_count} duplicates")
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"Total samples collected: {len(all_samples)}")
    
    # Language distribution
    lang_counts = {}
    for sample in all_samples:
        lang_counts[sample.language] = lang_counts.get(sample.language, 0) + 1
    
    print("\nLanguage distribution:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")
    
    # Source file distribution
    source_counts = {}
    for sample in all_samples:
        source_counts[sample.source_file] = source_counts.get(sample.source_file, 0) + 1
    
    print("\nSource distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    
    # Save outputs
    save_to_csv(all_samples, output_csv)
    save_detailed_json(all_samples, output_json)
    
    return all_samples


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect parallel translation data from parsed rune inscriptions'
    )
    parser.add_argument(
        '--parsers-dir',
        default='parsers',
        help='Directory containing parsed CSV files (default: parsers)'
    )
    parser.add_argument(
        '--output-csv',
        default='parallel_corpus.csv',
        help='Output CSV file path (default: parallel_corpus.csv)'
    )
    parser.add_argument(
        '--output-json',
        default='parallel_corpus_detailed.json',
        help='Output JSON file path with metadata (default: parallel_corpus_detailed.json)'
    )
    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable deduplication'
    )
    
    args = parser.parse_args()
    
    collect_parallel_data(
        parsers_dir=args.parsers_dir,
        output_csv=args.output_csv,
        output_json=args.output_json,
        deduplicate=not args.no_dedup
    )
