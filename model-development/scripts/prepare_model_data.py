"""
Medical Summary Template and Data Preparation for Model Training
Author: Lab Lens Team
Description: Prepares MIMIC-III discharge summaries for BioBART fine-tuning
             Creates structured summaries from extracted clinical sections
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict
from sklearn.model_selection import train_test_split

# Maximum summary length constraints
MAX_SUMMARY_LENGTH = 1000  # Total max (structured + brief)
MAX_BRIEF_SUMMARY_LENGTH = 300  # Brief narrative max

# Summary template with 10 sections (9 structured + 1 brief)
SUMMARY_TEMPLATE = """
PATIENT: {age}-year-old {gender}

DATES: Admitted {admission_date}, Discharged {discharge_date}

ADMISSION: {chief_complaint}

HISTORY: {medical_history}

DIAGNOSIS: {diagnosis}

HOSPITAL COURSE: {hospital_course}

LABS: {lab_results}

MEDICATIONS: {medications}

FOLLOW-UP: {follow_up}

SUMMARY: {brief_summary}
"""

def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text preserving sentence boundaries"""
    if not text or pd.isna(text):
        return "Not documented"
    
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
    
    if last_period > max_chars * 0.7:
        return truncated[:last_period + 1]
    return truncated.rstrip() + '...'

def format_date(date_value) -> str:
    """Format date to YYYY-MM-DD"""
    if pd.isna(date_value):
        return 'Not documented'
    try:
        return pd.to_datetime(date_value).strftime('%Y-%m-%d')
    except:
        date_str = str(date_value)
        return date_str[:10] if len(date_str) >= 10 else 'Not documented'

def generate_brief_summary(row: pd.Series) -> str:
    """Generate brief narrative summary (key takeaways)"""
    diagnosis = row.get('discharge_diagnosis', '')
    hospital_course = row.get('hospital_course', '')
    follow_up = row.get('follow_up', '')
    
    summary_parts = []
    
    # Primary issue
    if pd.notna(diagnosis) and diagnosis and diagnosis != 'Not documented':
        primary_dx = str(diagnosis).split(',')[0].split('\n')[0].strip()
        if len(primary_dx) > 50:
            primary_dx = primary_dx[:50] + '...'
        summary_parts.append(f"Primary issue: {primary_dx}")
    
    # Management approach
    if pd.notna(hospital_course) and hospital_course and hospital_course != 'Not documented':
        course_lower = str(hospital_course).lower()
        if 'surgery' in course_lower or 'surgical' in course_lower or 'procedure' in course_lower:
            summary_parts.append("Managed surgically")
        elif 'medical' in course_lower or 'medication' in course_lower:
            summary_parts.append("Managed medically")
        else:
            summary_parts.append("Clinical management provided")
    
    # Disposition
    if pd.notna(follow_up) and follow_up and follow_up != 'Not documented':
        follow_lower = str(follow_up).lower()
        if 'home' in follow_lower:
            summary_parts.append("Discharged home with follow-up")
        elif 'rehab' in follow_lower or 'facility' in follow_lower:
            summary_parts.append("Transferred to facility")
        else:
            summary_parts.append("Appropriate follow-up arranged")
    else:
        summary_parts.append("Patient discharged")
    
    brief = '. '.join(summary_parts[:3]) + '.' if len(summary_parts) >= 2 else 'Clinical care provided.'
    
    if len(brief) > MAX_BRIEF_SUMMARY_LENGTH:
        brief = brief[:MAX_BRIEF_SUMMARY_LENGTH-3] + '...'
    
    return brief

def create_structured_summary(row: pd.Series) -> str:
    """Create complete structured summary"""
    age = row.get('age_at_admission', 'Unknown')
    gender = row.get('gender', 'Unknown')
    admission_date = format_date(row.get('admittime'))
    discharge_date = format_date(row.get('dischtime'))
    
    chief_complaint = row.get('chief_complaint', 'Not documented')
    medical_history = row.get('past_medical_history', 'Not documented')
    diagnosis = row.get('discharge_diagnosis', 'Not documented')
    hospital_course = row.get('hospital_course', 'Not documented')
    medications = row.get('discharge_medications', 'Not documented')
    follow_up = row.get('follow_up', 'Not documented')
    
    lab_results = row.get('lab_summary', 'Not available')
    if pd.notna(lab_results) and lab_results != 'Not available':
        lab_parts = str(lab_results).split(';')[:3]
        lab_results = '; '.join(lab_parts).strip()
    
    summary = SUMMARY_TEMPLATE.format(
        age=age,
        gender=gender,
        admission_date=admission_date,
        discharge_date=discharge_date,
        chief_complaint=truncate_text(chief_complaint, 60),
        medical_history=truncate_text(medical_history, 80),
        diagnosis=truncate_text(diagnosis, 80),
        hospital_course=truncate_text(hospital_course, 100),
        lab_results=truncate_text(lab_results, 60),
        medications=truncate_text(medications, 100),
        follow_up=truncate_text(follow_up, 60),
        brief_summary=generate_brief_summary(row)
    )
    
    if len(summary) > MAX_SUMMARY_LENGTH:
        summary = summary[:MAX_SUMMARY_LENGTH-3] + '...'
    
    return summary

def prepare_summarization_dataset(input_file: str, output_dir: str) -> Dict[str, str]:
    """Prepare train/val/test datasets"""
    print("="*60)
    print("DATA PREPARATION FOR BIOBART MODEL TRAINING")
    print("="*60)
    print(f"Loading data from: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} total records")
    print(f"Input has {len(df.columns)} columns")
    
    # Filter with OR logic for any section
    print("\nFiltering records...")
    print("Criteria: text > 100 chars AND any clinical section present")
    
    has_diagnosis = (df['discharge_diagnosis'].notna()) & (df['discharge_diagnosis'] != '')
    has_medications = (df['discharge_medications'].notna()) & (df['discharge_medications'] != '')
    has_followup = (df['follow_up'].notna()) & (df['follow_up'] != '')
    has_chief = (df['chief_complaint'].notna()) & (df['chief_complaint'] != '')
    has_history = (df['past_medical_history'].notna()) & (df['past_medical_history'] != '')
    has_course = (df['hospital_course'].notna()) & (df['hospital_course'] != '')
    
    df_valid = df[
        (df['cleaned_text'].notna()) & 
        (df['cleaned_text'].str.len() > 100) &
        (has_diagnosis | has_medications | has_followup | has_chief | has_history | has_course)
    ].copy()
    
    print(f"Filtered to {len(df_valid)} records ({len(df_valid)/len(df)*100:.1f}%)")
    print(f"Removed {len(df) - len(df_valid)} incomplete records")
    
    # Create pairs
    print("\nCreating input-output pairs...")
    df_valid['input_text'] = df_valid['cleaned_text']
    df_valid['target_summary'] = df_valid.apply(create_structured_summary, axis=1)
    
    df_valid['input_length'] = df_valid['input_text'].str.len()
    df_valid['summary_length'] = df_valid['target_summary'].str.len()
    df_valid['compression_ratio'] = df_valid['input_length'] / df_valid['summary_length']
    
    print(f"Created {len(df_valid)} summaries")
    print(f"Avg input: {df_valid['input_length'].mean():.0f} chars")
    print(f"Avg summary: {df_valid['summary_length'].mean():.0f} chars")
    print(f"Compression: {df_valid['compression_ratio'].mean():.1f}x")
    
    # Split datasets
    print("\nSplitting into train/val/test (70/15/15)...")
    train_val, test_df = train_test_split(df_valid, test_size=0.15, random_state=42,
        stratify=df_valid['age_group'] if 'age_group' in df_valid.columns else None)
    train_df, val_df = train_test_split(train_val, test_size=0.176, random_state=42,
        stratify=train_val['age_group'] if 'age_group' in train_val.columns else None)
    
    print(f"Train: {len(train_df)} ({len(train_df)/len(df_valid)*100:.1f}%)")
    print(f"Val: {len(val_df)} ({len(val_df)/len(df_valid)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df_valid)*100:.1f}%)")
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.csv')
    val_file = os.path.join(output_dir, 'validation.csv')
    test_file = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\nSaved datasets to {output_dir}/")
    
    # Save samples
    samples = []
    for idx in range(min(5, len(df_valid))):
        row = df_valid.iloc[idx]
        samples.append({
            'record_id': int(row['hadm_id']),
            'input_length': int(row['input_length']),
            'summary_length': int(row['summary_length']),
            'generated_summary': str(row['target_summary'])
        })
    
    with open(os.path.join(output_dir, 'sample_summaries.json'), 'w') as f:
        json.dump(samples, f, indent=2)
    
    print("Saved sample summaries")
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    
    return {'train': train_file, 'validation': val_file, 'test': test_file}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
        default='data-pipeline/data/processed/processed_discharge_summaries.csv')
    parser.add_argument('--output', type=str, 
        default='model-development/data/model_ready')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input not found: {args.input}")
        exit(1)
    
    prepare_summarization_dataset(args.input, args.output)
    print("\nReady for training!")