#!/usr/bin/env python3
"""
Fast CPU-optimized inference script for Quantamind Legal AI Agent
This version uses lighter models and optimizations for much faster CPU performance.
"""

import os
import json
import argparse
import re
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import warnings
warnings.filterwarnings("ignore")

class FastCPULegalAIAgent:
    def __init__(self, cpu_only=True):
        """Initialize the fast CPU-optimized Legal AI Agent"""
        self.cpu_only = cpu_only
        self.device = "cpu"
        
        print(f"🚀 Initializing Fast Legal AI Agent (CPU Mode)")
        print(f"Device: {self.device}")
        
        # Force CPU usage and optimize for speed
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.embedder = None
        self.faiss_index = None
        self.clause_metadata = None
        
        # Rule-based patterns for fast clause detection
        self.clause_patterns = {
            'exclusivity': [
                r'exclusive\s+rights?',
                r'shall\s+not\s+compete',
                r'exclusively?\s+distribute',
                r'non-compete',
                r'exclusive\s+dealing',
                r'sole\s+and\s+exclusive',
                r'exclusively?\s+entitled'
            ],
            'termination': [
                r'terminat\w+\s+(?:this\s+)?agreement',
                r'(?:thirty|30|sixty|60|ninety|90)\s+days?\s+(?:written\s+)?notice',
                r'either\s+party\s+may\s+terminat',
                r'upon\s+terminat\w+',
                r'end\s+this\s+agreement',
                r'expir\w+\s+of\s+(?:this\s+)?agreement'
            ],
            'liability': [
                r'liability\s+(?:shall\s+)?(?:not\s+)?exceed',
                r'limitation\s+of\s+liability',
                r'(?:no|not)\s+(?:be\s+)?liable\s+for',
                r'indirect\s+(?:or\s+)?(?:incidental\s+)?(?:or\s+)?consequential\s+damages',
                r'total\s+amount\s+paid',
                r'maximum\s+liability'
            ],
            'governing law': [
                r'governed\s+by\s+(?:and\s+construed\s+in\s+accordance\s+with\s+)?(?:the\s+)?laws?\s+of',
                r'jurisdiction\s+(?:and\s+)?(?:venue\s+)?(?:shall\s+be|lies\s+in)',
                r'courts?\s+of\s+\w+\s+(?:county|state)',
                r'applicable\s+law',
                r'governed\s+by.*laws?'
            ],
            'payment terms': [
                r'(?:net\s+)?(?:thirty|30|sixty|60|ninety|90)\s+days?',
                r'payment\s+(?:shall\s+be\s+)?due',
                r'invoice\s+date',
                r'late\s+(?:payment\s+)?(?:fee|charge)',
                r'(?:1\.5|one\s+and\s+half)\s*%\s*per\s+month',
                r'outstanding\s+balance'
            ],
            'confidentiality': [
                r'confidential\w*\s+information',
                r'non-disclosure',
                r'shall\s+not\s+disclose',
                r'maintain\s+(?:the\s+)?confidentiality',
                r'proprietary\s+information',
                r'third\s+parties\s+without.*consent'
            ]
        }
        
        self._load_components()
    
    def _load_components(self):
        """Load components optimized for speed"""
        try:
            print("🔍 Loading lightweight sentence embedder...")
            # Use the smallest, fastest model for CPU
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            
            # Load search index if available (optional for speed)
            self._load_search_index()
            
            print("✅ All components loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading components: {e}")
            print("📝 Continuing with rule-based analysis only...")
    
    def _load_search_index(self):
        """Load FAISS search index and metadata (optional)"""
        try:
            faiss_path = Path('data/faiss_index')
            metadata_path = Path('data/clause_metadata.json')
            
            if faiss_path.exists() and metadata_path.exists():
                print("📚 Loading search index...")
                self.faiss_index = faiss.read_index(str(faiss_path))
                
                with open(metadata_path, 'r') as f:
                    self.clause_metadata = json.load(f)
                
                print(f"✅ Search index loaded: {len(self.clause_metadata)} clauses")
            else:
                print("⚠️  Search index not found. Using rule-based analysis only.")
                
        except Exception as e:
            print(f"⚠️  Could not load search index: {e}")
            print("📝 Continuing with rule-based analysis...")
    
    def find_clause_by_patterns(self, text, clause_type):
        """Fast rule-based clause detection"""
        clause_type_clean = clause_type.lower().strip()
        patterns = self.clause_patterns.get(clause_type_clean, [])
        
        if not patterns:
            return None, f"No patterns defined for clause type: {clause_type}"
        
        # Convert text to lowercase for matching
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        
        found_sentences = []
        pattern_matches = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Find the sentence containing this match
                match_pos = match.start()
                
                # Split into better sentence boundaries
                better_sentences = re.split(r'[.!?]+|(?:\n\s*){2,}', text)
                
                for sentence in better_sentences:
                    sentence_clean = sentence.strip()
                    if not sentence_clean or len(sentence_clean) < 10:
                        continue
                        
                    sentence_start = text_lower.find(sentence_clean.lower())
                    sentence_end = sentence_start + len(sentence_clean)
                    
                    if sentence_start <= match_pos <= sentence_end:
                        # Take just the relevant sentence, not the whole paragraph
                        if len(sentence_clean) > 200:
                            # If sentence is too long, extract around the match
                            relative_pos = match_pos - sentence_start
                            start_pos = max(0, relative_pos - 50)
                            end_pos = min(len(sentence_clean), relative_pos + 100)
                            sentence_clean = "..." + sentence_clean[start_pos:end_pos] + "..."
                        
                        found_sentences.append(sentence_clean)
                        pattern_matches.append(pattern)
                        break
        
        if found_sentences:
            # Return the most relevant sentence (longest one)
            best_sentence = max(found_sentences, key=len)
            return best_sentence, f"Pattern matched: {pattern_matches[found_sentences.index(best_sentence)]}"
        
        return None, f"No {clause_type} patterns found in the contract"
    
    def search_similar_clauses_fast(self, query, top_k=2):
        """Fast similarity search if index is available"""
        if not self.faiss_index or not self.clause_metadata:
            return []
        
        try:
            # Quick embedding for search
            query_embedding = self.embedder.encode([query])
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.clause_metadata):
                    results.append(self.clause_metadata[idx])
            
            return results[:2]  # Limit to 2 for speed
            
        except Exception as e:
            print(f"⚠️  Fast search error: {e}")
            return []
    
    def generate_analysis(self, clause_text, clause_type, pattern_info=""):
        """Generate fast rule-based analysis"""
        if not clause_text:
            return f"No {clause_type} clause found in the provided contract text."
        
        # Simple but effective analysis based on clause type
        analysis_templates = {
            'exclusivity': f"This exclusivity clause establishes exclusive rights and prevents competing activities. The clause '{clause_text}' creates binding obligations for exclusive arrangements.",
            
            'termination': f"This termination clause defines how the contract can be ended. The provision '{clause_text}' specifies the conditions and procedures for contract termination.",
            
            'liability': f"This liability clause limits financial exposure and defines damage responsibilities. The statement '{clause_text}' establishes liability boundaries and limitations.",
            
            'governing law': f"This governing law clause establishes legal jurisdiction and applicable regulations. The provision '{clause_text}' determines which laws and courts have authority.",
            
            'payment terms': f"This payment clause defines financial obligations and payment schedules. The terms '{clause_text}' establish payment timing and consequences for delays.",
            
            'confidentiality': f"This confidentiality clause protects sensitive information disclosure. The provision '{clause_text}' establishes obligations for information protection."
        }
        
        clause_type_clean = clause_type.lower().strip()
        analysis = analysis_templates.get(clause_type_clean, 
            f"This {clause_type} clause contains important contractual provisions: '{clause_text}'")
        
        return analysis
    
    def analyze_contract_fast(self, contract_text, clause_type):
        """Fast contract analysis using rule-based approach"""
        start_time = time.time()
        print(f"🔍 Analyzing contract for: {clause_type}")
        
        # Step 1: Fast pattern-based detection
        print("🎯 Running pattern-based analysis...")
        clause_text, pattern_info = self.find_clause_by_patterns(contract_text, clause_type)
        
        # Step 2: Optional similarity search (if available and fast)
        similar_clauses = []
        if self.faiss_index and clause_text:
            print("🔍 Quick similarity search...")
            similar_clauses = self.search_similar_clauses_fast(f"{clause_type} {clause_text[:100]}", top_k=2)
        
        # Step 3: Generate analysis
        print("💭 Generating analysis...")
        analysis = self.generate_analysis(clause_text, clause_type, pattern_info)
        
        analysis_time = time.time() - start_time
        
        return {
            'clause_type': clause_type,
            'clause_text': clause_text or f"No {clause_type} clause detected",
            'analysis': analysis,
            'pattern_info': pattern_info,
            'similar_clauses_found': len(similar_clauses),
            'analysis_time': f"{analysis_time:.2f} seconds",
            'method': 'Fast CPU Rule-based Analysis'
        }

def load_sample_contract():
    """Load a sample contract for testing"""
    sample_contracts = {
        'software_license': """
        SOFTWARE LICENSE AGREEMENT
        
        This Agreement grants Company exclusive rights to distribute the Product
        in the Territory during the Term. Company shall not compete with similar
        products or engage in conflicting business activities.
        
        Either party may terminate this Agreement with thirty (30) days written
        notice to the other party. Upon termination, all rights and obligations
        shall cease, except for those provisions that by their nature should survive.
        
        Company's liability under this Agreement shall not exceed the total amount
        paid by Client in the twelve (12) months preceding the claim. In no event
        shall Company be liable for indirect, incidental, or consequential damages.
        
        This Agreement shall be governed by and construed in accordance with the
        laws of the State of California. Any disputes shall be resolved in the
        courts of San Francisco County.
        
        Payment terms are Net 30 days from invoice date. Late payments shall incur
        a service charge of 1.5% per month on the outstanding balance.
        
        Company shall maintain the confidentiality of all Client information and
        shall not disclose such information to third parties without prior written consent.
        """,
        
        'service_agreement': """
        PROFESSIONAL SERVICES AGREEMENT
        
        Provider grants Client exclusive access to the specified services within
        the designated market area. Provider agrees not to offer similar services
        to competing entities during the contract period.
        
        This Agreement may be terminated by either party with sixty (60) days
        written notice. Immediate termination is permitted for material breach.
        
        Provider's maximum liability shall be limited to fees paid in the prior
        six months. No liability exists for consequential or punitive damages.
        
        This contract is governed by New York State law and disputes will be
        handled in Manhattan courts.
        
        Invoices are due within forty-five (45) days. Late fees of 2% monthly
        apply to overdue amounts.
        """
    }
    
    return sample_contracts['software_license']

def main():
    """Main function for fast CPU inference"""
    parser = argparse.ArgumentParser(description="Fast Legal AI Agent - CPU Mode")
    parser.add_argument("--clause-type", required=True, 
                       help="Type of clause to extract (e.g., 'exclusivity', 'termination')")
    parser.add_argument("--contract-file", 
                       help="Path to contract file (optional)")
    parser.add_argument("--use-sample", action="store_true",
                       help="Use built-in sample contract")
    
    args = parser.parse_args()
    
    try:
        start_total = time.time()
        
        # Initialize agent (much faster now)
        agent = FastCPULegalAIAgent(cpu_only=True)
        
        # Get contract text
        if args.contract_file and Path(args.contract_file).exists():
            with open(args.contract_file, 'r') as f:
                contract_text = f.read()
            print(f"📄 Loaded contract from: {args.contract_file}")
        else:
            contract_text = load_sample_contract()
            print("📄 Using sample contract")
        
        # Analyze contract (much faster)
        result = agent.analyze_contract_fast(contract_text, args.clause_type)
        
        total_time = time.time() - start_total
        
        # Display results
        print("\n" + "="*60)
        print("FAST ANALYSIS RESULT (CPU MODE)")
        print("="*60)
        print(f"Clause Type: {result['clause_type']}")
        print(f"Method: {result['method']}")
        print(f"Analysis Time: {result['analysis_time']}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Similar clauses found: {result['similar_clauses_found']}")
        
        if result['clause_text'] != f"No {args.clause_type} clause detected":
            print(f"\nExtracted Clause:")
            print(f'"{result["clause_text"]}"')
        
        print(f"\nAnalysis:")
        print(result['analysis'])
        
        if result['pattern_info']:
            print(f"\nPattern Info: {result['pattern_info']}")
        
        print("="*60)
        
        # Performance note
        print(f"\n💡 Fast CPU Performance:")
        print(f"- Analysis completed in {total_time:.2f} seconds")
        print(f"- Uses rule-based pattern matching for speed")
        print(f"- No large model loading required")
        print(f"- Suitable for real-time analysis")
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()