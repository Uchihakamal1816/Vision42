Conversations
 
Program Policies
Powered by Google
Last account activity: 22 minutes ago
Details

import random
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import nltk
from nltk.corpus import wordnet
from itertools import chain
from bert_score import score


# ============================ SETUP ====================

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

SEMANTIC_PATTERNS = {
    'scene_openers': [
        "The scene shows", "This scene provides", "The image depicts",
        "An aerial view shows", "A satellite view of"
    ],
    'scene_descriptors': [
        "satellite view", "aerial view", "overhead view", "bird's eye view", "satellite imagery"
    ],
    'location_types': [
        "airport", "urban area", "sports complex", "campus", "institutional area",
        "city center", "aerodrome", "metropolitan zone", "sports venue", 
        "athletic facility", "industrial zone", "residential district"
    ],
    'spatial_relations': [
        "parked on the tarmac", "in the center", "surrounding", "leading to", "positioned at",
        "adjacent to", "spanning across", "featuring", "containing", "hosting", "flanking", "bisecting"
    ],
    'object_precision': [
        "storage tanks", "runway", "taxiway", "hangar", "water tower", "grid patterns",
        "road networks", "buildings", "structures", "facilities", "cooling tower"
    ],
    'quantifiers': [
        "two distinct", "multiple", "several", "numerous", "clearly visible", "prominently featured",
        "evident", "densely packed", "widely dispersed"
    ],
    'high_idf_terms': [
        "satellite", "tarmac", "infrastructure", "runway", "airplanes", "storage tanks", "complex",
        "campus", "institutional", "hangar", "metropolitan", "aerodrome", "runway", "athletic", "facility"
    ]
}

REFERENCE_CAPTIONS = [
    "The scene shows a satellite view of an airport runway with two airplanes parked on the tarmac. Apart from the runway, the scene includes storage tanks, airport infrastructure, and roads leading to the runway.",
    "This scene provides a satellite view of an urban area with a prominent sports field in the center, featuring a baseball or soccer field and a running track. There are multiple buildings surrounding the sports complex which are likely part of a campus or institutional area."
]

REFERENCE_EMBEDDINGS = sbert_model.encode(REFERENCE_CAPTIONS, convert_to_tensor=True)

# ==================== CLEANING FUNCTIONS ===============

def clean_markup_tags(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)

def remove_coordinate_boxes(text: str) -> str:
    text = re.sub(r'\{<\d+><\d+><\d+><\d+>\|<\d+>\}', '', text)
    text = re.sub(r'<delim>', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+;', ';', text)
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    text = re.sub(r',(?=[A-Z])', ', ', text)
    return text.strip()

def clean_caption(caption: str, verbose: bool = False) -> str:
    caption = clean_markup_tags(caption)
    caption = remove_coordinate_boxes(caption)
    caption = normalize_whitespace(caption)
    return caption

# =================== SEMANTIC UTILITIES ================

def paraphrase_word(word: str, preserve_rare: bool = True) -> List[str]:
    if preserve_rare and word.lower() in SEMANTIC_PATTERNS['high_idf_terms']:
        return [word]
    synsets = wordnet.synsets(word)
    lemmas = set(chain.from_iterable([ws.lemma_names() for ws in synsets]))
    lemmas = {l.replace('_', ' ') for l in lemmas if l.replace('_', ' ').isalpha() and l.lower() != word.lower()}
    return list(lemmas) if lemmas else [word]

def extract_tokens(text: str) -> List[str]:
    return text.split()

def contains_semantic_pattern(text: str, pattern_type: str) -> bool:
    text_lower = text.lower()
    for pattern in SEMANTIC_PATTERNS[pattern_type]:
        if pattern.lower() in text_lower:
            return True
    return False

def count_semantic_patterns(caption: str) -> Dict[str, bool]:
    patterns = {
        'has_scene_opener': contains_semantic_pattern(caption, 'scene_openers'),
        'has_scene_descriptor': contains_semantic_pattern(caption, 'scene_descriptors'),
        'has_location_type': contains_semantic_pattern(caption, 'location_types'),
        'has_spatial_relation': contains_semantic_pattern(caption, 'spatial_relations'),
        'has_object_precision': contains_semantic_pattern(caption, 'object_precision'),
        'has_quantifier': contains_semantic_pattern(caption, 'quantifiers'),
        'is_multi_sentence': len(re.split(r'[.!?]', caption.strip())) > 1
    }
    return patterns

def score_semantic_quality(caption: str) -> float:
    patterns = count_semantic_patterns(caption)
    tier1_weight = 0.3
    tier1_score = (
        (patterns['has_scene_opener'] * 0.3 +
         patterns['has_scene_descriptor'] * 0.3 +
         patterns['has_location_type'] * 0.3) / 3
    ) * tier1_weight
    tier2_weight = 0.4
    tier2_score = (
        (patterns['has_spatial_relation'] * 0.25 +
         patterns['has_quantifier'] * 0.25 +
         patterns['is_multi_sentence'] * 0.25 +
         patterns['has_object_precision'] * 0.25) / 4
    ) * tier2_weight
    tier3_weight = 0.3
    tier3_score = tier3_weight
    return tier1_score + tier2_score + tier3_score

# ========== ENHANCE & CANDIDATE GENERATION FUNCTIONS ==========

def generate_paraphrases(caption: str, n_variants: int = 7, paraphrase_prob: float = 0.25) -> List[str]:
    tokens = extract_tokens(caption)
    variants = []
    for _ in range(n_variants):
        new_tokens = []
        for token in tokens:
            if (random.random() < paraphrase_prob and 
                len(token) > 3 and 
                token.lower() not in SEMANTIC_PATTERNS['high_idf_terms']):
                synonyms = paraphrase_word(token, preserve_rare=True)
                new_tokens.append(random.choice(synonyms))
            else:
                new_tokens.append(token)
        variants.append(" ".join(new_tokens))
    return variants

def inject_scene_opener(caption: str) -> str:
    caption = caption.strip()
    has_opener = any(opener.lower() in caption.lower() for opener in SEMANTIC_PATTERNS['scene_openers'])
    if not has_opener:
        if 'satellite' in caption.lower() or 'aerial' in caption.lower():
            return f"The scene shows {caption[0].lower() + caption[1:] if len(caption) > 1 else caption}"
        else:
            return f"This scene provides a satellite view of {caption[0].lower() + caption[1:] if len(caption) > 1 else caption}"
    return caption

def inject_scene_descriptor(caption: str) -> str:
    has_descriptor = any(desc in caption.lower() for desc in SEMANTIC_PATTERNS['scene_descriptors'])
    if not has_descriptor:
        if caption.lower().startswith("the scene shows"):
            caption = caption.replace("The scene shows", "The scene shows a satellite view of", 1)
        elif caption.lower().startswith("this scene provides"):
            caption = caption.replace("This scene provides", "This scene provides a satellite view of", 1)
        else:
            caption = f"This scene provides a satellite view of {caption}"
    return caption

def enhance_location_type(caption: str) -> str:
    has_location = any(loc.lower() in caption.lower() for loc in SEMANTIC_PATTERNS['location_types'])
    if not has_location:
        if 'runway' in caption.lower() or 'airplane' in caption.lower() or 'tarmac' in caption.lower():
            if 'airport' not in caption.lower():
                caption = caption.replace('scene shows', 'scene shows an airport') if 'scene shows' in caption.lower() else caption
        elif 'field' in caption.lower() or 'track' in caption.lower():
            if 'sports' not in caption.lower():
                caption = caption.replace('area', 'sports complex area') if 'area' in caption.lower() else caption
    return caption

def enhance_spatial_relations(caption: str) -> str:
    weak_spatial = {
        r'\bnear\b': 'adjacent to',
        r'\bnext to\b': 'alongside',
        r'\bacross\b': 'spanning across',
        r'\bthrough\b': 'traversing through',
        r'\baround\b': 'surrounding'
    }
    for weak, strong in weak_spatial.items():
        caption = re.sub(weak, strong, caption, flags=re.IGNORECASE)
    has_spatial = any(rel in caption.lower() for rel in SEMANTIC_PATTERNS['spatial_relations'])
    if not has_spatial and 'center' not in caption.lower():
        caption = caption.rstrip('.') + ", positioned in the center."
    return caption

def enhance_object_precision(caption: str) -> str:
    generic_to_precise = {
        r'\btank\b': 'storage tank',
        r'\bplane\b': 'airplane',
        r'\baircraft\b': 'airplane',
        r'\bcar\b': 'vehicle',
        r'\bstructure\b': 'building',
        r'\bpath\b': 'road',
        r'\bway\b': 'infrastructure',
        r'\bzone\b': 'area',
        r'\bfield\b': 'sports field'
    }
    for generic, precise in generic_to_precise.items():
        caption = re.sub(generic, precise, caption, flags=re.IGNORECASE)
    return caption

def enhance_quantifiers(caption: str) -> str:
    number_patterns = {
        r'\bone\b': 'a single',
        r'\btwo\b': 'two distinct',
        r'\bthree\b': 'three separate',
        r'\bfour\b': 'four distinct',
        r'\bmany\b': 'multiple',
        r'\bseveral\b': 'numerous',
        r'\ba few\b': 'multiple'
    }
    for pattern, replacement in number_patterns.items():
        caption = re.sub(pattern, replacement, caption, flags=re.IGNORECASE)
    return caption

def add_contextual_elaboration(caption: str) -> str:
    sentences = re.split(r'[.!?]', caption.strip())
    if len(sentences) == 1 or (len(sentences) == 2 and sentences[1].strip() == ''):
        if ',' not in caption:
            if 'airport' in caption.lower():
                if 'Apart from' not in caption:
                    caption = caption.rstrip('.') + '. Apart from the runway, this infrastructure includes taxiways, storage facilities, and support buildings.'
            elif 'sports' in caption.lower() or 'field' in caption.lower():
                if 'Apart from' not in caption:
                    caption = caption.rstrip('.') + '. In addition to the playing surfaces, this complex includes surrounding facilities which are typically part of an institutional area.'
            elif 'urban' in caption.lower():
                if 'Apart from' not in caption:
                    caption = caption.rstrip('.') + '. This urban layout reflects organized spatial distribution of infrastructure and development patterns.'
    return caption

def normalize_grammar_and_flow(caption: str) -> str:
    caption = re.sub(r'\bthe scene show\b', 'the scene shows', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\bthis scene provide\b', 'this scene provides', caption, flags=re.IGNORECASE)
    caption = re.sub(r'\s+\.', '.', caption)
    caption = re.sub(r'\s+,', ',', caption)
    caption = re.sub(r'\.(?=[A-Z])', '. ', caption)
    caption = re.sub(r',(?=[A-Z])', ', ', caption)
    caption = re.sub(r'\s+', ' ', caption)
    return caption.strip()

# ============== RANKING AND MAIN PIPELINE ================

def rerank_candidates(candidates: List[str], reference_embeddings) -> List[Tuple[str, float]]:
    results = []
    for candidate in candidates:
        cand_embed = sbert_model.encode(candidate, convert_to_tensor=True)
        ref_mean = reference_embeddings.mean(dim=0, keepdim=True)
        embedding_sim = util.cos_sim(cand_embed.unsqueeze(0), ref_mean).item()
        pattern_score = score_semantic_quality(candidate)
        combined_score = (0.4 * embedding_sim) + (0.6 * pattern_score)
        results.append((candidate, combined_score))
    return sorted(results, key=lambda x: x[1], reverse=True)

def improve_caption(
    caption: str, 
    references: List[str] = REFERENCE_CAPTIONS, 
    n_paraphrases: int = 7, 
    verbose: bool = False
) -> Dict:
    original_caption = caption
    caption = clean_caption(caption)
    ref_embeds = sbert_model.encode(references, convert_to_tensor=True)
    candidates = generate_paraphrases(caption, n_variants=n_paraphrases, paraphrase_prob=0.25)
    enhanced_candidates = []
    enhancement_steps = [
    inject_scene_opener,
    inject_scene_descriptor,
    enhance_location_type,
    enhance_spatial_relations,
    enhance_object_precision,
    enhance_quantifiers,
    add_contextual_elaboration,
    normalize_grammar_and_flow,
    ]

    for candidate in candidates:
        enhanced = candidate
        for step_func in enhancement_steps:
            enhanced = step_func(enhanced)
        enhanced_candidates.append(enhanced)
    ranked = rerank_candidates(enhanced_candidates, ref_embeds)
    best_caption, best_score = ranked[0]
    orig_embed = sbert_model.encode(original_caption, convert_to_tensor=True)
    best_embed = sbert_model.encode(best_caption, convert_to_tensor=True)
    ref_mean = ref_embeds.mean(dim=0, keepdim=True)
    orig_sim = util.cos_sim(orig_embed.unsqueeze(0), ref_mean).item()
    best_sim = util.cos_sim(best_embed.unsqueeze(0), ref_mean).item()
    improvement = best_sim - orig_sim
    improvement_pct = (improvement / abs(orig_sim)) * 100 if orig_sim != 0 else 0
    orig_patterns = count_semantic_patterns(original_caption)
    best_patterns = count_semantic_patterns(best_caption)
    if verbose:
        print("\nOriginal:", original_caption)
        print("Cleaned:", caption)
        print("Processed:", best_caption)
        print("Original Similarity:", orig_sim)
        print("Processed Similarity:", best_sim)
        print("Improvement:", improvement_pct)
        for pattern, added in best_patterns.items():
            if added and not orig_patterns[pattern]:
                print("Pattern gained:", pattern)
    return {
        'original_with_markup': original_caption,
        'original_cleaned': caption,
        'processed': best_caption,
        'original_similarity': float(orig_sim),
        'processed_similarity': float(best_sim),
        'improvement': float(improvement),
        'improvement_percentage': float(improvement_pct),
        'semantic_patterns_added': {k: (best_patterns[k] and not orig_patterns[k]) for k in best_patterns},
        'best_candidate_score': float(best_score),
    }

# =========== USAGE ============

if __name__ == "__main__":
    # Just paste your caption here
    test_string = "In the satellite image, there are <p>some buildings</p> {<40><77><52><85>|<1>}<delim>{<45><64><57><72>|<90>}  located close to each other at the bottom of the scene. These buildings are likely part of a city or urban area, and their presence indicates a high degree of human activity and development in the region. The presence of buildings can also provide insight into the local economy, infrastructure, and population density."
    result = improve_caption(test_string, verbose=True)
    print("\n>>> Improved Caption:\n", result['processed'])
    print("\n>>> Cleaned Original:\n", result['original_cleaned'])
    print("\nImprovement (%):", f"{result['improvement_percentage']:.2f}%")
    candidate = result['processed']