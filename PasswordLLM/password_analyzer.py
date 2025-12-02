"""
Password Analyzer
Checks passwords against common lists, calculates entropy, detects patterns, and generates secure passwords.
"""
import re
import os
import math
import secrets
import string
from typing import Dict, List, Tuple, Optional
from collections import Counter


class PasswordAnalyzer:
    def __init__(self, password_lists_path: str = None):
        if password_lists_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            password_lists_path = os.path.join(current_dir, "data", "common_password")
        
        self.password_lists_path = password_lists_path
        self.common_passwords = self._load_password_lists()
        
        self.keyboard_rows = [
            "qwertyuiop",
            "asdfghjkl",
            "zxcvbnm",
            "1234567890"
        ]
        
        self.keyboard_diagonals = [
            "qaz", "wsx", "edc", "rfv", "tgb", "yhn", "ujm", "ik", "ol", "p",
            "zaq", "xsw", "cde", "vfr", "bgt", "nhy", "mju", "ki", "lo", "p",
            "1qaz", "2wsx", "3edc", "4rfv", "5tgb", "6yhn", "7ujm", "8ik", "9ol", "0p"
        ]
        
        self.common_words = [
            'password', 'admin', 'welcome', 'login', 'master', 'letmein',
            'qwerty', 'dragon', 'baseball', 'football', 'monkey', 'shadow',
            'superman', 'batman', 'trustno1', 'iloveyou', 'princess', 'welcome',
            'sunshine', 'password1', 'password123', 'admin123', 'root', 'toor',
            'pass', 'passw0rd', 'p@ssw0rd', 'welcome123', 'letmein123'
        ]
        
        self.leet_substitutions = {
            'a': ['4', '@'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['5', '$'],
            't': ['7'],
            'l': ['1'],
            'z': ['2']
        }
        
        self.common_years = [str(year) for year in range(1900, 2030)]
        self.common_suffixes = ['123', '1234', '12345', '123456', '!', '!!', '!@#', '2024', '2025']
    
    def _load_password_lists(self) -> set:
        """Load common passwords from text files"""
        passwords = set()
        if not os.path.exists(self.password_lists_path):
            print(f"[WARNING] Password lists path not found: {self.password_lists_path}")
            return passwords
        
        for filename in os.listdir(self.password_lists_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.password_lists_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            password = line.strip().lower()
                            if password and len(password) >= 3:
                                passwords.add(password)
                except Exception as e:
                    print(f"[WARNING] Could not load {filename}: {e}")
        
        return passwords
    
    def _normalize_for_comparison(self, password: str) -> List[str]:
        """
        Generate normalized variants of password for comparison.
        Returns list of variants to check against common passwords.
        """
        variants = set()
        password_lower = password.lower()
        
        variants.add(password_lower)
        
        for suffix in self.common_suffixes:
            if password_lower.endswith(suffix):
                variants.add(password_lower[:-len(suffix)])
        
        for year in self.common_years:
            if password_lower.endswith(year):
                variants.add(password_lower[:-len(year)])
            if password_lower.startswith(year):
                variants.add(password_lower[len(year):])
        
        leet_normalized = password_lower
        for char, replacements in self.leet_substitutions.items():
            for replacement in replacements:
                leet_normalized = leet_normalized.replace(replacement, char)
        if leet_normalized != password_lower:
            variants.add(leet_normalized)
        
        stripped = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', password_lower)
        if stripped and stripped != password_lower:
            variants.add(stripped)
        
        return list(variants)
    
    def check_common_password(self, password: str) -> Tuple[bool, Optional[str]]:
        """
        Check if password is in common password lists.
        Also checks variants (leet, suffixes removed, etc.)
        """
        variants = self._normalize_for_comparison(password)
        
        for variant in variants:
            if variant in self.common_passwords:
                if variant == password.lower():
                    return True, "Found in common password database"
                else:
                    return True, f"Variant '{variant}' found in common password database"
        
        return False, None
    
    def _detect_keyboard_walk(self, password: str, min_length: int = 3) -> Tuple[bool, Optional[str]]:
        """
        Detect keyboard walks including:
        - Horizontal rows (left-right, right-left)
        - Vertical columns
        - Diagonals
        - Offset patterns
        """
        password_lower = password.lower()
        
        # Check horizontal rows (forward and backward)
        for row in self.keyboard_rows:
            # Forward
            for i in range(len(row) - min_length + 1):
                pattern = row[i:i+min_length+1]
                if pattern in password_lower:
                    return True, f"Contains keyboard walk: {pattern}"
            # Backward
            row_reverse = row[::-1]
            for i in range(len(row_reverse) - min_length + 1):
                pattern = row_reverse[i:i+min_length+1]
                if pattern in password_lower:
                    return True, f"Contains keyboard walk (reverse): {pattern}"
        
        # Check diagonals
        for diag in self.keyboard_diagonals:
            if len(diag) >= min_length and diag in password_lower:
                return True, f"Contains keyboard diagonal: {diag}"
        
        # Check vertical columns (adjacent keys vertically)
        # Q-A-Z, W-S-X, E-D-C, etc.
        vertical_patterns = [
            "qaz", "wsx", "edc", "rfv", "tgb", "yhn", "ujm", "ik", "ol", "p",
            "zaq", "xsw", "cde", "vfr", "bgt", "nhy", "mju", "ki", "lo"
        ]
        for pattern in vertical_patterns:
            if len(pattern) >= min_length and pattern in password_lower:
                return True, f"Contains vertical keyboard pattern: {pattern}"
        
        return False, None
    
    def detect_keyboard_pattern(self, password: str) -> Tuple[bool, Optional[str]]:
        """Detect keyboard patterns using comprehensive walk detection"""
        return self._detect_keyboard_walk(password, min_length=3)
    
    def detect_sequential_pattern(self, password: str) -> Tuple[bool, Optional[str]]:
        # Check for ascending sequential numbers
        if re.search(r'012|123|234|345|456|567|678|789|890', password):
            return True, "Contains ascending sequential numbers"
        
        # Check for descending sequential numbers
        if re.search(r'210|321|432|543|654|765|876|987|098', password):
            return True, "Contains descending sequential numbers"
        
        # Check for ascending sequential letters
        ascending_letters = r'abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz'
        if re.search(ascending_letters, password.lower()):
            return True, "Contains ascending sequential letters"
        
        # Check for descending sequential letters
        descending_letters = r'cba|dcb|edc|fed|gfe|hgf|ihg|jih|kji|lkj|mlk|nml|onm|pon|qpo|rqp|srq|tsr|uts|vut|wvu|xwv|yxw|zyx'
        if re.search(descending_letters, password.lower()):
            return True, "Contains descending sequential letters"
        
        # Check for repeated characters (4+ same character)
        if re.search(r'(.)\1{3,}', password):
            return True, "Contains repeated characters (4+ same character)"
        
        return False, None
    
    def _detect_leet_variant(self, word: str, password: str) -> bool:
        """Check if password contains a leetspeak variant of a dictionary word"""
        password_lower = password.lower()
        
        if word in password_lower:
            return True
        
        def generate_leet_variants(text: str, depth: int = 0, max_depth: int = 2) -> List[str]:
            """Recursively generate leetspeak variants"""
            if depth > max_depth or not text:
                return [text]
            
            variants = []
            for i, char in enumerate(text):
                if char in self.leet_substitutions:
                    for replacement in self.leet_substitutions[char]:
                        new_text = text[:i] + replacement + text[i+1:]
                        variants.extend(generate_leet_variants(new_text, depth + 1, max_depth))
            
            if not variants:
                return [text]
            return variants
        
        leet_variants = generate_leet_variants(word)
        for variant in leet_variants:
            if variant in password_lower:
                return True
        
        for suffix in self.common_suffixes:
            if (word + suffix) in password_lower or (variant + suffix) in password_lower:
                return True
        
        return False
    
    def detect_dictionary_word(self, password: str) -> Tuple[bool, Optional[str]]:
        """
        Check if password contains common dictionary words.
        Detects leetspeak variants, mangling, and common suffixes.
        """
        password_lower = password.lower()
        
        for word in self.common_words:
            if self._detect_leet_variant(word, password):
                if word in password_lower:
                    return True, f"Contains common word: {word}"
                else:
                    leet_variants = []
                    for char in word:
                        if char in self.leet_substitutions:
                            for replacement in self.leet_substitutions[char]:
                                leet_variants.append(word.replace(char, replacement))
                    for variant in leet_variants:
                        if variant in password_lower:
                            return True, f"Contains leetspeak variant of '{word}': {variant}"
                    return True, f"Contains variant of common word: {word}"
        
        for word in self.common_words:
            if re.search(rf'\b{re.escape(word)}\d+', password_lower):
                return True, f"Contains common word with digits: {word}###"
            for year in self.common_years[:5]:  
                if word + year in password_lower:
                    return True, f"Contains common word with year: {word}{year}"
            if re.search(rf'\b{re.escape(word)}[!@#$%^&*]+', password_lower):
                return True, f"Contains common word with special chars: {word}!"
        
        return False, None
    
    def calculate_entropy(self, password: str) -> float:
        if not password:
            return 0.0
        
        char_counts = Counter(password)
        length = len(password)
        
        # Calculate Shannon entropy: -Σ(p(x) * log2(p(x)))
        entropy = 0.0
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        total_entropy = entropy * length
        
        return round(total_entropy, 2)
    
    def _calculate_charset_size(self, password: str) -> int:
        """
        Calculate actual charset size based on unique characters present.
        Handles Unicode properly.
        """
        unique_chars = set(password)
        charset_size = len(unique_chars)
        return charset_size
    
    def analyze_character_variety(self, password: str) -> Dict[str, int]:
        """Analyze character variety in password"""
        # Count Unicode-aware
        lowercase_count = len(re.findall(r'[a-z]', password))
        uppercase_count = len(re.findall(r'[A-Z]', password))
        digits_count = len(re.findall(r'[0-9]', password))
        # Count all non-alphanumeric (including Unicode)
        special_count = len(re.findall(r'[^a-zA-Z0-9]', password))
        
        return {
            'length': len(password),
            'lowercase': lowercase_count,
            'uppercase': uppercase_count,
            'digits': digits_count,
            'special': special_count,
            'unique_chars': len(set(password)),
            'charset_size': self._calculate_charset_size(password)
        }
    
    def calculate_strength_score(self, password: str, analysis: Dict) -> int:
        """
        Calculate password strength score (1-10).
        Based on entropy, length, character variety, and pattern penalties.
        """
        score = 0
        length = len(password)
        variety = analysis['character_variety']
        entropy = analysis['entropy']
        
        # Length scoring (more granular, max 3 points)
        if length >= 20:
            score += 3
        elif length >= 16:
            score += 2.5  # Better granularity
        elif length >= 12:
            score += 1.5
        elif length >= 8:
            score += 0.5
        else:
            score -= 2  # Very short passwords penalized
        
        # Character variety (max 3 points)
        char_types = sum([
            variety['lowercase'] > 0,
            variety['uppercase'] > 0,
            variety['digits'] > 0,
            variety['special'] > 0
        ])
        score += min(char_types * 0.75, 3)  # More gradual increase
        
        # Entropy scoring (based on actual Shannon entropy, max 2.5 points)
        # Using lower thresholds because actual entropy is usually lower than theoretical
        if entropy >= 50:
            score += 2.5
        elif entropy >= 40:
            score += 2
        elif entropy >= 30:
            score += 1
        elif entropy >= 20:
            score += 0.5
        # Very low entropy gets penalty
        elif entropy < 10:
            score -= 1
        
        # Uniqueness ratio bonus (max 1 point)
        uniqueness_ratio = variety['unique_chars'] / length if length > 0 else 0
        if uniqueness_ratio > 0.9:
            score += 1
        elif uniqueness_ratio > 0.8:
            score += 0.5
        
        # Penalties (more severe for common passwords)
        if analysis['is_common']:
            score = max(1, score - 6)  # Very heavy penalty
        if analysis['has_keyboard_pattern']:
            score = max(1, score - 3)
        if analysis['has_sequential_pattern']:
            score = max(1, score - 2.5)
        if analysis['has_dictionary_word']:
            score = max(1, score - 3)  # Increased penalty
        
        # Convert to integer and clamp to 1-10
        score = int(round(score))
        return max(1, min(10, score))
    
    def analyze(self, password: str) -> Dict:
        """Perform comprehensive password analysis"""
        if not password:
            return {
                'error': 'No password provided'
            }
        
        # Basic checks
        is_common, common_reason = self.check_common_password(password)
        has_keyboard, keyboard_reason = self.detect_keyboard_pattern(password)
        has_sequential, sequential_reason = self.detect_sequential_pattern(password)
        has_dict_word, dict_reason = self.detect_dictionary_word(password)
        
        # Metrics
        character_variety = self.analyze_character_variety(password)
        entropy = self.calculate_entropy(password)
        
        # Compile analysis
        analysis = {
            'password': password,
            'length': len(password),
            'is_common': is_common,
            'common_reason': common_reason,
            'has_keyboard_pattern': has_keyboard,
            'keyboard_reason': keyboard_reason,
            'has_sequential_pattern': has_sequential,
            'sequential_reason': sequential_reason,
            'has_dictionary_word': has_dict_word,
            'dictionary_reason': dict_reason,
            'character_variety': character_variety,
            'entropy': entropy,
            'strength_score': 0  # Will be calculated
        }
        
        # Calculate final score
        analysis['strength_score'] = self.calculate_strength_score(password, analysis)
        
        return analysis
    
    def format_analysis_report(self, analysis: Dict) -> str:
        """Format analysis as a readable report"""
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
        
        report = []
        report.append(f"Password: {analysis['password']}")
        report.append(f"Length: {analysis['length']} characters")
        report.append(f"Security Score: {analysis['strength_score']}/10")
        report.append(f"Entropy: {analysis['entropy']} bits (Shannon entropy)")
        report.append("")
        
        # Character breakdown
        var = analysis['character_variety']
        report.append("Character Breakdown:")
        report.append(f"  - Lowercase: {var['lowercase']}")
        report.append(f"  - Uppercase: {var['uppercase']}")
        report.append(f"  - Digits: {var['digits']}")
        report.append(f"  - Special: {var['special']}")
        report.append(f"  - Unique characters: {var['unique_chars']}")
        report.append(f"  - Charset size: {var['charset_size']}")
        report.append("")
        
        # Issues found
        issues = []
        if analysis['is_common']:
            issues.append(f"{analysis['common_reason']}")
        if analysis['has_keyboard_pattern']:
            issues.append(f"{analysis['keyboard_reason']}")
        if analysis['has_sequential_pattern']:
            issues.append(f"{analysis['sequential_reason']}")
        if analysis['has_dictionary_word']:
            issues.append(f"{analysis['dictionary_reason']}")
        
        if issues:
            report.append("Security Issues Found:")
            for issue in issues:
                report.append(f"  {issue}")
        else:
            report.append("No obvious security issues detected")
        
        return "\n".join(report)
    
    def generate_password(self, length: int = 20, min_length: int = 12, max_length: int = 20) -> str:
        """
        Generate a secure random password.
        
        Args:
            length: Desired password length (default 20)
            min_length: Minimum length (default 12)
            max_length: Maximum length (default 20)
            
        Returns:
            Generated password string
        """
        if isinstance(length, str):
            try:
                length = int(length)
            except:
                length = 20
        
        if length < min_length:
            length = min_length
        if length > max_length:
            length = max_length
        
        if length < 6:
            length = 6
        
        # Character sets
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Use variable distribution instead of fixed length//4 to avoid patterns
        rng = secrets.SystemRandom()
        
        # For short passwords, ensure at least 1 of each type; for longer, at least 2
        min_per_type = 1 if length < 8 else 2
        max_per_type = max(1, length // 3)  # Up to 1/3 of length per type
        
        password_chars = []
        
        # Add variable amounts of each type
        # For very short passwords, ensure at least 1 of each type
        if length < 8:
            # For short passwords, ensure at least 1 of each type
            lowercase_count = max(1, rng.randint(1, min(max_per_type, length // 2)))
            uppercase_count = max(1, rng.randint(1, min(max_per_type, length // 2)))
            digits_count = max(1, rng.randint(1, min(max_per_type, length // 2)))
            special_count = max(1, rng.randint(1, min(max_per_type, length // 2)))
        else:
            lowercase_count = rng.randint(min_per_type, min(max_per_type, length // 4 + 1))
            uppercase_count = rng.randint(min_per_type, min(max_per_type, length // 4 + 1))
            digits_count = rng.randint(min_per_type, min(max_per_type, length // 4 + 1))
            special_count = rng.randint(min_per_type, min(max_per_type, length // 4 + 1))
        
        # Add characters
        for _ in range(lowercase_count):
            password_chars.append(secrets.choice(lowercase))
        for _ in range(uppercase_count):
            password_chars.append(secrets.choice(uppercase))
        for _ in range(digits_count):
            password_chars.append(secrets.choice(digits))
        for _ in range(special_count):
            password_chars.append(secrets.choice(special))
        
        # Fill remaining slots randomly from all character sets
        all_chars = lowercase + uppercase + digits + special
        remaining = length - len(password_chars)
        for _ in range(remaining):
            password_chars.append(secrets.choice(all_chars))
        
        # Shuffle a few times to mix it up
        for _ in range(5):
            rng.shuffle(password_chars)
        
        password = ''.join(password_chars)
        
        # Verify password meets criteria
        # Short passwords can't get 9+, so target 6 instead
        # Longer passwords should aim for 9-10
        if length < 8:
            target_score = 6  
        else:
            target_score = 9  
        max_attempts = 10  
        
        for attempt in range(max_attempts):
            test_analysis = self.analyze(password)
            score = test_analysis.get('strength_score', 0)
            if score >= target_score:
                return password
            
            password_chars = []
            
            if length < 8:
                target_per_type = 1  
            else:
                target_per_type = max(2, length // 5)  
            
            for _ in range(target_per_type):
                password_chars.append(secrets.choice(lowercase))
            for _ in range(target_per_type):
                password_chars.append(secrets.choice(uppercase))
            for _ in range(target_per_type):
                password_chars.append(secrets.choice(digits))
            for _ in range(target_per_type):
                password_chars.append(secrets.choice(special))
            
            remaining = length - len(password_chars)
            for _ in range(remaining):
                password_chars.append(secrets.choice(all_chars))
            
            for _ in range(5):
                rng.shuffle(password_chars)
            password = ''.join(password_chars)
        
        return password

