"""
Initialize encryption key for NoteVault.
Run this once to generate the AES-256 key.
The key will be stored in ~/.notevault/encryption.key
"""

import sys
from crypto import generate_key, save_key, key_exists

def main():
    if key_exists():
        response = input("Encryption key already exists. Regenerate? (y/n): ").strip().lower()
        if response != 'y':
            print("Keeping existing key.")
            return
    
    print("Generating AES-256-GCM encryption key...")
    key = generate_key()
    save_key(key)
    print("✓ Encryption key generated and saved successfully!")
    print(f"  Key location: ~/.notevault/encryption.key")
    print(f"  Key size: 256 bits")

if __name__ == "__main__":
    main()
