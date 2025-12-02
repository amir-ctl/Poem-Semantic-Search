# =============================================================================
# inspect_db.py
# =============================================================================
# A simple utility to look inside our ChromaDB folders and list the names
# of the collections they contain.

import chromadb

# --- Configuration: Point to your database folders ---
DB_PATHS = {
    "ParsBERT DB": "./vector_databases/poetry_db",
    "MiniLM DB": "./vector_databases/poetry_db_minilm"
}

def inspect_databases():
    print("--- Inspecting ChromaDB Collections ---")
    for db_key, db_path in DB_PATHS.items():
        try:
            print(f"\nChecking database: '{db_key}' at path '{db_path}'...")
            client = chromadb.PersistentClient(path=db_path)
            collections = client.list_collections()
            
            if collections:
                print(f"✅ Found the following collections:")
                for collection in collections:
                    print(f"   - Name: '{collection.name}' | Item Count: {collection.count()}")
            else:
                print("🟡 No collections found in this database.")

        except Exception as e:
            print(f"❌ Error inspecting database at '{db_path}': {e}")
            print("   Please ensure the folder exists and is a valid ChromaDB directory.")

if __name__ == "__main__":
    inspect_databases()