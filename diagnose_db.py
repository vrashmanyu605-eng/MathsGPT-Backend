
import psycopg2
from app.training import create_embeddings, get_db_connection

def diagnose_db_issue():
    print("Diagnosing DB Issue...")
    
    # 1. Check Table Structure
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns
                WHERE table_name = 'documents';
            """)
            columns = cur.fetchall()
            print("\nTable Structure:")
            for col in columns:
                print(col)
                
            # Check vector dimensions if possible (harder to check directly from info schema for vector dims in some versions)
            # Try to insert a dummy record with new embedding size (768) and see if it fails
            
    except Exception as e:
        print(f"Error checking DB structure: {e}")
    finally:
        conn.close()

    # 2. Try Insertion with Logging
    print("\nAttempting Insertion...")
    try:
        # Using a dummy text that is long enough to have content
        create_embeddings("Diagnosis test content", filename="diagnosis_test", source="diagnosis")
    except Exception as e:
        print(f"Insertion failed with error: {e}")

if __name__ == "__main__":
    diagnose_db_issue()
