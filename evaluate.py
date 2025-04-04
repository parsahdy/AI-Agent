import json
import os
import pandas as pd
from tqdm import tqdm
import traceback
from llm_connector import get_llm
from knowledge_base import load_vector_store
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def load_model():
    """Load the LLM model and QA chain"""
    try:
        llm = get_llm("ollama")
        vector_store = load_vector_store()
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        
        return llm, qa_chain
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        return None, None

def explore_directory(directory):
    """
    Explore the directory to find JSON files that could contain Spider test data
    Returns a list of potential test data files
    """
    potential_files = []
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return potential_files
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith('.json') and os.path.isfile(file_path):
            # Check if file is likely to contain Spider format data
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Spider files should be lists of dictionaries
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # Check for common Spider fields
                    if 'question' in data[0] or 'utterance' in data[0]:
                        potential_files.append((file, file_path))
            except:
                # Skip files that can't be parsed
                pass
    
    return potential_files

def load_spider_test_data(spider_data_dir):
    """Load test data from the Spider dataset format with more robust error handling"""
    
    test_data = []
    
    # First, try to find suitable JSON files
    print(f"Exploring directory: {spider_data_dir}")
    potential_files = explore_directory(spider_data_dir)
    
    if not potential_files:
        print("No potential Spider test files found.")
        return test_data
    
    print("Found potential Spider test data files:")
    for i, (filename, filepath) in enumerate(potential_files):
        print(f"{i+1}. {filename}")
    
    # Let the user choose which file to use
    try:
        choice = int(input("Enter the number of the file to use for evaluation: ")) - 1
        if 0 <= choice < len(potential_files):
            file_name, file_path = potential_files[choice]
        else:
            print("Invalid choice, using first file.")
            file_name, file_path = potential_files[0]
    except:
        print("Invalid input, using first file.")
        file_name, file_path = potential_files[0]
    
    print(f"Loading data from: {file_name}")
    
    # Load the selected JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} items in the file.")
    
    # Print sample of data structure to understand format
    if len(data) > 0:
        print("Sample data structure:")
        print(json.dumps(data[0], indent=2))
        
        # Ask for field mapping
        print("\nPlease identify the field names in your data:")
        question_field = input("Field name for question (common: 'question', 'utterance'): ")
        database_field = input("Field name for database ID (common: 'db_id', 'database'): ")
        sql_field = input("Field name for SQL query (common: 'query', 'sql'): ")
        
        # Process data with user-specified fields
        for item in data:
            question = item.get(question_field, "")
            db_id = item.get(database_field, "")
            gold_query = item.get(sql_field, "")
            
            test_data.append({
                'question': question,
                'db_id': db_id,
                'gold_query': gold_query
            })
            
    return test_data

def evaluate_model(qa_chain, test_data, num_samples=None):
    """
    Evaluate the model using test data from Spider
    
    Args:
        qa_chain: The QA chain to evaluate
        test_data: List of test data dictionaries
        num_samples: Number of samples to evaluate (None for all)
    """
    if not test_data:
        print("No test data available for evaluation.")
        return pd.DataFrame()
        
    results = []
    
    # Limit number of samples if specified
    if num_samples is not None and num_samples < len(test_data):
        test_data = test_data[:num_samples]
    
    for item in tqdm(test_data, desc="Evaluating model"):
        question = item['question']
        gold_query = item['gold_query']
        db_id = item['db_id']
        
        # Add database context to the question
        question_with_context = f"Database: {db_id}\nQuestion: {question}"
        
        # Get model's response
        try:
            result = qa_chain({"question": question_with_context})
            model_answer = result["answer"]
            
            # Simple word overlap score
            score = calculate_text_overlap(model_answer, question)
            
            results.append({
                'question': question,
                'database': db_id,
                'gold_query': gold_query,
                'model_response': model_answer,
                'relevance_score': score
            })
            
        except Exception as e:
            print(f"Error processing question: {question}\nError: {str(e)}")
            results.append({
                'question': question,
                'database': db_id,
                'gold_query': gold_query,
                'model_response': "ERROR",
                'relevance_score': 0.0
            })
    
    return pd.DataFrame(results)

def calculate_text_overlap(response, question):
    """
    Calculate a simple text overlap score between response and question
    This measures how relevant the response is to the question
    """
    if not response or not question:
        return 0.0
    
    # Convert to lowercase and split into words
    response_words = set(str(response).lower().split())
    question_words = set(str(question).lower().split())
    
    # Remove common stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
                 "in", "on", "at", "to", "for", "with", "by", "about", "like", "from"}
    
    response_words = response_words - stop_words
    question_words = question_words - stop_words
    
    # Calculate overlap
    if not question_words:
        return 0.0
    
    intersection = response_words.intersection(question_words)
    return len(intersection) / len(question_words)

def analyze_results(results_df):
    """Analyze evaluation results"""
    if results_df.empty:
        print("No results to analyze.")
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    # Basic statistics
    avg_score = results_df['relevance_score'].mean() if 'relevance_score' in results_df.columns else 0
    errors = (results_df['model_response'] == "ERROR").sum() if 'model_response' in results_df.columns else 0
    
    # Group by database if available
    db_performance = pd.DataFrame()
    if 'database' in results_df.columns and 'relevance_score' in results_df.columns:
        db_performance = results_df.groupby('database')['relevance_score'].mean().reset_index()
        db_performance = db_performance.sort_values('relevance_score', ascending=False)
    
    # Find best and worst performing questions
    best_questions = pd.DataFrame()
    worst_questions = pd.DataFrame()
    if 'relevance_score' in results_df.columns:
        best_questions = results_df.sort_values('relevance_score', ascending=False).head(5)
        worst_questions = results_df.sort_values('relevance_score', ascending=True).head(5)
    
    # Summary metrics
    metrics = {
        'avg_relevance_score': avg_score,
        'total_samples': len(results_df),
        'error_count': errors,
        'error_percentage': errors / len(results_df) * 100 if len(results_df) > 0 else 0
    }
    
    return metrics, db_performance, best_questions, worst_questions

def main():
    # Path to your Spider dataset directory
    default_path = "C:\\Users\\Parsa\\Downloads\\spider_data"
    spider_data_dir = input(f"Enter path to Spider data directory [{default_path}]: ") or default_path
    
    # Load model
    print("Loading model...")
    llm, qa_chain = load_model()
    
    if not llm or not qa_chain:
        print("Failed to load the model. Exiting.")
        return
    
    # Load test data
    print("Loading Spider test data...")
    test_data = load_spider_test_data(spider_data_dir)
    print(f"Loaded {len(test_data)} test examples")
    
    if not test_data:
        print("No test data found. Exiting.")
        return
    
    # Ask user if they want to limit the number of samples
    use_limit = input("Do you want to limit the number of test samples? (y/n): ").lower() == 'y'
    num_samples = None
    if use_limit:
        try:
            num_samples = int(input(f"Enter number of samples (max {len(test_data)}): "))
        except ValueError:
            print("Invalid input, using all samples.")
    
    # Evaluate model
    print("Evaluating model...")
    results_df = evaluate_model(qa_chain, test_data, num_samples)
    
    if results_df.empty:
        print("No evaluation results generated. Exiting.")
        return
    
    # Analyze results
    metrics, db_performance, best_questions, worst_questions = analyze_results(results_df)
    
    # Print summary
    print("\n===== Evaluation Results =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    if not db_performance.empty:
        print("\n===== Database Performance =====")
        print(db_performance.to_string(index=False))
    
    if not best_questions.empty:
        print("\n===== Best Performing Questions =====")
        for idx, row in best_questions.iterrows():
            print(f"Q: {row['question']}")
            print(f"Score: {row['relevance_score']:.2f}")
            print(f"DB: {row['database']}")
            print("---")
    
    if not worst_questions.empty:
        print("\n===== Worst Performing Questions =====")
        for idx, row in worst_questions.iterrows():
            print(f"Q: {row['question']}")
            print(f"Score: {row['relevance_score']:.2f}")
            print(f"DB: {row['database']}")
            print("---")
    
    # Save detailed results
    results_df.to_csv("spider_evaluation_results.csv", index=False)
    print("\nDetailed results saved to spider_evaluation_results.csv")

if __name__ == "__main__":
    main()