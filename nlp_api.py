import json
import logging
import os
import tempfile
from typing import Dict, Any

import spacy
from dotenv import load_dotenv


load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


nlp_model = None
model_loaded = False


async def load_model_from_s3():
    """Load pre-trained model from S3"""
    import boto3
    import os
    import tempfile
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
    bucket_name = "toxic-chat-classifier"
    model_prefix = "toxic_chat_model_full/"
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "toxic_chat_model_full")
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # List and download all files in the model folder
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=model_prefix)
        
        if 'Contents' not in response:
            raise Exception(f"No files found in {model_prefix}")
        
        for obj in response['Contents']:
            key = obj['Key']
            if not key.endswith('/'):  # Skip folder entries
                # Remove prefix to get relative path
                relative_path = key.replace(model_prefix, '')
                local_file_path = os.path.join(model_path, relative_path)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download file
                s3_client.download_file(bucket_name, key, local_file_path)
        
        # Load the SpaCy model
        nlp = spacy.load(model_path)
        return nlp
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


async def load_model():
    global nlp_model, model_loaded
    
    if model_loaded and nlp_model is not None:
        print("Using cached model")
        return nlp_model  # Use cached version
    
    try:
        # Check if model already exists in /tmp (warm start optimization)
        model_path = "/tmp/toxic_chat_model_full"
        if not os.path.exists(os.path.join(model_path, "config.cfg")):
            print("Model not found locally, downloading from S3...")
            nlp_model = await load_model_from_s3()
        else:
            print("Found existing model in /tmp")
        
        model_loaded = True
        print("Model loaded successfully!")
        
        return nlp_model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


async def lambda_handler(event, context):
    """Main Lambda handler function"""
    
    try:
        # Parse the incoming request
        if 'body' in event:
            
            body = json.loads(event['body'])
    
        else:
            # Direct invocation format
            body = event           

        if 'text' not in body:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Missing required field: text'
                })
            }
        
        text = body['text']
        
        # Load model (cached after first call)
        await load_model()
        
        # Make prediction
        result = await predict_toxicity(text)
        
        # Return successful response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        print(f"Lambda error: {e}")
        
        # Return error response
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error'
            })
        }




async def predict_toxicity(text: str) -> Dict[str, Any]:
    """Predict if text is toxic"""

    global nlp_model

    if nlp_model is None:
        raise Exception("Model not loaded")
    
    
    try:
        doc = nlp_model(text)
        toxic_score = doc.cats["TOXIC"]
        not_toxic_score = doc.cats["NOT_TOXIC"]
        is_toxic = toxic_score >= 0.5
        
        return {
            "text": text,
            "is_toxic": is_toxic,
            "toxic_score": float(toxic_score),
            "not_toxic_score": float(not_toxic_score),
            "confidence": float(max(toxic_score, not_toxic_score))
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise


if __name__ == "__main__":
    test_event = {
        "text": "fuck you."
    }
    
    import asyncio
    result = asyncio.run(lambda_handler(test_event, None))
    print(json.dumps(result, indent=2))
    #uvicorn.run(app, host="0.0.0.0", port=8000)