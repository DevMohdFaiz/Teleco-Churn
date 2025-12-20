"""
MINIMAL FASTAPI - Debugging Version
====================================
Let's test if basic FastAPI works first
"""

from fastapi import FastAPI
import uvicorn

# Create app
app = FastAPI()

# Simple endpoints
@app.get("/")
def home():
    return {"message": "API is working!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/test")
def test():
    return {"test": "This endpoint works!"}

# Run
if __name__ == "__main__":
    print("="*60)
    print("🔍 DEBUGGING FASTAPI")
    print("="*60)
    print("Testing these endpoints:")
    print("  - http://localhost:8000/")
    print("  - http://localhost:8000/health")
    print("  - http://localhost:8000/test")
    print("  - http://localhost:8000/docs")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)