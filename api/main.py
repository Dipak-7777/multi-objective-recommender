from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from src.api.service import RecommenderService

app = FastAPI(title="Multi-Objective Recommender API")

# Initialize the service as a singleton
service = RecommenderService()

class RecRequest(BaseModel):
    user_id: str
    n: int = 10
    mode: str = "multi_objective"

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Objective Recommender API. Use /recommend endpoint."}

@app.post("/recommend")
def recommend(request: RecRequest):
    try:
        recs = service.get_recommendations(
            user_id=request.user_id,
            n=request.n,
            mode=request.mode
        )
        return {
            "user_id": request.user_id,
            "recommendations": recs,
            "mode": request.mode,
            "count": len(recs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)
