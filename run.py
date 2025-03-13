from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        port=8000,
        reload=False
    )
